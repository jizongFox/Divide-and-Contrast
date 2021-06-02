import argparse
import os

import torch
from deepclustering2.dataloader.sampler import InfiniteRandomSampler
from deepclustering2.meters2 import AverageValueMeter
from deepclustering2.schedulers import GradualWarmupScheduler
from loguru import logger
from torch import nn
from torch.backends import cudnn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from data import get_train_datasets
from network import Model
from utils import AverageMeter

cudnn.benchmark = True


def get_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    checkpoint_parser = parser.add_argument_group("checkpoint")
    checkpoint_parser.add_argument("--pretrained_checkpoint", default=None, type=str,
                                   help="pretrained checkpoint trained by `pretrain.py`")
    checkpoint_parser.add_argument("--checkpoint", default=None, type=str, help="checkpoint trained by `train.py`")

    gradient_parser = parser.add_argument_group("gradient")
    gradient_parser.add_argument("--enable_grad_4_extractor", action="store_true", default=False,
                                 help="enable gradient update for feature extractor")

    trainer_parser = parser.add_argument_group("trainer")
    trainer_parser.add_argument("--save_dir", required=True, type=str, help="save_dir")
    trainer_parser.add_argument("--num_batches", type=int, default=200, help="batch_size")
    trainer_parser.add_argument("--batch_size", type=int, default=1024, help="batch_size")
    trainer_parser.add_argument("--max_epoch", type=int, default=500, help="max_epoch")
    trainer_parser.add_argument("--lr", type=float, default=0.8, help="lr")

    mix_parser = parser.add_argument_group("mixed training")
    mix_parser.add_argument("--enable-scale", default=False, action="store_true")
    mix_parser.add_argument("--iters_to_accumulate", type=int, default=2, help="iterations to accumulate the gradient")

    args = parser.parse_args()

    if args.pretrained_checkpoint is None and args.enable_grad_4_extractor is False:
        raise RuntimeError("You should either provide a checkpoint path or enable extractor gradient.")
    return args


args = get_args()
save_dir = args.save_dir
# if os.path.exists(save_dir):
#     raise FileExistsError(save_dir)
logger.add(os.path.join(save_dir, "loguru.log"), level="TRACE")
logger.info(args)
writer = SummaryWriter(log_dir=os.path.join(save_dir, "tensorboard"))

tra_set, test_set = get_train_datasets()
train_loader = iter(DataLoader(tra_set, batch_size=args.batch_size, num_workers=16,
                               sampler=InfiniteRandomSampler(tra_set, shuffle=True), pin_memory=True))
test_loader = DataLoader(test_set, batch_size=256, shuffle=False, num_workers=16, pin_memory=True)

model = Model(input_dim=3, num_classes=10).cuda()
model = nn.DataParallel(model)

optimizer = torch.optim.SGD(model.parameters(), momentum=0.9, lr=args.lr / 100, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_epoch - 10, eta_min=1e-7)
scheduler = GradualWarmupScheduler(optimizer, multiplier=100, total_epoch=10, after_scheduler=scheduler)

best_score = 0

if args.pretrained_checkpoint:
    checkpoint = torch.load(args.pretrained_checkpoint, map_location="cuda")
    model.load_state_dict(checkpoint["model"])
    logger.info(f"loaded checkpoint from {args.pretrained_checkpoint}.")

if args.checkpoint:
    checkpoint = torch.load(args.checkpoint, map_location="cuda")
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    scheduler.load_state_dict(checkpoint["scheduler"])
    best_score = checkpoint["best_score"]
    logger.info(f"loaded checkpoint from {args.checkpoint}.")

criterion = nn.CrossEntropyLoss()


@torch.no_grad()
def val(epoch):
    model.eval()
    indicator = tqdm(test_loader)
    indicator.set_description_str(f"Validating Epoch {epoch: 3d}")
    loss_meter = AverageValueMeter()
    acc_meter = AverageValueMeter()
    true_acc_meter = AverageMeter()
    for i, data in enumerate(indicator):
        image, target = data
        image, target = image.cuda(), target.cuda()
        with autocast(enabled=args.enable_scale):
            pred_logits, _ = model(image)
            loss = criterion(pred_logits, target)
        loss_meter.add(loss.item())
        batch_acc_mean = torch.eq(pred_logits.max(1)[1], target).float().mean().cpu().item()
        acc_meter.add(batch_acc_mean)
        true_acc_meter.update(batch_acc_mean, n=image.shape[0])
        indicator.set_postfix_str(
            f"loss: {loss_meter.summary()['mean']:.3f}, acc: {acc_meter.summary()['mean']:.3f}, "
            f"true_acc: {true_acc_meter.avg:.3f}")
    logger.info(indicator.desc + "  " + indicator.postfix)
    writer.add_scalar("val/loss", loss_meter.summary()['mean'], global_step=epoch)
    writer.add_scalar("val/acc", acc_meter.summary()['mean'], global_step=epoch)
    writer.add_scalar("val/true_acc", true_acc_meter.avg, global_step=epoch)

    return acc_meter.summary()['mean']


num_batches = args.num_batches
with model.module.set_grad(enable_fc=True, enable_extractor=args.enable_grad_4_extractor):
    scaler = GradScaler(enabled=args.enable_scale)

    for epoch in range(1, args.max_epoch):
        model.train()
        indicator = tqdm(range(num_batches))
        indicator.set_description_str(f"Training Epoch {epoch: 3d} lr:{optimizer.param_groups[0]['lr']:.3e}")
        loss_meter, acc_meter, true_acc_meter = AverageValueMeter(), AverageValueMeter(), AverageMeter()
        lr_meter = AverageValueMeter()
        lr_meter.add(optimizer.param_groups[0]['lr'])

        is_best = False
        for i, data in zip(indicator, train_loader):
            image, target = data
            image, target = image.cuda(), target.cuda()
            with autocast(enabled=args.enable_scale):
                pred_logits, _ = model(image)
                loss = criterion(pred_logits, target)
            scaler.scale(loss).backward()
            if (i + 1) % args.iters_to_accumulate == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            loss_meter.add(loss.item())
            batch_acc_mean = torch.eq(pred_logits.max(1)[1], target).float().mean().item()
            acc_meter.add(batch_acc_mean)
            true_acc_meter.update(batch_acc_mean, n=image.shape[0])
            indicator.set_postfix_str(
                f"loss: {loss_meter.summary()['mean']:.3f}, acc: {acc_meter.summary()['mean']:.3f}, "
                f"true_acc: {true_acc_meter.avg:.3f}")

        logger.info(indicator.desc + "  " + indicator.postfix)
        writer.add_scalar("train/loss", loss_meter.summary()['mean'], global_step=epoch)
        writer.add_scalar("train/acc", acc_meter.summary()['mean'], global_step=epoch)
        writer.add_scalar("train/lr", lr_meter.summary()['mean'], global_step=epoch)
        writer.add_scalar("train/true_acc", true_acc_meter.avg, global_step=epoch)

        cur_score = val(epoch)
        if cur_score > best_score:
            best_score = cur_score
            is_best = True

        checkpoint = {
            "model": model.state_dict(),
            "best_score": best_score,
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict()
        }
        torch.save(checkpoint, os.path.join(save_dir, "train_last.pth"))
        if is_best:
            torch.save(checkpoint, os.path.join(save_dir, "train_best.pth"))
        scheduler.step()
    logger.info(f"the best score is: {best_score:.3f}")
