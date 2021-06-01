import argparse
import os

import torch
from deepclustering2.dataloader.sampler import InfiniteRandomSampler
from deepclustering2.meters2 import AverageValueMeter
from deepclustering2.schedulers import GradualWarmupScheduler
from loguru import logger
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from data import get_train_datasets
from network import Model


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_checkpoint", default=None, type=str,
                        help="pretrained checkpoint trained by `pretrain.py`")
    parser.add_argument("--checkpoint", default=None, type=str, )
    parser.add_argument("--enable_grad_4_extractor", action="store_true", default=False)
    parser.add_argument("--save_dir", required=True, type=str, help="save_dir")
    parser.add_argument("--num_batches", type=int, default=200, help="batch_size")
    parser.add_argument("--batch_size", type=int, default=1024, help="batch_size")
    parser.add_argument("--max_epoch", type=int, default=500, help="max_epoch")
    parser.add_argument("--lr", type=float, default=0.01, help="lr")

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
                               sampler=InfiniteRandomSampler(tra_set, shuffle=True)))
test_loader = DataLoader(test_set, batch_size=256, shuffle=False, num_workers=16)

model = Model(input_dim=3, num_classes=10).cuda()
model = nn.DataParallel(model)

optimizer = Adam(model.parameters(), lr=args.lr / 100, weight_decay=5e-5)
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
    length = len(test_loader)
    indicator = tqdm(range(length))
    indicator.set_description_str(f"Validating Epoch {epoch: 3d}")
    loss_meter = AverageValueMeter()
    acc_meter = AverageValueMeter()
    for i, data in zip(indicator, test_loader):
        image, target = data
        image, target = image.cuda(), target.cuda()
        pred_logits, _ = model(image)
        loss = criterion(pred_logits, target)
        loss_meter.add(loss.item())
        acc_meter.add(torch.eq(pred_logits.max(1)[1], target).float().mean().cpu())
        indicator.set_postfix_str(
            f"loss: {loss_meter.summary()['mean']:.3f}, acc: {acc_meter.summary()['mean']:.3f}")
    logger.info(indicator.desc + "  " + indicator.postfix)
    writer.add_scalar("val/loss", loss_meter.summary()['mean'], global_step=epoch)
    writer.add_scalar("val/acc", acc_meter.summary()['mean'], global_step=epoch)

    return acc_meter.summary()['mean']


num_batches = args.num_batches
with model.module.set_grad(enable_fc=True, enable_extractor=args.enable_grad_4_extractor):
    for epoch in range(1, args.max_epoch):
        model.train()
        indicator = tqdm(range(num_batches))
        indicator.set_description_str(f"Training Epoch {epoch: 3d} lr:{optimizer.param_groups[0]['lr']:.3e}")
        loss_meter, acc_meter = AverageValueMeter(), AverageValueMeter()
        lr_meter = AverageValueMeter()
        lr_meter.add(optimizer.param_groups[0]['lr'])

        is_best = False
        for i, data in zip(indicator, train_loader):
            image, target = data
            image, target = image.cuda(), target.cuda()
            pred_logits, _ = model(image)
            loss = criterion(pred_logits, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_meter.add(loss.item())
            acc_meter.add(torch.eq(pred_logits.max(1)[1], target).float().mean().cpu())
            indicator.set_postfix_str(
                f"loss: {loss_meter.summary()['mean']:.3f}, acc: {acc_meter.summary()['mean']:.3f}")

        logger.info(indicator.desc + "  " + indicator.postfix)
        writer.add_scalar("train/loss", loss_meter.summary()['mean'], global_step=epoch)
        writer.add_scalar("train/acc", acc_meter.summary()['mean'], global_step=epoch)
        writer.add_scalar("train/lr", lr_meter.summary()['mean'], global_step=epoch)

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
