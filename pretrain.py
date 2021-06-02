import argparse
import os

import torch
from deepclustering2.dataloader.sampler import InfiniteRandomSampler
from deepclustering2.meters2 import AverageValueMeter
from deepclustering2.models import ema_updater as EMA_Updater
from deepclustering2.schedulers.warmup_scheduler import GradualWarmupScheduler
from loguru import logger
from torch import nn
from torch.cuda.amp import GradScaler, autocast
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from data import get_pretrain_dataset
from loss import SupConLoss1
from network import Model, Projector, detach_grad


def get_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    checkpoint_parser = parser.add_argument_group("checkpoint")
    checkpoint_parser.add_argument("--checkpoint", default=None, type=str, help="checkpoint trained by `train.py`")

    trainer_parser = parser.add_argument_group("trainer")
    trainer_parser.add_argument("--save_dir", required=True, type=str, help="save_dir")
    trainer_parser.add_argument("--num_batches", type=int, default=200, help="batch_size")
    trainer_parser.add_argument("--batch_size", type=int, default=1024, help="batch_size")
    trainer_parser.add_argument("--max_epoch", type=int, default=500, help="max_epoch")
    trainer_parser.add_argument("--lr", type=float, default=0.8, help="lr")

    contrast_parser = parser.add_argument_group("contrastive")
    contrast_parser.add_argument("--contrastive-name", choices=["moclr", "simclr"], default="simclr",
                                 help="contrastive name")

    mix_parser = parser.add_argument_group("mixed training")
    mix_parser.add_argument("--enable-scale", default=False, action="store_true")
    mix_parser.add_argument("--iters_to_accumulate", type=int, default=2, help="iterations to accumulate the gradient")

    args = parser.parse_args()
    return args


args = get_args()
save_dir = args.save_dir
logger.add(os.path.join(save_dir, "loguru.log"), level="TRACE")
logger.info(args)
writer = SummaryWriter(log_dir=os.path.join(save_dir, "tensorboard"))

tra_set = get_pretrain_dataset()
train_loader = iter(DataLoader(tra_set, batch_size=args.batch_size, num_workers=16,
                               sampler=InfiniteRandomSampler(tra_set, shuffle=True)))

model = Model(input_dim=3, num_classes=10).cuda()
model = nn.DataParallel(model)

projector = Projector(input_dim=model.module.feature_dim, hidden_dim=model.module.feature_dim, output_dim=128).cuda()
projector = nn.DataParallel(projector)

optimizer = torch.optim.SGD(model.parameters(), momentum=0.9, lr=args.lr / 100, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_epoch - 10, eta_min=1e-7)
scheduler = GradualWarmupScheduler(optimizer, multiplier=100, total_epoch=10, after_scheduler=scheduler)

if args.contrastive_name == "simclr":
    teacher_model = model
    teacher_projector = projector
    ema_updater1 = None
    ema_updater2 = None
else:
    teacher_model = Model(input_dim=3, num_classes=10).cuda()
    teacher_model = nn.DataParallel(teacher_model)
    teacher_model = detach_grad(teacher_model)
    teacher_projector = Projector(input_dim=model.module.feature_dim, hidden_dim=model.module.feature_dim,
                                  output_dim=128).cuda()
    teacher_projector = nn.DataParallel(teacher_projector)
    teacher_projector = detach_grad(teacher_projector)

    ema_updater1 = EMA_Updater()
    ema_updater2 = EMA_Updater()

criterion = SupConLoss1(temperature=0.07)
# pretrain
with model.module.set_grad(enable_fc=False, enable_extractor=True):
    scaler = GradScaler(enabled=args.enable_scale)

    for epoch in range(1, args.max_epoch):
        model.train()
        indicator = tqdm(range(args.num_batches))
        loss_meter = AverageValueMeter()
        lr_meter = AverageValueMeter()
        indicator.set_description_str(f"Pretrain Epoch {epoch:3d} lr:{optimizer.param_groups[0]['lr']}")
        lr_meter.add(optimizer.param_groups[0]['lr'])

        for i, data in zip(indicator, train_loader):
            (image, image_tf), target = data
            image, image_tf = image.cuda(), image_tf.cuda()

            with autocast(enabled=args.enable_scale):
                _, feature = model(image)
                _, feature_tf = teacher_model(image_tf)
                proj_feat, proj_feat_tf = projector(feature), teacher_projector(feature_tf)
                norm_proj_feat, norm_proj_feat_tf = F.normalize(proj_feat, dim=1), F.normalize(proj_feat_tf, dim=1)

                loss = criterion(norm_proj_feat, norm_proj_feat_tf, target=None)
            scaler.scale(loss).backward()
            if (i + 1) % args.iters_to_accumulate == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            if args.contrastive_name == "moclr":
                ema_updater1(teacher_model, model)
                ema_updater2(teacher_projector, projector)
            loss_meter.add(loss.item())
            indicator.set_postfix_str(f"loss: {loss_meter.summary()['mean']:.3f}")

        scheduler.step()
        logger.info(indicator.desc + indicator.postfix)
        writer.add_scalar("pre_train/loss", loss_meter.summary()['mean'], global_step=epoch)
        writer.add_scalar("pre_train/lr", lr_meter.summary()['mean'], global_step=epoch)

        checkpoint = {
            "model": model.state_dict(),
            "projector": projector.state_dict(),
            "teacher_model": teacher_model.state_dict(),
            "teacher_projector": teacher_projector.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict()
        }
        torch.save(checkpoint, os.path.join(save_dir, "pretrain.pth"))
