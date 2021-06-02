import argparse
import os
from itertools import chain

import torch
from deepclustering2.dataloader.sampler import InfiniteRandomSampler
from deepclustering2.meters2 import AverageValueMeter
from deepclustering2.models import ema_updater as EMA_Updater
from deepclustering2.schedulers.warmup_scheduler import GradualWarmupScheduler
from loguru import logger
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch_optimizer import RAdam
from tqdm import tqdm

from data import get_pretrain_dataset
from loss import SupConLoss1
from network import Model, Projector, detach_grad


def get_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--save_dir", required=True, type=str, help="save_dir")
    parser.add_argument("--max_epoch", type=int, default=100, help="max_epoch")
    parser.add_argument("--num_batches", type=int, default=500, help="num_batches")
    parser.add_argument("--lr", type=float, default=1e-3, help="lr")
    parser.add_argument("--batch_size", type=int, default=256, help="batch_size")
    parser.add_argument("--use-simclr", default=False, action="store_true")
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

optimizer = RAdam(chain(model.parameters(), projector.parameters()), lr=args.lr / 100, weight_decay=5e-5)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_epoch - 10, eta_min=1e-7)
scheduler = GradualWarmupScheduler(optimizer, multiplier=100, total_epoch=10, after_scheduler=scheduler)

if args.use_simclr:
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
            _, feature = model(image)
            _, feature_tf = teacher_model(image_tf)

            proj_feat, proj_feat_tf = projector(feature), teacher_projector(feature_tf)
            norm_proj_feat, norm_proj_feat_tf = F.normalize(proj_feat, dim=1), F.normalize(proj_feat_tf, dim=1)

            loss = criterion(norm_proj_feat, norm_proj_feat_tf, target=None)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if not args.use_simclr:
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
