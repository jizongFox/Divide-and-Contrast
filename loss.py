import random
from contextlib import contextmanager
from typing import Tuple

import matplotlib
import torch
from deepclustering2.configparser._utils import get_config  # noqa
from loguru import logger
from torch import Tensor, nn


@contextmanager
def switch_plt_backend(env="agg"):
    prev = matplotlib.get_backend()
    matplotlib.use(env, force=True)
    yield
    matplotlib.use(prev, force=True)


def is_normalized(feature: Tensor, dim=1):
    norms = feature.norm(dim=dim)
    return torch.allclose(norms, torch.ones_like(norms))


def exp_sim_temperature(proj_feat1: Tensor, proj_feat2: Tensor, t: float) -> Tuple[Tensor, Tensor]:
    projections = torch.cat([proj_feat1, proj_feat2], dim=0)
    sim_logits = torch.mm(projections, projections.t().contiguous()) / t
    max_value = sim_logits.max().detach()
    sim_logits -= max_value
    sim_exp = torch.exp(sim_logits)
    return sim_exp, sim_logits


class SupConLoss1(nn.Module):
    def __init__(self, temperature=0.07, exclude_other_pos=False):
        super().__init__()
        self._t = temperature
        self._exclude_pos = exclude_other_pos
        logger.info(f"initializing {self.__class__.__name__} with t: {self._t}, exclude_pos: {self._exclude_pos}")

    def forward(self, proj_feat1, proj_feat2, target=None, mask: Tensor = None, **kwargs):
        batch_size = proj_feat1.size(0)
        if mask is not None:
            assert mask.shape == torch.Size([batch_size, batch_size])
            pos_mask = mask == 1
            neg_mask = mask == 0

        elif target is not None:
            if isinstance(target, list):
                target = torch.Tensor(target).to(device=proj_feat2.device)
            mask = torch.eq(target[..., None], target[None, ...])

            pos_mask = mask == True
            neg_mask = mask == False
        else:
            # only postive masks are diagnal of the sim_matrix
            pos_mask = torch.eye(batch_size, dtype=torch.float, device=proj_feat2.device)  # SIMCLR
            neg_mask = 1 - pos_mask
        return self._forward(proj_feat1, proj_feat2, pos_mask.float(), neg_mask.float(), **kwargs)

    def _forward(self, proj_feat1, proj_feat2, pos_mask, neg_mask, **kwargs):
        """
        Here the proj_feat1 and proj_feat2 should share the same mask within and cross proj_feat1 and proj_feat2
        :param proj_feat1:
        :param proj_feat2:
        :return:
        """
        assert is_normalized(proj_feat1) and is_normalized(proj_feat2), f"features need to be normalized first"
        assert proj_feat1.shape == proj_feat2.shape, (proj_feat1.shape, proj_feat2.shape)

        batch_size = len(proj_feat1)
        unselect_diganal_mask = 1 - torch.eye(
            batch_size * 2, batch_size * 2, dtype=torch.float, device=proj_feat2.device
        )

        # upscale
        pos_mask = pos_mask.repeat(2, 2)
        neg_mask = neg_mask.repeat(2, 2)

        pos_mask *= unselect_diganal_mask
        neg_mask *= unselect_diganal_mask

        # 2n X 2n
        sim_exp, sim_logits = exp_sim_temperature(proj_feat1, proj_feat2, self._t)
        assert pos_mask.shape == sim_exp.shape == neg_mask.shape, (pos_mask.shape, sim_exp.shape, neg_mask.shape)

        # =============================================
        # in order to have a hook for further processing
        self.sim_exp = sim_exp
        self.sim_logits = sim_logits
        self.pos_mask = pos_mask
        self.neg_mask = neg_mask
        # ================= end =======================
        pos_count, neg_count = pos_mask.sum(1), neg_mask.sum(1)
        pos_sum = (sim_exp * pos_mask).sum(1, keepdim=True).repeat(1, batch_size * 2)
        neg_sum = (sim_exp * neg_mask).sum(1, keepdim=True).repeat(1, batch_size * 2)
        if self._exclude_pos:
            neg_ratio = neg_count.float() / (pos_count + neg_count).float()
            log_pos_div_sum_pos_neg = sim_logits - torch.log(
                sim_exp + neg_sum / (neg_ratio + 1e-4)[..., None].repeat(1, batch_size * 2) + 1e-16)
        else:
            log_pos_div_sum_pos_neg = sim_logits - torch.log(pos_sum + neg_sum + 1e-16)

        # over positive mask
        loss = (log_pos_div_sum_pos_neg * pos_mask).sum(1) / pos_count
        loss = -loss.mean()

        if torch.isnan(loss):
            raise RuntimeError(loss)
        return loss
