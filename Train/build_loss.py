import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from nnunetv2.training.loss.compound_losses import DC_and_CE_loss, DC_and_BCE_loss
from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
from nnunetv2.training.loss.dice import get_tp_fp_fn_tn, MemoryEfficientSoftDiceLoss
import numpy as np
import torch

def build_loss(batch_dice,has_regions,ignore_label,enable_deep_supervision,deep_supervision_scales,world_size):
    if world_size > 1:
        is_ddp = True
    else:
        is_ddp = False
        
    if has_regions:
        loss = DC_and_BCE_loss({},
                                {'batch_dice': batch_dice,
                                'do_bg': True, 'smooth': 1e-5, 'ddp': is_ddp},
                                use_ignore_label=ignore_label is not None,
                                dice_class=MemoryEfficientSoftDiceLoss)
    else:
        loss = DC_and_CE_loss({'batch_dice': batch_dice,
                                'smooth': 1e-5, 'do_bg': False, 'ddp': is_ddp}, {}, weight_ce=1, weight_dice=1,
                                ignore_label=ignore_label, dice_class=MemoryEfficientSoftDiceLoss)

    #if self._do_i_compile():
        #loss.dc = torch.compile(loss.dc)

    # we give each output a weight which decreases exponentially (division by 2) as the resolution decreases
    # this gives higher resolution outputs more weight in the loss

    if enable_deep_supervision:
        weights = torch.Tensor([1 / (2 ** i) for i in range(len(deep_supervision_scales))])
        if world_size > 1:# and not self._do_i_compile():
            # very strange and stupid interaction. DDP crashes and complains about unused parameters due to
            # weights[-1] = 0. Interestingly this crash doesn't happen with torch.compile enabled. Strange stuff.
            # Anywho, the simple fix is to set a very low weight to this.
            weights[-1] = 1e-6
        else:
            weights[-1] = 0

        # we don't use the lowest 2 outputs. Normalize weights so that they sum to 1
        weights = weights / weights.sum()
        # now wrap the loss
        loss = DeepSupervisionWrapper(loss, weights)

    return loss