import numpy as np


def dice_coef(mask_pred, mask_ref):
    """
    Calcule le coefficient de Dice entre deux masques binaires.
    """
    mask_pred = np.asarray(mask_pred) > 0
    mask_ref = np.asarray(mask_ref) > 0

    intersection = np.logical_and(mask_pred, mask_ref).sum()

    total = mask_pred.sum() + mask_ref.sum()

    if total == 0:
        return 1.0

    return 2 * intersection / total