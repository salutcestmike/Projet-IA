import numpy as np
import tensorflow as tf


class DiceCoefficient(tf.keras.metrics.Metric):
    """Dice sur la classe 1 (mitochondries). y_true: (B,H,W) entiers, y_pred: (B,H,W,2) softmax."""

    def __init__(self, name="dice", smooth=1e-6, **kwargs):
        super().__init__(name=name, **kwargs)
        self.smooth = smooth
        self.intersection = self.add_weight(name="intersection", initializer="zeros")
        self.total = self.add_weight(name="total", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        if len(y_true.shape) == 4:                      # (B,H,W,1) -> (B,H,W)
            y_true = tf.squeeze(y_true, -1)
        y_true = tf.cast(tf.equal(y_true, 1), tf.float32)
        y_pred = tf.cast(tf.equal(tf.argmax(y_pred, axis=-1), 1), tf.float32)

        self.intersection.assign_add(tf.reduce_sum(y_true * y_pred))
        self.total.assign_add(tf.reduce_sum(y_true) + tf.reduce_sum(y_pred))

    def result(self):
        return (2.0 * self.intersection + self.smooth) / (self.total + self.smooth)

    def reset_state(self):
        self.intersection.assign(0.0)
        self.total.assign(0.0)


def dice_coef(mask_pred, mask_ref):
    mask_pred = np.squeeze(np.asarray(mask_pred)) > 0
    mask_ref = np.squeeze(np.asarray(mask_ref)) > 0
    assert mask_pred.shape == mask_ref.shape, f"{mask_pred.shape} != {mask_ref.shape}"

    total = mask_pred.sum() + mask_ref.sum()
    if total == 0:
        return 1.0
    return 2.0 * np.logical_and(mask_pred, mask_ref).sum() / total

def calculate_metrics(y_pred, y_true):
    """
    Calcule l'IoU, la précision et le rappel pour une segmentation binaire.

    Parameters
    ----------
    y_true : array-like
        Masque de référence (ground truth).
    y_pred : array-like
        Masque prédit par le modèle.

    Returns
    -------
    dict
        IoU, précision et rappel.
    """
    # Convertir les masques en tableaux binaires
    y_true = np.asarray(y_true) > 0
    y_pred = np.asarray(y_pred) > 0

    # Calculer les vrais positifs, faux positifs et faux négatifs
    TP = np.sum(y_true & y_pred)
    FP = np.sum(~y_true & y_pred)
    FN = np.sum(y_true & ~y_pred)
    TN = np.sum(~y_true & ~y_pred)


    # Calculer les métriques
    iou = TP / (TP + FP + FN) if TP + FP + FN > 0 else 1.0
    precision = TP / (TP + FP) if TP + FP > 0 else 0.0
    recall = TP / (TP + FN) if TP + FN > 0 else 0.0
    dice = dice_coef(y_pred, y_true)

    return {
        "IoU": float(iou),
        "Precision": float(precision),
        "Recall": float(recall),
        "FPR" : float(FP / (FP + TN) if FP + TN > 0 else 0.0),
        "Dice": float(dice)
    }