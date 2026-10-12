import tifffile
import numpy as np

original = tifffile.imread("./data/HCT116-0Gy-3j-2900X-0002.tif")
labels = tifffile.imread("./data/segmentations/HCT116-0Gy-3j-2900X-0002_seg.tif")

mask = (labels > 0).astype(np.uint8) * 255

tifffile.imwrite("./data/segmentations/HCT116-0Gy-3j-2900X-0002_mask.tif", mask)
