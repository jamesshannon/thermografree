import numpy as np
import cv2

def _save_img(fname, celsius_arr, options=None):
  img_range = (np.nanmin(celsius_arr), np.nanmax(celsius_arr))

  # subtract the lower bound from the array so that that value becomes 0
  #   then multiply everything by the multiplier to distribute the values
  #   inside of the range
  scale = 255 / (img_range[1] - img_range[0])

  # astype() causes nan's to become 0
  img = np.clip((celsius_arr - img_range[0]) * scale, 0, 255).astype('uint8')

  img = cv2.applyColorMap(img, cv2.COLORMAP_SPRING)

  cv2.imwrite(fname, img, options)