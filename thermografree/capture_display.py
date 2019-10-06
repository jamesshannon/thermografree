import time
import png
import os

import numpy as np

from thermografree.htpa import *

np.set_printoptions(threshold=1500, linewidth=150, precision=2)

def _imageify_array(values, img_range=None):
  """Produce a grayscale image (array) from an array with arbitrary values.

  Pixel value will be 0 at the minimum range value and 255 at the max range
  value.

  Args:
      values (ndarray): Array of arbitrary values (e.g., celsius, cloud height).
      img_range (list(int, int), optional): Set with min/max range values. If
        None then the image range will include the entire range in the array.

  Returns:
      ndarray: Greyscale array where black is high side of range.
  """
  if img_range is None:
    img_range = (np.nanmin(values), np.nanmax(values))

  # subtract the lower bound from the array so that that value becomes 0
  #   then multiply everything by the multiplier to distribute the values
  #   inside of the range
  scale = 255 / (img_range[1] - img_range[0])

  # astype() causes nan's to become 0
  return np.clip((values - img_range[0]) * scale, 0, 255).astype('uint8')


def bigify(nparr):
  new_size = 512
  bigarr = np.zeros((new_size,new_size))
  mult = int(new_size/32)
  for x in range(nparr.shape[0]):
    for y in range(nparr.shape[1]):
      bigarr[x*mult:x*mult+mult, y*mult:y*mult+mult] = nparr[x, y]

  return bigarr.astype('uint8')

print('* Creating HTPA device')
cnt = 0
with HTPA(0x1A, pull_ups=False) as dev:
  while True:
    print(time.ctime())
    ambient = dev.measure_ambient_temperature()
    print('Ambient Temp: {:+.1f}c'.format(to_celsius(ambient)))

    im = dev.measure_temperatures(num_frames=15)

    #print(_imageify_array(im))
    #print(bigify(_imageify_array(im)))
    fname = 'out{}.png'.format(cnt)
    png.from_array(bigify(_imageify_array(im)), 'L').save(fname)

    os.system('imgcat {}'.format(fname))

    print('32x32 image (celsius)')
    print(im.astype('int32'))

    cnt += 1
    time.sleep(10)

    print('\n\n')
