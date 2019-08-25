""" Utils to move HTPA bytes around """

import numpy as np

def broadcast_offset_param(data):
  """
  Broadcasts 8x32 electrical offset data to corresponding 32x32 pixel array
  accordingto Section 6 and Section 10.3

  Args:
    data (np.ndarray): electrical offsets data (8, 32)

  Returns:
    np.ndarray: per-pixel electrical offset
  """
  return np.array([[data[int(i / 16)][(j + i * 32) % 128]
                    for j in range(32)] 
                   for i in range(32)])


def flip_bottom_part(data):
  """
  Flips bottom part of the `data` buffer according to
  Section 6 - "Serial Order of Frame"

  Args:
   data (np.ndarray): buffer data

  Returns:
    np.ndarray: continuously indexed data
  """
  shape = data.shape
  data = data.reshape((-1, 32))
  return np.flipud(data).reshape(shape)