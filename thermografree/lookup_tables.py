from __future__ import division

import functools
import os

import numpy as np

def interpolate_tables(t_ambient, image, device, table_num):
  # TODO: More unit tests for me, please
  ta_axes, dk_axes, table, offset = get_table_and_axes(device, table_num)

  ta_axes = np.array(ta_axes, dtype='float')
  dk_axes = np.array(dk_axes, dtype='float') - offset

  d_ta = ta_axes[1] - ta_axes[0]
  d_dk = dk_axes[1] - dk_axes[0]

  #TODO: Run through this code again
  data = image.ravel()
  ta_idx = np.searchsorted(ta_axes, t_ambient) - 1
  dk_idx = np.searchsorted(dk_axes, data) - 1

  # get the value of the row / column after which the value is found
  x1 = ta_axes[ta_idx]
  y1 = dk_axes[dk_idx]

  # top-left corner of the mini-table
  table_y_x = table[dk_idx].transpose()[ta_idx]
  # bottom-left corner of the mini-table
  table_y_x1 = table[dk_idx].transpose()[ta_idx + 1]
  # top-right corner of the mini-table
  table_y1_x = table[dk_idx + 1].transpose()[ta_idx]
  # bottom-right corner of the mini-table
  table_y1_x1 = table[dk_idx + 1].transpose()[ta_idx + 1]

  v_x = (table_y_x1 - table_y_x) * (t_ambient - x1) / d_ta + table_y_x
  v_y = (table_y1_x1 - table_y1_x) * (t_ambient - x1) / d_ta + table_y1_x

  result = (v_y - v_x) * (data - y1) / d_dk + v_x

  return result.reshape(image.shape)

@functools.lru_cache(maxsize=3)
def get_table_and_axes(device, table_num):
  fname = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       'data', '{}.npz'.format(device))
  with np.load(fname, allow_pickle=True) as data:
    table = data['table']
    ta_axes = data['ta_axes']
    dk_axes = data['dk_axes']

    metadata = data['metadata'].item()

    # Cross-check the data we've loaded.
    # TODO: Move some of this to the table .npz writing (though keep some here)
    assert metadata['_DEVICE_NAME'] == device
    assert not table_num or metadata['TABLENUMBER'] == table_num, \
        '{} has table number {} not {}'.format(fname, metadata['TABLENUMBER'],
                                               table_num)

    assert len(ta_axes.shape) == 1
    assert len(dk_axes.shape) == 1
    assert table.shape == (len(dk_axes), len(ta_axes))

    assert (ta_axes[1] - ta_axes[0]) == metadata['TAEQUIDISTANCE']
    assert (dk_axes[1] - dk_axes[0]) == metadata['ADEQUIDISTANCE']

    return ta_axes, dk_axes, table, metadata['TABLEOFFSET']
