import numpy as np
import pytest
from itertools import chain

from mock import Mock

from thermografree import htpa as htpa_module
from thermografree.utils import flip_bottom_part, broadcast_offset_param

mean_vdd = 35000
mean_ptat = 38152
v00 = 34435
v00_comp = 199

class I2CMock(object):
  def __init__(self, dev):
    pass

@pytest.fixture
def htpa():
  htpa_module.I2C = I2CMock
  htpa = htpa_module.HTPA()

  # Set constants as defined in the datasheet example (10.6)
  config = Mock()

  config.PCSCALEVAL = 1e8
  config.ptat_grad = 0.0211
  config.ptat_offset = 2195

  config.grad_scale = 24
  config.th_grad = 11137
  config.th_offset = -30

  config.calib1_vdd = 33942
  config.calib2_vdd = 36942
  config.calib1_ptat = 30000
  config.calib2_ptat = 42000

  config.vdd_comp_grad = 10356
  config.vdd_comp_offset = -14146
  config.vdd_scaling_grad = 16
  config.vdd_scaling_offset = 23

  config.pix_c = 1.087 * 1e8

  # global_offset isn't defined in the example
  config.global_offset = 0

  htpa.config = config

  # this affects the lookup table used
  htpa.device = 'lookup_table_example'
  htpa.config.table_num = None

  return htpa

def test_ambient_temp(htpa):
  assert htpa.ambient_temperature(mean_ptat) == pytest.approx(3000, rel=1e-3)

def test_thermal_compensation(htpa):
  assert v00 - htpa.thermal_offset(mean_ptat) == pytest.approx(34439, rel=1e-3)

def test_electrical_compensation(htpa):
  img = np.full((32,32), 34435, dtype='float64')
  electric_offset = np.full((32,32), 34240, dtype='uint32')

  t = htpa.temperature_compensation(img, electric_offset, mean_ptat, mean_vdd)

  # This disagrees with the datasheet but agrees with the spreadsheet (which
  #   uses the same vdd inputs as the datasheet). See the two tests in
  #   test_lookup_tables
  assert t[0][0] == pytest.approx(4026, rel=0.5)

def test_voltage_offset(htpa):
  v00_vddcomp = v00_comp - htpa.voltage_offset(mean_ptat, mean_vdd)
  assert v00_vddcomp == pytest.approx(198, rel=1e-3)

def test_sensitivity_compensation(htpa):
  assert htpa.sensitivity_compensation(198) == pytest.approx(182, rel=0.25)

def test_flip_bottom_part():
  expected = np.array(range(256)).reshape((2, 128))
  inp = np.array(list(chain(range(128),
                            range(224, 256),
                            range(192, 224),
                            range(160, 192),
                            range(128, 160)))).reshape((2, 128))
  assert np.equal(expected[1], flip_bottom_part(inp[1])).all()

def test_broadcast_offset_param(htpa):
  # create a list with 8 rows and 32 cols (with values as the rownumber)
  offsets = np.array([[i] * 32 for i in range(8)]).reshape((2, -1))
  # expected output is the rows repeated in groups of 4 for each half
  expected = ([[i] * 32 for i in range(4)] * 4 \
              + [[i] * 32 for i in range(4, 8)] * 4)

  offsets = broadcast_offset_param(offsets)

  assert offsets.shape == (32, 32)
  assert np.array_equal(offsets, expected)

def test_get_pix_idx():
  assert htpa_module.HTPA.get_pix_idx(15) == (0, 15)
  assert htpa_module.HTPA.get_pix_idx(512) == (16, 0)
  assert htpa_module.HTPA.get_pix_idx(703) == (21, 31)

def test_actual_pixels():
  inp = np.array((15, 300, 500, 997, 661))
  expect = (15, 300, 500, 517, 977)

  """
The datasheet says that 661 should be 997, but the correct answer APPEARS
  to be 885. TBD until I can get confirmation.
>>> 0x295
661
>>> 661 % 32
21
>>> 21 *2
42
>>> 1024 + 512 - 661 + 42 -32
885 <<< -- The datasheet says 661 should be 977
>>>
  """
  expect = [15, 300, 500, 517, 885]

  assert htpa_module.HTPA.get_actual_pixels(inp).tolist() == expect

def _get_masked_test_data():
  img_arr = np.full((32, 32), 100, dtype='float')
  surrounding_pix = ((3010, 3012, 3005), (3007, 20, 3008), (3008, 3011, 3009))
  dead_pixels = (15, 300, 977)
  masks = (0x7c, 0x8f, 0xFE)
  expected = (3008.6, 3008.8, 3008.428)

  # update subsets of the array with the surrounding pix blocks
  for pix, surrouding in zip(dead_pixels, surrounding_pix):
    pix_y, pix_x = htpa_module.HTPA.get_pix_idx(pix)

    if pix_y == 0:
      # This replaces a chunk that's at the top edge of the array by only
      #   replacing 2 rows (not 3). Other edges (left,right,bottom) would have
      #   to be handled separately
      img_arr[0:pix_y+2,pix_x-1:pix_x+2] = surrounding_pix[1:]
    else:
      img_arr[pix_y-1:pix_y+2,pix_x-1:pix_x+2] = surrounding_pix

  return img_arr, dead_pixels, masks, expected


def test_get_masked_values():
  img_arr, dead_pix, masks, expected = _get_masked_test_data()

  for pix, mask, exp in zip(dead_pix, masks, expected):
    assert htpa_module.HTPA.get_masked_avg(img_arr, pix, mask) \
        == pytest.approx(exp)


def test_update_img_with_masked_avg():
  img_arr, dead_pix, masks, expected = _get_masked_test_data()

  img_arr = htpa_module.HTPA.update_img_with_masked_avg(img_arr, len(dead_pix),
                                                        dead_pix, masks)
  for pix, exp in zip(dead_pix, expected):
    pix_y, pix_x = htpa_module.HTPA.get_pix_idx(pix)
    assert img_arr[pix_y][pix_x].tolist() == pytest.approx(exp)
