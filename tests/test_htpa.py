import numpy as np
import pytest
from itertools import chain

from thermografree import htpa as htpa_module

mean_vdd = 35000
mean_ptat = 38152

class I2CMock(object):
  def __init__(self, dev):
    pass

@pytest.fixture
def htpa():
  htpa_module.I2C = I2CMock
  htpa = htpa_module.HTPA()

  # Set constants as defined in the datasheet example (10.6)
  htpa.ptat_grad = 0.0211
  htpa.ptat_offset = 2195

  htpa.grad_scale = 24
  htpa.th_grad = 11137
  htpa.th_offset = -30

  htpa.calib1_vdd = 33942
  htpa.calib2_vdd = 36942
  htpa.calib1_ptat = 30000
  htpa.calib2_ptat = 42000

  htpa.vdd_comp_grad = 10356
  htpa.vdd_comp_offset = -14146
  htpa.vdd_scaling_grad = 16
  htpa.vdd_scaling_offset = 23

  htpa.pix_c = 1.087 * 1e8

  # global_offset isn't defined in the example
  htpa.global_offset = 0

  # this affects the lookup table used
  htpa.device = 'lookup_table_example'

  return htpa

def test_ambient_temp(htpa):
  assert htpa.ambient_temperature(38152) == pytest.approx(3000, rel=1e-2)

def test_thermal_compensation(htpa):
  # function does not include V00 (as in the example)
  assert htpa.thermal_offset(38152) == pytest.approx(55, rel=0.5)

def test_electrical_compensation(htpa):
  img = np.full((32,32), 34435, dtype='float64')
  electric_offset = np.full((32,32), 34240, dtype='uint32')

  t = htpa.temperature_compensation(img, electric_offset, mean_ptat, mean_vdd)

  # This disagrees with the datasheet but agrees with the spreadsheet (which
  #   uses the same vdd inputs as the datasheet). See the two tests in
  #   test_lookup_tables
  assert t[0][0] == pytest.approx(4026, rel=0.5)

def test_voltage_offset(htpa):
  # function does not include Vij_Comp (as in the example)
  assert htpa.voltage_offset(mean_ptat, mean_vdd) == pytest.approx(1, rel=1e-1)

def test_sensitivity_compensation(htpa):
  assert htpa.sensitivity_compensation(198) == pytest.approx(182, rel=0.25)


def test_flip_bottom_part():
  expected = np.array(range(256)).reshape((2, 128))
  input = np.array(list(chain(range(128),
                              range(224, 256),
                              range(192, 224),
                              range(160, 192),
                              range(128, 160)))).reshape((2, 128))
  assert np.equal(expected[1], htpa_module.HTPA.flip_bottom_part(input[1])).all()




