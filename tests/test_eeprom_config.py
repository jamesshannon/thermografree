""" Test the eeprom_config module """
import numpy as np

import pytest

from thermografree.eeprom_config import EEPROMConfiguration
from thermografree import eeprom_config as ec

from tests.eeprom_config_sample_values import SAMPLE_EEPROM_ARRAY
from tests.eeprom_config_sample_values import SAMPLE_TH_GRAD_ARRAY
from tests.eeprom_config_sample_values import SAMPLE_TH_OFFSET_ARRAY
from tests.eeprom_config_sample_values import SAMPLE_P_ARRAY

BYTES_OBJ = bytes([180] * 10)

@pytest.fixture
def config_obj():
  return EEPROMConfiguration(SAMPLE_EEPROM_ARRAY)

def test_unpack_byte():
  assert ec._unpack(b'\xb4') == 180
  assert ec._unpack(b'\xb4', signed=True) == -76

  # For a bytes object b, b[0] will be an integer, while b[0:1] will be a bytes
  assert ec._unpack(BYTES_OBJ[1]) == 180
  assert ec._unpack(BYTES_OBJ[1], signed=True) == -76

def test_unpack_2bytes():
  assert ec._unpack(BYTES_OBJ[:2]) == 46260
  assert ec._unpack(BYTES_OBJ[:2], signed=True) == -19276

def test_unpack_float():
  assert ec._unpack(BYTES_OBJ[:4]) == -3.3659091513982276e-07
  assert ec._unpack(BYTES_OBJ[:4], signed=True) == -3.3659091513982276e-07

def test_unpack_long():
  assert ec._unpack(BYTES_OBJ[:4], lng=True) == 3031741620

def test_eeprom_config(config_obj):
  assert config_obj.p_min == 42149544
  assert config_obj.p_max == 67916384
  assert config_obj.grad_scale == 23

  assert config_obj.table_num == 113
  assert config_obj.epsilon == 100

  assert config_obj.ptat_grad == pytest.approx(0.020186334848)
  assert config_obj.ptat_offset == pytest.approx(2251.07763671875)

  assert config_obj.device_id == 48344

def test_eeprom_config_thgrad(config_obj):
  np.testing.assert_array_equal(config_obj._th_grad, SAMPLE_TH_GRAD_ARRAY)

def test_eeprom_config_thoffset(config_obj):
  np.testing.assert_array_equal(config_obj._th_offset, SAMPLE_TH_OFFSET_ARRAY)

def test_eeprom_config_pix_c(config_obj):
  np.testing.assert_allclose(config_obj._pix_c, SAMPLE_P_ARRAY)
