""" Test the eeprom_config module """

from thermografree import eeprom_config as ec

def test_unpack_byte():
  assert ec._unpack(b'\xb4') == 180
  assert ec._unpack(b'\xb4', signed=True) == -76


def test_unpack_2bytes():
  assert ec._unpack(b'\xb4\xb4') == 46260
  assert ec._unpack(b'\xb4\xb4', signed=True) == -19276

def test_unpack_4bytes():
  assert ec._unpack(b'\xb4\xb4\xb4\xb4') == -3.3659091513982276e-07
  assert ec._unpack(b'\xb4\xb4\xb4\xb4', signed=True) == -3.3659091513982276e-07