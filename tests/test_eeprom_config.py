""" Test the eeprom_config module """

from thermografree import eeprom_config as ec

BYTES_OBJ = bytes([180] * 10)

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