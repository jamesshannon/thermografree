"""HTPA EEPROM configuration parameters"""
import struct

import numpy as np

from .utils import broadcast_offset_param
from .utils import flip_bottom_part

UNPACK_FORMATS = {
    True: {1: 'b', 2: 'h', 4: 'f',},
    False: {1: 'B', 2: 'H', 4: 'f',},
}

def _unpack(byts, signed=False, lng=False):
  # Since bytes objects are sequences of integers (akin to a tuple), for a
  #   bytes object b, b[0] will be an integer, while b[0:1] will be a bytes
  #   object of length 1.
  if lng:
    assert len(byts) == 4
    fmt = '<L'
  else:
    if isinstance(byts, int):
      byts = bytes([byts])

    fmt = '<' + UNPACK_FORMATS[signed][len(byts)]

  return struct.unpack(fmt, byts)[0]

class EEPROMConfiguration:
  """
  EEPROM configuration parameters
  """
  PCSCALEVAL = 1e8

  def __init__(self, eeprom):
    self.raw = eeprom
    ebytes = eeprom.astype('u1').tobytes()

    self.num_dead_pix = eeprom[0x007F]
    # DeadPixAdr as 16 bit unsigned values
    self.dead_pix_addr = np.frombuffer(ebytes[0x0080:0x00B0], dtype='<u2') \
      .copy()
    self.dead_pix_mask = np.frombuffer(ebytes[0x00B0:0x00C8], dtype='<u1') \
      .copy()

    # VddCompGradij stored as 16 bit signed values
    # JS: Looks to be an (8, 32) shape
    self.vdd_comp_grad = np.frombuffer(ebytes[0x0340:0x0540], dtype='<i2') \
      .copy().reshape((2, -1))
    self.vdd_comp_grad[1] = flip_bottom_part(self.vdd_comp_grad[1])
    self.vdd_comp_grad = broadcast_offset_param(self.vdd_comp_grad)

    # VddCompOffij stored as 16 bit signed values
    # JS: Though the original thermografee code read as unsigned
    # JS: Looks to be an (8, 32) shape
    self.vdd_comp_offset = np.frombuffer(ebytes[0x0540:0x0740], dtype='<i2') \
      .copy().reshape((2, -1))
    self.vdd_comp_offset[1] = flip_bottom_part(self.vdd_comp_offset[1])
    self.vdd_comp_offset = broadcast_offset_param(self.vdd_comp_offset)

    # ThGradij stored as 16 bit signed values
    self._th_grad = np.frombuffer(ebytes[0x0740:0x0F40], dtype='<i2').copy() \
      .reshape((32, 32))
    self.th_grad = self._th_grad.copy()

    # ThOffsetij stored as 16 bit signed values
    # JS: Though the original thermografee code read as unsigned
    self._th_offset = np.frombuffer(ebytes[0x0F40:0x1740], dtype='<i2').copy() \
      .reshape((32, 32))
    self.th_offset = self._th_offset.copy()

    # Pij stored as 16 bit unsigned values
    # NB: Not yet flipped
    self._P = np.frombuffer(ebytes[0x1740:], dtype='<u2').copy() \
      .reshape((32, 32))

    # The corresponding order of ThGradij, ThOffsetij, and Pij to the
    #   Pixelnumber is given by the following overview:...
    self.th_grad[16:,:] = np.flipud(self.th_grad[16:,:])
    self.th_offset[16:,:] = np.flipud(self.th_offset[16:,:])

    self.epsilon = eeprom[0x000D]
    self.global_offset = _unpack(ebytes[0x0054], signed=True)
    #  GlobalGain and VddCalib are both stored as 16 bit unsigned
    self.global_gain = _unpack(ebytes[0x0055:0x0057])

    self.p_min = _unpack(ebytes[0x0000:0x0004])
    self.p_max = _unpack(ebytes[0x0004:0x0008])
    self._pix_c = ((self._P * (self.p_max - self.p_min) / 65535. + self.p_min)
                  * (self.epsilon / 100))

    # Add gain and flip the values. The example spreadsheet multiplies by
    #   epsilon but not globalgain, so we keep _pix_c without globalgain for
    #   unit testing purposes
    # Note, though, that the sample C code doesn't use global gain
    self.pix_c = self._pix_c.copy() * (self.global_gain / 10000)
    self.pix_c[16:, :] = np.flipud(self.pix_c[16:,:])

    self.grad_scale = eeprom[0x0008]
    #  GlobalGain and VddCalib are both stored as 16 bit unsigned
    self.vdd_calib = _unpack(ebytes[0x0046:0x0048])
    self.vdd = 3280.0
    self.vdd_scaling_grad = eeprom[0x004E]
    self.vdd_scaling_offset = eeprom[0x004F]

    self.ptat_grad = _unpack(ebytes[0x0034:0x0038])
    self.ptat_offset = _unpack(ebytes[0x0038:0x003c])

    self.table_num = _unpack(ebytes[0x000B:0x000D])

    self.calib_mbit = eeprom[0x001A]
    self.calib_bias = eeprom[0x001B]
    self.calib_clk = eeprom[0x001C]
    self.calib_bpa = eeprom[0x001D]
    self.calib_pu = eeprom[0x001E]

    self.calib1_vdd = _unpack(ebytes[0x0026:0x0028])
    self.calib2_vdd = _unpack(ebytes[0x0028:0x002A])
    self.calib1_ptat = _unpack(ebytes[0x003C:0x003E])
    self.calib2_ptat = _unpack(ebytes[0x003E:0x0040])

    self.device_id = struct.unpack('<L', ebytes[0x0074:0x0078])[0]
