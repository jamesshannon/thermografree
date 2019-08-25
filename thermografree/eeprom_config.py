"""HTPA EEPROM configuration parameters"""

import numpy as np

from .utils import broadcast_offset_param
from .utils import flip_bottom_part

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
    self.th_grad = np.frombuffer(ebytes[0x0740:0x0F40], dtype='<i2').copy() \
      .reshape((32, 32))

    # ThOffsetij stored as 16 bit signed values
    # JS: Though the original thermografee code read as unsigned
    self.th_offset = np.frombuffer(ebytes[0x0F40:0x1740], dtype='<i2').copy() \
      .reshape((32, 32))

    # Pij stored as 16 bit unsigned values
    self.P = np.frombuffer(ebytes[0x1740:], dtype='<u2').copy() \
      .reshape((32, 32))

    # The corresponding order of ThGradij, ThOffsetij, and Pij to the
    #   Pixelnumber is given by the following overview:...
    self.th_grad[16:,:] = np.flipud(self.th_grad[16:,:])
    self.th_offset[16:,:] = np.flipud(self.th_offset[16:,:])
    self.P[16:, :] = np.flipud(self.P[16:,:])

    epsilon = eeprom[0x000D]
    self.global_offset = self.unpack(ebytes[0x0054], signed=True)
    #  GlobalGain and VddCalib are both stored as 16 bit unsigned
    global_gain = self.unpack(ebytes[0x0055:0x0057])

    p_min = self.unpack(ebytes[0x0000:0x0004])
    p_max = self.unpack(ebytes[0x0004:0x0008])
    self.pix_c = ((self.P * (p_max - p_min) / 65535. + p_min)
                  * (epsilon / 100) * (global_gain / 100))

    self.grad_scale = eeprom[0x0008]
    #  GlobalGain and VddCalib are both stored as 16 bit unsigned
    self.vdd_calib = self.unpack(ebytes[0x0046:0x0048])
    self.vdd = 3280.0
    self.vdd_scaling_grad = eeprom[0x004E]
    self.vdd_scaling_offset = eeprom[0x004F]

    self.ptat_grad = self.unpack(ebytes[0x0034:0x0038])
    self.ptat_offset = self.unpack(ebytes[0x0038:0x003c])

    self.table_num = self.unpack(ebytes[0x000B:0x000D])

    self.calib_mbit = eeprom[0x001A]
    self.calib_bias = eeprom[0x001B]
    self.calib_clk = eeprom[0x001C]
    self.calib_bpa = eeprom[0x001D]
    self.calib_pu = eeprom[0x001E]

    self.calib1_vdd = self.unpack(ebytes[0x0026:0x0028])
    self.calib2_vdd = self.unpack(ebytes[0x0028:0x002A])
    self.calib1_ptat = self.unpack(ebytes[0x003C:0x003E])
    self.calib2_ptat = self.unpack(ebytes[0x003E:0x0040])

    self.device_id = struct.unpack('<L', ebytes[0x0074:0x0078])[0]

  @staticmethod
  def unpack(byts, signed=False):
    fmt = '<' + unpack_formats[signed][len(byts)]
    return struct.unpack(fmt, byts)[0]