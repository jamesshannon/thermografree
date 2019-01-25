from __future__ import division

from periphery import I2C
import time
import numpy as np
import copy
import struct

import IPython
import pdb

revisions = {
  0: {'blockshift': 2},
  1: {'blockshift': 4},
}

unpack_formats = {
    True: {1: 'b', 2: 'h', 4: 'f',},
    False: {1: 'B', 2: 'H', 4: 'f',},
}

class HTPA:
  def __init__(self, address=0x1a, revision=1, pull_ups=True):
    assert revision in revisions.keys()
    assert pull_ups in (True, False)

    self.address = address
    self.i2c = I2C("/dev/i2c-1")

    self.blockshift = revisions[revision]['blockshift']

    wakeup_and_blind = self.generate_command(0x01, 0x01) # wake up the device
    adc_res = self.generate_command(0x03, 0x0C) # set ADC resolution to 16 bits

    print("Initializing capture settings")

    self.send_command(wakeup_and_blind)
    self.send_command(adc_res)
    if pull_ups:
      self.send_command(self.generate_command(0x09, 0x88))

    self.set_bias_current(0x05)
    self.set_clock_speed(0x15)
    self.set_cm_current(0x0C)

    print ("Grabbing EEPROM data")

    eeprom = self.get_eeprom()
    self.extract_eeprom_parameters(eeprom)
    self.eeprom = eeprom

    # initialize offset to zero
    self.offset = np.zeros((32, 32))

  def set_bias_current(self, bias):
    self._send_clamped_int((0x04, 0x05), bias, 0, 31)

  def set_clock_speed(self, clk):
    self._send_clamped_int(0x06, clk, 0, 63)

  def set_cm_current(self, cm):
    self._send_clamped_int((0x07, 0x08), cm, 0, 31)

  def _send_clamped_int(self, addresses, val, min_val, max_val):
    val = int(max(val, min(val, max_val)))

    addresses = addresses if isinstance(addresses, (list, tuple)) else (addresses, )

    for address in addresses:
      self.send_command(self.generate_command(address, val))

  def get_eeprom(self, eeprom_address=0x50):
    # Two separate I2C transfers in case the buffer size is small
    q1 = [I2C.Message([0x00, 0x00]), I2C.Message([0x00]*4000, read=True)]
    q2 = [I2C.Message([0x0f, 0xa0]), I2C.Message([0x00]*4000, read=True)]
    self.i2c.transfer(eeprom_address, q1)
    self.i2c.transfer(eeprom_address, q2)
    return np.array(q1[1].data + q2[1].data)

  def extract_eeprom_parameters(self, eeprom):
    ebytes = eeprom.astype('u1').tobytes()

    self.num_dead_pix = eeprom[0x007F]
    # DeadPixAdr as 16 bit unsigned values
    self.dead_pix_addr = np.frombuffer(ebytes[0x0080:0x00B0], dtype='<u2')\
        .copy()
    self.dead_pix_mask = np.frombuffer(ebytes[0x00B0:0x00C8], dtype='<u2')\
        .copy()


    # VddCompGradij stored as 16 bit signed values
    # JS: Looks to be an (8, 32) shape
    self.vdd_comp_grad = np.frombuffer(ebytes[0x0340:0x0540], dtype='<i2')\
        .copy()

    # VddCompOffij stored as 16 bit signed values
    # JS: Though the original thermografee code read as unsigned
    # JS: Looks to be an (8, 32) shape
    self.vdd_comp_offset = np.frombuffer(ebytes[0x0540:0x0740], dtype='<i2')

    # ThGradij stored as 16 bit signed values
    self.th_grad = np.frombuffer(ebytes[0x0740:0x0F40], dtype='<i2').copy()\
        .reshape((32, 32))

    # ThOffsetij stored as 16 bit signed values
    # JS: Though the original thermografee code read as unsigned
    self.th_offset = np.frombuffer(ebytes[0x0F40:0x1740], dtype='<i2').copy()\
        .reshape((32, 32))

    # Pij stored as 16 bit unsigned values
    self.P = np.frombuffer(ebytes[0x1740:], dtype='<u2').copy()\
      .reshape((32, 32))

    # The corresponding order of ThGradij, ThOffsetij, and Pij to the
    #   Pixelnumber is given by the following overview:...
    self.th_grad[16:,:] = np.flipud(self.th_grad[16:,:])
    self.th_offset[16:,:] = np.flipud(self.th_offset[16:,:])
    self.P[16:, :] = np.flipud(self.P[16:,:])

    epsilon = eeprom[0x000D]
    global_offset = self.unpack(ebytes[0x0054], signed=True)
    #  GlobalGain and VddCalib are both stored as 16 bit unsigned
    global_gain = self.unpack(ebytes[0x0055:0x0057])

    p_min = self.unpack(ebytes[0x0000:0x0004])
    p_max = self.unpack(ebytes[0x0004:0x0008])
    self.pix_c = ((self.P * (p_max - p_min) / 65535. + p_min)
                  * (epsilon / 100) * (GlobalGain / 100))

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

  def unpack(self, byts, signed=False):
    fmt = '<' + unpack_formats[signed][len(byts)]

    return struct.unpack(fmt, byts)[0]

  def temperature_compensation(self, im, ptat):
      comp = np.zeros((32,32))

      Ta = np.mean(ptat) * self.PTATgradient + self.PTAToffset
    #     temperature compensated voltage
      comp = ((self.ThGrad * Ta) / pow(2, self.gradScale)) + self.ThOffset

      Vcomp = np.reshape(im,(32, 32)) - comp
      return Vcomp

  def offset_compensation(self, im):
    return im - self.offset

  def sensitivity_compensation(self, im):
    return im/self.PixC

  def measure_observed_offset(self):
    print("Measuring observed offsets")
    print("    Camera should be against uniform temperature surface")
    mean_offset = np.zeros((32, 32))

    for i in range(10):
      print("    frame " + str(i))
      (p, pt) = self.capture_image()
      im = self.temperature_compensation(p, pt)
      mean_offset += im/10.0

    self.offset = mean_offset

  def measure_electrical_offset(self):
    (offsets, ptats) = self.capture_image(blind=True)
    self.offset = self.temperature_compensation(offsets, ptats)

  def capture_image(self, blind=False):
    pixel_values = np.zeros(1024)
    ptats = np.zeros(8)

    for block in range(4):
      # print("Exposing block " + str(block))
      self.send_command(self.generate_expose_block_command(block, blind=blind), wait=False)

      query = [I2C.Message([0x02]), I2C.Message([0x00], read=True)]
      expected = 1 + (block << 2)

      done = False

      while not done:
        self.i2c.transfer(self.address, query)

        if not (query[1].data[0] == expected):
          # print("Not ready, received " + str(query[1].data[0]) + ", expected " + str(expected))
          time.sleep(0.005)
        else:
          done = True

      read_block = [I2C.Message([0x0A]), I2C.Message([0x00]*258, read=True)]
      self.i2c.transfer(self.address, read_block)
      top_data = np.array(copy.copy(read_block[1].data))

      read_block = [I2C.Message([0x0B]), I2C.Message([0x00]*258, read=True)]
      self.i2c.transfer(self.address, read_block)
      bottom_data = np.array(copy.copy(read_block[1].data))

      top_data = top_data[1::2] + (top_data[0::2] << 8)
      bottom_data = bottom_data[1::2] + (bottom_data[0::2] << 8)

      pixel_values[(0+block*128):(128+block*128)] = top_data[1:]
      # bottom data is in a weird shape
      pixel_values[(992-block*128):(1024-block*128)] = bottom_data[1:33]
      pixel_values[(960-block*128):(992-block*128)] = bottom_data[33:65]
      pixel_values[(928-block*128):(960-block*128)] = bottom_data[65:97]
      pixel_values[(896-block*128):(928-block*128)] = bottom_data[97:]

      ptats[block] = top_data[0]
      ptats[7-block] = bottom_data[0]

    pixel_values = np.reshape(pixel_values, (32, 32))

    return (pixel_values, ptats)

  def generate_command(self, register, value):
    return [I2C.Message([register, value])]

  def generate_expose_block_command(self, block, blind=False):
    if blind:
      return self.generate_command(0x01, 0x09 + (block << self.blockshift) + 0x02)
    else:
      return self.generate_command(0x01, 0x09 + (block << self.blockshift))

  def send_command(self, cmd, wait=True):
    self.i2c.transfer(self.address, cmd)
    if wait:
      time.sleep(0.005) # sleep for 5 ms

  def close(self):
    sleep = self.generate_command(0x01, 0x00)
    self.send_command(sleep)

htpa = HTPA(pull_ups=False)
htpa.close()
