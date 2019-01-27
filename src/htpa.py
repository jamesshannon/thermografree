"""
Original htpa.py licensed under MIT License

Copyright (c) 2017 Logan Williams

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

from __future__ import division

from periphery import I2C
import time
import numpy as np
import copy
import struct

from lookup_tables import interpolate_tables

revisions = {
  0: {'blockshift': 2},
  1: {'blockshift': 4},
}

unpack_formats = {
    True: {1: 'b', 2: 'h', 4: 'f',},
    False: {1: 'B', 2: 'H', 4: 'f',},
}

class HTPA:
  PCSCALEVAL = 1e8

  def __init__(self, address=0x1a, revision=1, pull_ups=True):
    assert revision in revisions.keys()
    assert pull_ups in (True, False)

    self.device = 'HTPA32x32dR1L2_1HiSiF5_0_Gain3k3_Extended'

    self.address = address
    self.i2c = I2C("/dev/i2c-1")
    self.use_pullups = pull_ups

    self.awoken = False

  def init_device(self):
    wakeup_and_blind = self.generate_command(0x01, 0x01) # wake up the device
    adc_res = self.generate_command(0x03, 0x0C) # set ADC resolution to 16 bits

    print("Initializing capture settings")

    self.send_command(wakeup_and_blind)
    self.awoken = True

    self.send_command(adc_res)
    if self.use_pull_ups:
      self.set_pull_up()

    self.set_bias_current(0x05)
    self.set_clock_speed(0x15)
    self.set_bpa_current(0x0C)

    print ("Grabbing EEPROM data")

    eeprom = self.get_eeprom()
    self.extract_eeprom_parameters(eeprom)
    self.eeprom = eeprom

    print ("Setting parameters from EEPROM data")
    self.send_command(wakeup_and_blind)
    self.send_command(adc_res)
    self.set_bias_current(self.calib_bias)
    self.set_bpa_current(self.calib_bpa)
    self.set_clock_speed(self.calib_clk)
    if pull_ups:
      self.set_pull_up(self.calib_pu)

    # initialize offset to zero
    self.offset = np.zeros((32, 32))

  def set_bias_current(self, bias):
    self._send_clamped_int((0x04, 0x05), bias, 0, 31)

  def set_clock_speed(self, clk):
    self._send_clamped_int(0x06, clk, 0, 63)

  def set_pull_up(self, pu=0x88):
    self.generate_command(0x09, pu)

  def set_bpa_current(self, cm):
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

  def broadcast_offset_param(self, data):
    return np.array([[data[int(i / 16)][(j + i * 32) % 128] for j in range(32)] for i in range(32)])

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
    self.vdd_comp_grad = self.broadcast_offset_param(np.frombuffer(ebytes[0x0340:0x0540], dtype='<i2')\
      .copy().reshape((2, -1)))

    # VddCompOffij stored as 16 bit signed values
    # JS: Though the original thermografee code read as unsigned
    # JS: Looks to be an (8, 32) shape
    self.vdd_comp_offset = self.broadcast_offset_param(np.frombuffer(ebytes[0x0540:0x0740], dtype='<i2')\
      .copy().reshape((2, -1)))

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

  def unpack(self, byts, signed=False):
    fmt = '<' + unpack_formats[signed][len(byts)]

    return struct.unpack(fmt, byts)[0]

  def temperature_compensation(self, im, electric_offset, ptat, vdd):
    mean_ptat = np.mean(ptat)
    mean_vdd = np.mean(vdd)

    t_ambient = self.ambient_temperature(mean_ptat)

    v = np.reshape(im, (32, 32))
    v -= self.thermal_offset(t_ambient)
    v -= np.reshape(electric_offset, (32, 32))
    v -= self.voltage_offset(mean_ptat, mean_vdd)
    v = self.sensitivity_compensation(v)

    t = self.lookup_and_interpolate(v, t_ambient)

    return t + self.global_offset

  @staticmethod
  def scaled_mean_ptat(mean_ptat, coef, offset, scale):
    return coef * mean_ptat / pow(2, scale) + offset

  def voltage_offset(self, mean_ptat, mean_vdd):
    calibration_param = (self.calib2_vdd - self.calib1_vdd) / (self.calib2_ptat - self.calib1_ptat)
    offset = self.scaled_mean_ptat(mean_ptat, self.vdd_comp_grad, self.vdd_comp_offset, self.vdd_scaling_grad)
    offset *= mean_vdd - self.calib1_vdd - calibration_param * (mean_ptat - self.calib1_ptat)
    offset /= pow(2, self.vdd_scaling_offset)
    return offset

  def thermal_offset(self, mean_ptat):
    return ((self.th_grad * mean_ptat) / pow(2, self.grad_scale)) - self.th_offset

  def ambient_temperature(self, mean_ptat):
    return mean_ptat * self.ptat_grad + self.ptat_offset

  def offset_compensation(self, im):
    return im - self.offset

  def sensitivity_compensation(self, im):
    return im * self.PCSCALEVAL / self.pix_c

  def measure_observed_offset(self):
    print("Measuring observed offsets")
    print("    Camera should be against uniform temperature surface")

    mean_offset = np.zeros((32, 32))
    electric_offset, vdd = self.capture_offsets()

    for i in range(10):
      print("    frame " + str(i))
      image, ptat = self.capture_image()
      im = self.temperature_compensation(image, electric_offset, ptat, vdd)
      mean_offset += im / 10.0

    self.offset = mean_offset

  def capture(self, blocks, **kwargs):
    for block in range(blocks):
      msg_data = self.send_command(self.generate_expose_block_command(block, **kwargs), wait=False)
      query = [I2C.Message([0x02]), I2C.Message([0x00], read=True)]
      expected = msg_data[1] ^ self.config_flags(start=True)

      done = False

      while not done:
        self.i2c.transfer(self.address, query)

        if not (query[1].data[0] == expected):
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
      yield block, top_data, bottom_data

  def capture_offsets(self):
    for block, top_data, bottom_data in self.capture(1, blind=True, vdd_measure=True):
      vdds = np.zeros(2)
      vdds[block + 0] = top_data[0]
      vdds[block + 1] = bottom_data[0]
      return self.broadcast_offset_param(np.vstack([top_data, bottom_data])), vdds

  def capture_image(self):
    pixel_values = np.zeros(1024)
    ptats = np.zeros(8)

    for block, top_data, bottom_data in self.capture(4, blind=False):
      pixel_values[(0+block*128):(128+block*128)] = top_data[1:]
      # bottom data is in a weird shape
      pixel_values[(992-block*128):(1024-block*128)] = bottom_data[1:33]
      pixel_values[(960-block*128):(992-block*128)] = bottom_data[33:65]
      pixel_values[(928-block*128):(960-block*128)] = bottom_data[65:97]
      pixel_values[(896-block*128):(928-block*128)] = bottom_data[97:]

      ptats[block] = top_data[0]
      ptats[7-block] = bottom_data[0]

    pixel_values = np.reshape(pixel_values, (32, 32))
    return pixel_values, ptats

  @staticmethod
  def generate_command(register, value):
    return [I2C.Message([register, value])]

  @staticmethod
  def config_flags(flags=0, wakeup=False, blind=False, vdd_measure=False, start=False, block=None):
    if wakeup:
      flags |= 1
    if blind:
      flags |= 1 << 1
    if vdd_measure:
      flags |= 1 << 2
    if start:
      flags |= 1 << 3
    if block is not None:
      flags |= 1 << 4
    return flags

  def generate_expose_block_command(self, block, blind=False, vdd_measure=False):
    return self.generate_command(0x01, self.config_flags(wakeup=True, start=True, block=block, blind=blind, vdd_measure=vdd_measure))

  def send_command(self, cmd, wait=True):
    self.i2c.transfer(self.address, cmd)
    if wait:
      time.sleep(0.005) # sleep for 5 ms
    return cmd[0].data

  def close(self):
    sleep = self.generate_command(0x01, 0x00)
    self.send_command(sleep)

  def lookup_and_interpolate(self, t, t_ambient):
    return interpolate_tables(t_ambient, t, device=self.device)

  def __enter__(self):
    self.init_device()
    return self

  def __exit__(self, exc_type, exc_value, exc_traceback):
    self.close()

def _to_celsius(dK, rnd=2):
  """Convert degrees Kelvin to degrees Celsius.

  Args:
      vals (float,ndarray): K value (or array of values)
      rnd (int, optional): Decimal points to round to

  Returns:
      float,ndarray: Rounded value(s) in degrees Celsius
  """
  return np.around(dK - 273.15, rnd)

def _save_img(fname, celsius_arr):
  img_range = (np.nanmin(celsius_arr), np.nanmax(celsius_arr))

  # subtract the lower bound from the array so that that value becomes 0
  #   then multiply everything by the multiplier to distribute the values
  #   inside of the range
  scale = 255 / (img_range[1] - img_range[0])

  # astype() causes nan's to become 0
  img = np.clip((celsius_arr - img_range[0]) * scale, 0, 255).astype('uint8')

  img = cv2.applyColorMap(img, cv2.COLORMAP_SPRING)

  cv2.imwrite(fname, img, options)


if __name__ == '__main__':
  # FYI: The RPi already has 1.8k pull up resistors, so additional pullups are
  #   unnecessary and is likely to be out of I2C spec and will increase
  #   current draw
  with HTPA(pull_ups=False) as htpa:
    htpa.measure_observed_offset()
