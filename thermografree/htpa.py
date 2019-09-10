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

import copy
import logging
import time

import numpy as np
from periphery import I2C

from .eeprom_config import EEPROMConfiguration
from .lookup_tables import interpolate_tables
from .utils import broadcast_offset_param
from .utils import flip_bottom_part

logger = logging.getLogger(__name__)

revisions = {
  0: {'blockshift': 2},
  1: {'blockshift': 4},
}


class HTPA:
  def __init__(self, address=0x1a, revision=1, pull_ups=True):
    """Create HTPA object.

    Note that this doesn't do any i2c communication. init_device() wakes up
    the HTPA and loads the eeprom config data; this needs to be called but
    is done so automatically with a with() statement.

    Keyword Arguments:
        address {hexadecimal} -- i2c address of HTPA deviced (default: {0x1a})
        revision {int} -- HTPA device revision (default: {1})
        pull_ups {bool} -- Enable pull ups on HTPA device (default: {True})
    """
    assert revision in revisions.keys()
    assert pull_ups in (True, False)

    self.device = 'HTPA32x32dR1L2_1HiSiF5_0_Gain3k3_Extended'

    self.address = address
    self.i2c = I2C("/dev/i2c-1")
    self.use_pull_ups = pull_ups
    self.awoken = False
    self.config = None
    self.offset = None

  def init_device(self):
    """Wake up and initialize the HTPA device using configuration set during
    initialization plus eeprom settings which are pulled from the device.
    """
    # wake up the device
    wakeup_and_blind = self.generate_command(0x01, 0x01)
    # set ADC resolution to 16 bits
    adc_res = self.generate_command(0x03, 0x2C)

    logger.debug('Initializing capture settings')

    self.send_command(wakeup_and_blind)
    self.awoken = True

    self.send_command(adc_res)
    if self.use_pull_ups:
      self.set_pull_up()
    self.set_bias_current(0x0C)
    self.set_clock_speed(0x14)
    self.set_bpa_current(0x0C)

    logger.info("Grabbing EEPROM data")

    self.config = EEPROMConfiguration(self.get_eeprom())

    logger.info("Setting parameters from EEPROM data")
    self.send_command(wakeup_and_blind)
    self.send_command(adc_res)
    self.set_bias_current(self.config.calib_bias)
    self.set_bpa_current(self.config.calib_bpa)
    self.set_clock_speed(self.config.calib_clk)
    if self.use_pull_ups:
      self.set_pull_up(self.config.calib_pu)

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

    addresses = (addresses if isinstance(addresses, (list, tuple))
                     else (addresses, ))

    for address in addresses:
      self.send_command(self.generate_command(address, val))

  def get_eeprom(self, eeprom_address=0x50):
    """Get dump of EEPROM configuration data from the HTPA device.

    Keyword Arguments:
        eeprom_address {hexadecimal} -- Communication address (default: {0x50})

    Returns:
        np.array(bytes) -- EEPROM bytes
    """
    # Two separate I2C transfers in case the buffer size is small
    q1 = [I2C.Message([0x00, 0x00]), I2C.Message([0x00]*4000, read=True)]
    q2 = [I2C.Message([0x0f, 0xa0]), I2C.Message([0x00]*4000, read=True)]
    self.i2c.transfer(eeprom_address, q1)
    self.i2c.transfer(eeprom_address, q2)
    return np.array(q1[1].data + q2[1].data)

  def temperature_compensation(self, im, electric_offset, ptat, vdd):
    """
    Calculate pixel temperature using calibration values stored in EEPROM as
    per Section 10

    Args:
      im (np.ndarray): pixel sensor data (32, 32)
      electric_offset (np.ndarray): per-pixel electrical offset, (32, 32)
      ptat (np.ndarray): PTAT (proportional to absolute temperature)
        temperature sensor readings, (8)
      vdd (np.ndarray): VDD (power supply voltage) measurement (8)

    Returns:
    np.ndarray: pixel temperatures (32, 32)
    """
    mean_ptat = np.mean(ptat)
    mean_vdd = np.mean(vdd)

    # section 10.1
    t_ambient = self.ambient_temperature(mean_ptat)

    v = np.reshape(im, (32, 32))
    # section 10.2
    v -= self.thermal_offset(t_ambient)
    # section 10.3
    v -= np.reshape(electric_offset, (32, 32))
    # section 10.4 - Vdd compensation
    v -= self.voltage_offset(mean_ptat, mean_vdd)
    # section 10.5 - step 1
    v = self.sensitivity_compensation(v)
    # section 10.5 - step 2
    t = self.lookup_and_interpolate(v, t_ambient)

    # adding calibration temperature offset
    return t + self.config.global_offset

  @staticmethod
  def get_pix_idx(pix):
    """Get the 2d array index for a given pixel number.

    NB: Returns (y, x) because numpy array is row-based.

    Args:
        pix (int): Pixel number. HTPA is 0 based. E.g., [0-1024)

    Returns:
        (int,int): (0-based) index of  pixel within a 32x32 array.
    """
    return (pix // 32, pix % 32)

  @staticmethod
  def get_actual_pixels(pixels):
    """Convert readout pixels to actual pixels.

    EEPROM data is stored as readout pixels and pixel numbers need to be
    converted to match the read order of the pixels.

    Readout pixels are the same as actual pixels when value <= 512 (0x200)
    but above that they're flipped around a bit.

    Described in section 10.7.

    Args:
        pixels (np.ndarray): Numpy array of readout pixel numbers

    Returns:
        np.ndarray: Numpy array of actual pixel numbers
    """
    bot_mask = pixels > 0x200
    bot_vals = pixels[bot_mask]
    pixels[bot_mask] = 1024 + 512 - bot_vals + (bot_vals % 32 * 2) - 32

    return pixels

  @staticmethod
  def update_img_with_masked_avg(img, num_dead, pixels, masks):
    """Update dead pixels with average of surrounding pixels (based on mask).

    Described in Section 10.7.

    Args:
        img (np.ndarray): 2d numpy array
        num_dead (int): Number of dead pixels
        pixels (list(int)): List of pixel numbers, adapted for readout-order
        masks (list(int)): List of masks

    Returns:
        np.ndarray: Original array with updated pixels
    """
    for i in range(num_dead):
      mask = masks[i]
      pix = pixels[i]

      # The eeprom will return more pixels than are actually dead -- but any
      #   above the num_dead count will be 0.
      assert pix != 0

      pix_y, pix_x = HTPA.get_pix_idx(pix)

      img[pix_y][pix_x] = HTPA.get_masked_avg(img, pix, mask)

    return img

  @staticmethod
  def get_masked_avg(img, pix, mask):
    """Get the average value of surrounding pixels based on a HTPA mask.

    The average value of up to 8 pixels surrounding a given pixel. The subset
    of the 8 which are included is based positions in the bitmask.

    Described in Section 10.7.

    Args:
        img (np.ndarray): 2d numpy array
        pix (int): Pixel number, adapted for readout-order
        mask (int): 8-bit mask.
          We assume that the device will not ask us to read pixels which
          don't exist (are out of bounds).

    Returns:
        float: Average value of masked surrounding pixels.
    """
    pix_y, pix_x = HTPA.get_pix_idx(pix)

    if pix > 0x200:
      # bottom half mask ordering by LSB
      mask_trans = ((1,0),(1,1),(0,1),(-1,1),(-1,0),(-1,-1),(0,-1),(1,-1))
    else:
      # top half mask ordering by LSB
      mask_trans = ((-1,0),(-1,1),(0,1),(1,1),(1,0),(1,-1),(0,-1),(-1,-1))

    vals = 0

    for i in range(8):
      if mask >> i & 0x1:
        off_y, off_x = mask_trans[i]
        vals += img[pix_y + off_y][pix_x + off_x]

    return vals / bin(mask).count('1')

  def voltage_offset(self, mean_ptat, mean_vdd):
    """
    Calculates thermal offset that compensates differences between current
    supply voltage and calibration values as per Section 10.4.

    Args:
      mean_ptat (float): mean temperature reading
      mean_vdd (float): mean supply voltage reading

    Returns:
      np.ndarray: supply voltage offset compensation
    """
    calibration_param = ((self.config.calib2_vdd - self.config.calib1_vdd) /
                         (self.config.calib2_ptat - self.config.calib1_ptat))
    offset = (self.config.vdd_comp_grad * mean_ptat /
              pow(2, self.config.vdd_scaling_grad) +
              self.config.vdd_comp_offset)
    offset *= (mean_vdd - self.config.calib1_vdd - calibration_param *
               (mean_ptat - self.config.calib1_ptat))
    offset /= pow(2, self.config.vdd_scaling_offset)
    return offset

  def thermal_offset(self, mean_ptat):
    """
    Calculates per-pixel thermal drift offset, see Section 10.2
    Args:
      mean_ptat (float): mean PTAT (temperature) sensor reading

    Returns:
      (32,32) array of offsets, to be subtracted from Vij (raw pixel data)
    """
    # This is much more complex than the datasheet because we're returning an
    #   offset rather than doing the math. The return value should be
    #   subtractable from the pixel data (to keep some consistency with the
    #   datasheet).
    # Formula in datasheet is Vij - [gradient] - offset
    # If the values are 100, 50, 20 then result should be 30 (100 - 50 - 20)
    # If we return 50 - 20 (30) and that's later subtracted from 100 we get 70
    # Instead we do (50 * -1 - 20) * -1 = 70. 100 - 70 = 30
    return (self.config.th_grad * mean_ptat / pow(2, self.config.grad_scale)) + self.config.th_offset

  def ambient_temperature(self, mean_ptat):
    """
    Calculates ambient temperature as per 9.1

    Args:
      mean_ptat (float): mean PTAT (temperature) sensor reading

    Returns:
      float: ambient temperature
    """
    return mean_ptat * self.config.ptat_grad + self.config.ptat_offset

  def sensitivity_compensation(self, im):
    """
    Applies pixel sensitivity compensation as per first part of Section 10.6

    Args:
      im (np.ndarray): pixel data (32, 32)

    Returns:
      np.ndarray: scaled pixel data (32, 32)
    """
    return im * self.config.PCSCALEVAL / self.config.pix_c

  def offset_compensation(self, im):
    return im - self.offset

  def measure_ambient_temperature(self):
    """Measure the ambient temperature. Returns dK."""
    pixel_values, ptats = self.capture_image()
    return self.ambient_temperature(np.mean(ptats))

  def measure_temperatures(self):
    """Measure temperatures in dK"""
    electric_offset, vdd = self.capture_offsets()
    pixel_values, ptats = self.capture_image()
    thermal_image = self.temperature_compensation(pixel_values, electric_offset, ptats, vdd)
    thermal_image = self.offset_compensation(thermal_image)
    return thermal_image

  def measure_observed_offset(self):
    logger.info(('Measuring observed offsets. Camera should be against '
                 'uniform temperature surface.'))

    mean_offset = np.zeros((32, 32))
    electric_offset, vdd = self.capture_offsets()

    for i in range(10):
      logger.debug(' > frame %s', i)
      image, ptat = self.capture_image()
      im = self.temperature_compensation(image, electric_offset, ptat, vdd)
      mean_offset += im / 10.0

    self.offset = mean_offset

  def capture(self, blocks, **kwargs):
    """
    Capture sensor data as per Section 9.3 Sensor commands
      (tables 15, 16, 17 and 18)
    Data layout is described in Section 6.

    Args:
      blocks (int): number of blocks to read
      **kwargs: command flags, see `self.generate_expose_block_command`

    Returns:
      (int, np.ndarray, np.ndarray): block number, pixel data top part, pixel
        data bottom part
    """
    for block in range(blocks):
      msg_data = self.send_command(
          self.generate_expose_block_command(block, **kwargs), wait=False)
      query = [I2C.Message([0x02]), I2C.Message([0x00], read=True)]

      # start flag is not returned in the response message so we're not
      #   expecting it to be transmitted
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
    """
    Capture electrical offsets data. Issues read sensor commands as per
    Section 9.3 (tables 17 and 18)

    Returns:
      np.ndarray: pixels' electrical offsets (32, 32)
    """
    for block, top_data, bottom_data in self.capture(1, blind=True,
                                                     vdd_measure=True):
      vdds = np.zeros(2)
      vdds[block + 0] = top_data[0]
      vdds[block + 1] = bottom_data[0]
      return broadcast_offset_param(np.vstack([top_data[1:],
                                              flip_bottom_part(bottom_data[1:])
                                              ])), vdds

  def capture_image(self):
    """
    Capture thermal image. Issues read sensor commands as per
    Section 9.3 (tables 15 and 16)
    Returns:
      np.ndarray: sensor pixels' voltage readings (32,32)
    """
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
  def config_flags(flags=0, wakeup=False, blind=False, vdd_measure=False,
                   start=False, block=None):
    """
    Set sensor command flags according to Section 9.3.

    Args:
      flags (int): packed flag value
      wakeup: send wake-up signal to sensor
      blind: capture electrical offset instead of thermal readings
      vdd_measure: capture Vdd (supply voltage) instead of PTAT (temperature)
      start: start the transmission of data
      block: block number, (0-3) for thermal data, (0, 1) for electrical offsets

    Returns:
      int: packed flag value
    """
    if wakeup:
      flags |= 1
    if blind:
      flags |= 1 << 1
    if vdd_measure:
      flags |= 1 << 2
    if start:
      flags |= 1 << 3
    if block is not None:
      flags |= block << 4
    return flags

  def generate_expose_block_command(self, block, blind=False,
                                    vdd_measure=False):
    """
    Generates I2C command that reads sensor data block from device.
    See section 9.3.
    Args:
      block (int): block number
      blind (bool): read electrical offsets instead of thermal image
      vdd_measure (bool): read Vdd instead of PTAT

    Returns:

    """
    val = self.config_flags(wakeup=True, start=True, block=block, blind=blind,
                            vdd_measure=vdd_measure)
    return self.generate_command(0x01, val)

  def send_command(self, cmd, wait=True):
    self.i2c.transfer(self.address, cmd)
    if wait:
      time.sleep(0.005) # sleep for 5 ms
    return cmd[0].data

  def close(self):
    sleep = self.generate_command(0x01, 0x00)
    self.send_command(sleep)

  def lookup_and_interpolate(self, t, t_ambient):
    """
    Calculates 'true' pixels temperature by looking up cell containing
    (t, t_ambient) in temperature lookup table and interpolating value using
    cell corners' values.
    Args:
      t (np.ndarray): measured temperatures (32, 32)
      t_ambient (float): measured ambient temperature

    Returns:
      np.ndarray: 'true' thermal image
    """
    return interpolate_tables(t_ambient, t, self.device, self.config.table_num)

  def __enter__(self):
    self.init_device()
    return self

  def __exit__(self, exc_type, exc_value, exc_traceback):
    self.close()


def to_celsius(dK, rnd=2):
  """Convert degrees Kelvin to degrees Celsius.

  Args:
      vals (float,ndarray): dK (deci-Kelvin or K * 10) value
        (or array of values)
      rnd (int, optional): Decimal points to round to

  Returns:
      float,ndarray: Rounded value(s) in degrees Celsius
  """
  return np.around(dK / 10.0 - 273.15, rnd)


if __name__ == '__main__':
  # FYI: The RPi already has 1.8k pull up resistors, so additional pullups are
  #   unnecessary and is likely to be out of I2C spec and will increase
  #   current draw
  with HTPA(pull_ups=False) as htpa:
    htpa.measure_observed_offset()
