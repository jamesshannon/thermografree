import time

import numpy as np

from thermografree.htpa import *

np.set_printoptions(threshold=1500, linewidth=150, precision=2)

print('* Creating HTPA device')
with HTPA(0x1A, pull_ups=False) as dev:
  while True:
    print(time.ctime())
    ambient = dev.measure_ambient_temperature()
    print('Ambient Temp: {:+.1f}c'.format(to_celsius(ambient)))

    im = dev.measure_temperatures()
    print('32x32 image (celsius)')
    print(to_celsius(im).astype('int16'))

    time.sleep(10)

    print('\n\n')
