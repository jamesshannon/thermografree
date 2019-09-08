from thermografree.htpa import *

with HTPA(0x1A, pull_ups=False) as dev:
  im = dev.measure_temperatures()

  print(to_celsius(im))
