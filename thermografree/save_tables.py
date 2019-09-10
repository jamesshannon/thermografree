""" Parse Heimann defs and table and write to a format we can use. """

import argparse
import ast
import itertools
import os
from pprint import pprint
import re

import numpy as np

# REGEXP for #define statements in defs.h
# some amount of whitespace then #define then whitespace
# identifier starts with uppercase then can be any word character
# then whitespace
# then (optionally) a token string which is -- at least so far -- numeric,
#   including hex
# #define statements often end in // comments -- make sure to skip those.
RE_DEF = re.compile(r'\s*#define\s+(?P<ident>[A-Z]\w*)\s+(?P<token>[0-9A-Fx]*)')
# REGEXP for the IFDEF in both defs.h and table.c
RE_IFDEF = re.compile(r'\s*#ifdef\s+(?P<ident>[A-Z]\w*)')
# REGEXP for the ENDIF in both defs.h and table.c
RE_ENDIF = re.compile(r'\s*#endif')
# REGEXP for the CONST arrays in Table.c
RE_CONST = re.compile(r'\s*const\s+unsigned\s+int\s+([A-Z]\w*)')

DATA_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data')

PARSER = argparse.ArgumentParser(description=('Create data files for the '
                                              'sensor constants and tables'))
PARSER.add_argument('SampleCode_dir',
                    help='Heimann SampleCode directory for the sensor type')
PARSER.add_argument('device_name', nargs='?',
                    help='Heimann device name string from defs.h')
ARGS = PARSER.parse_args()


def parse_defs(defsh):
  """ Parse the defs.h file and create a dict of device_name's and their
  constants. Also includes the _ALL "device" key for global constants. """
  all_defines = {'_ALL': {}}
  current_device = None

  with open(defsh) as fh:
    for line in fh:
      # Try to match this line against a #define
      match = RE_DEF.match(line)
      if match:
        ident, token = match.groups()
        if current_device is not None:
          current_device[ident] = _parse_int_bool(token)
        else:
          all_defines['_ALL'][ident] = _parse_int_bool(token)

        continue

      # Wasn't a #define; try to match against an #ifdef
      # If so, then "open" the device in our little state machine
      match = RE_IFDEF.match(line)
      if match:
        ident = match.groups()[0]
        if ident.startswith('HTPA'):
          current_device = {'_DEVICE_NAME': ident}
          all_defines[ident] = current_device

  return all_defines

def parse_tables(tablec):
  # set up a dict for tables.

  with open(tablec) as fh:
    tables = {}
    current_device = None

    for line in fh:
      # Match for #ifdef and in order to look for the device name
      match = RE_IFDEF.match(line)
      if match:
        ident = match.groups()[0]
        if ident.startswith('HTPA'):
          current_device = {}
          tables[ident] = current_device

        continue

      # Not #ifdef. Look for a constant definition
      match = RE_CONST.match(line)
      if match:
        varname = match.groups()[0]
        assert current_device is not None, "const line without a current device"

        # Get everything after the = sign. For some definitions this is just
        #   the opening bracket; for some it's the entire array
        tablelinestr = line.split('=')[1]
        tablestrs = []

        while True:
          # append the latest bit of the contstant to our queue
          tablestrs.append(tablelinestr)
          # look for the end of the definition and break out of this loop
          if '};' in tablelinestr:
            break

          tablelinestr = next(fh)

        # Create a python-ized version of the C table we've collected:
        #   Replace the {'s with ['s and get rid of the ;
        tablestr = ' '.join(tablestrs).replace('{', '[').replace('}', ']')\
            .replace(';', '')
        # Fix tabs
        #   Some arrays are separated by tables and not commas. Replace any
        #   tab which is surrounded by digits
        tablestr = re.sub(r'(\d)\t(\d)', r'\1,\2', tablestr)

        # literal_eval converts our table to a list (1d or 2d), including hex
        #   values
        pylist = ast.literal_eval(tablestr)
        # we're assuming that the values in tables.c are all uint32s
        flattened = pylist if isinstance(pylist[0], int) \
                    else list(itertools.chain.from_iterable(pylist))
        assert max(flattened) <= 4294967295, \
            'List has value %s' % max(flattened)
        assert min(flattened) >= 0

        current_device[varname] = np.array(pylist, dtype='uint32')

        continue

      match = RE_ENDIF.match(line)
      if match:
        current_device = None

  return tables

def _parse_int_bool(val):
  if not val:
    return True

  return int(val, 0)



if __name__ == '__main__':
  defsh = os.path.join(ARGS.SampleCode_dir, 'defs.h')
  defines = parse_defs(defsh)

  tablec = os.path.join(ARGS.SampleCode_dir, 'Table.c')
  tables = parse_tables(tablec)

  if not ARGS.device_name:
    tables_list_by_tablenum = {}

    for k, v in defines.items():
      if k != '_ALL':
        tablenum = v.get('TABLENUMBER')
        if tablenum not in tables_list_by_tablenum:
          tables_list_by_tablenum[tablenum] = []

        tables_list_by_tablenum[tablenum].append(k)

    pprint(tables_list_by_tablenum)
  else:
    device = ARGS.device_name

    if device not in tables or device not in defines:
      print('Defs or Table for {} not found'.format(device))
    else:
      device_defs = defines[device]
      table = tables[device]
      pprint(device_defs)
      pprint(table)

      # Ensure our parsed table sizes match the lengths defined
      assert device_defs['TABLENUMBER']
      assert table['TempTable'].shape == (device_defs['NROFADELEMENTS'], device_defs['NROFTAELEMENTS'])

      # NROF*ELEMENTS is used in defs.h to define the size of the array, and the
      #   C code to loop through the array. However, there are instances where
      #   the number of elements defined in defs.h is smaller than the defined
      #   size. I guess that's OK.
      assert len(table['XTATemps']) <= device_defs['NROFTAELEMENTS']
      assert len(table['YADValues']) <= device_defs['NROFADELEMENTS']


      fname = os.path.join(DATA_DIR, device)
      np.savez_compressed(fname, table=table['TempTable'],
                          ta_axes=table['XTATemps'], dk_axes=table['YADValues'],
                          metadata=device_defs)

      print('{} saved'.format(fname))