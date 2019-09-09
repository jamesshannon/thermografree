import numpy as np
import pytest

from thermografree import lookup_tables

def test_load_htpa_l2_1_datafile():
  """Ensure that the HTPA datafile can be loaded and passes internal checks."""
  ta_axes, dk_axes, table, offset = lookup_tables.get_table_and_axes(
      'HTPA32x32dR1L2_1HiSiF5_0_Gain3k3_Extended', 114)

  assert table.shape == (1595, 12)


def test_lookup_table_spreadsheet_interpolate():
  """Test the lookup table against the example lookup table spreadsheet."""
  v_vdd_comp = np.full((32,32), 182)
  a = lookup_tables.interpolate_tables(3000, v_vdd_comp,
                                       'lookup_table_example', None)
  assert a[0][0] == pytest.approx(4026, rel=0.5)


@pytest.mark.skip
def test_lookup_table_datasheet():
  # The datasheet appears wrong. The "object temperature" is the same as vx.
  # Email to Boselec is pending
  v_vdd_comp = np.full((32,32), 182)
  a = lookup_tables.interpolate_tables(3000, v_vdd_comp,
                                       'lookup_table_example', None)
  assert a[0][0] == 3941
