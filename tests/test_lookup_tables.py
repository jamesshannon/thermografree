import numpy as np
import pytest

import lookup_tables

def test_lookup_table_spreadsheet():
  v_vdd_comp = np.full((32,32), 182)
  a = lookup_tables.interpolate_tables(3000, v_vdd_comp,
                                       device='lookup_table_example')
  assert a[0][0] == pytest.approx(4026, rel=0.5)

@pytest.mark.skip
def test_lookup_table_datasheet():
  # The datasheet appears wrong. The "object temperature" is the same as vx.
  # Email to Boselec is pending
  v_vdd_comp = np.full((32,32), 182)
  a = lookup_tables.interpolate_tables(3000, v_vdd_comp,
                                       device='lookup_table_example')
  assert a[0][0] == 3941
