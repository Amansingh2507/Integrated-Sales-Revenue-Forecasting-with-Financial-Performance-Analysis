"""
Unit tests for data loading, wide-to-long transformation, and feature engineering.
Compatible with both pytest and python -m unittest.
"""

import unittest
import pandas as pd
import numpy as np
from pathlib import Path

class TestDataPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = Path(__file__).resolve().parent.parent
        cls.raw_dir = cls.root / 'data' / 'raw'
        cls.proc_dir = cls.root / 'data' / 'processed'

    def test_raw_files_exist(self):
        """Verify that all raw scoped data files are present."""
        self.assertTrue((self.raw_dir / 'sales_pilot_scoped.csv').exists(), "sales_pilot_scoped.csv missing")
        self.assertTrue((self.raw_dir / 'prices_pilot_scoped.csv').exists(), "prices_pilot_scoped.csv missing")
        self.assertTrue((self.raw_dir / 'calendar_pilot_scoped.csv').exists(), "calendar_pilot_scoped.csv missing")

    def test_sales_dimensions(self):
        """Verify pilot sales dataset matches 30 SKUs and 1913 historical days."""
        df_sales = pd.read_csv(self.raw_dir / 'sales_pilot_scoped.csv')
        self.assertEqual(len(df_sales), 30, "Expected 30 pilot products")
        d_cols = [c for c in df_sales.columns if c.startswith('d_')]
        self.assertEqual(len(d_cols), 1913, "Expected 1913 historical daily sales columns")

    def test_calendar_continuity(self):
        """Verify zero missing dates in the calendar table."""
        df_cal = pd.read_csv(self.raw_dir / 'calendar_pilot_scoped.csv')
        dates = pd.to_datetime(df_cal['date']).sort_values()
        expected = pd.date_range(start=dates.iloc[0], end=dates.iloc[-1])
        self.assertEqual(len(dates), len(expected), "Calendar contains date gaps")
        self.assertEqual(df_cal['date'].duplicated().sum(), 0, "Calendar has duplicate dates")

    def test_daily_store_revenue(self):
        """Verify daily store aggregated dataset."""
        df_daily = pd.read_csv(self.proc_dir / 'daily_store_revenue.csv')
        self.assertEqual(len(df_daily), 1913, "Expected 1913 daily aggregate records")
        self.assertTrue((df_daily['total_units'] < 0).sum() == 0, "Found negative sales units")
        self.assertTrue((df_daily['total_revenue'] < 0).sum() == 0, "Found negative revenue values")
        self.assertTrue((df_daily['total_revenue'] == 0).sum() > 0, "Expected 0 revenue on Christmas store closures")

if __name__ == '__main__':
    unittest.main()
