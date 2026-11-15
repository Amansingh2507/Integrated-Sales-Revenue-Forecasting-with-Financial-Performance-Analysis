"""
Unit tests for financial calculations, COGS, Gross Profit, OPEX, EBIT, and Scenarios.
Compatible with both pytest and python -m unittest.
"""

import unittest
import pandas as pd
import numpy as np
from pathlib import Path

class TestFinancialEngine(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = Path(__file__).resolve().parent.parent
        cls.models_dir = cls.root / 'models'
        cls.reports_dir = cls.root / 'reports'
        cls.df_fin = pd.read_csv(cls.models_dir / 'final_financial_projections.csv')
        cls.df_scenarios = pd.read_csv(cls.reports_dir / '08_scenario_summary.csv')

    def test_revenue_identity(self):
        """Verify Revenue = Units_Forecast * Selling_Price."""
        calculated_rev = np.round(self.df_fin['Units_Forecast'] * self.df_fin['Selling_Price'], 2)
        diff = np.abs(self.df_fin['Revenue'] - calculated_rev).max()
        self.assertLess(diff, 0.05, "Revenue does not match Units * Price")

    def test_gross_profit_identity(self):
        """Verify Gross Profit = Revenue - COGS."""
        calculated_gp = np.round(self.df_fin['Revenue'] - self.df_fin['COGS'], 2)
        diff = np.abs(self.df_fin['Gross_Profit'] - calculated_gp).max()
        self.assertLess(diff, 0.05, "Gross Profit does not match Revenue - COGS")

    def test_operating_profit_identity(self):
        """Verify Operating Profit = Gross Profit - Total_OPEX."""
        calculated_ebit = np.round(self.df_fin['Gross_Profit'] - self.df_fin['Total_OPEX'], 2)
        diff = np.abs(self.df_fin['Operating_Profit'] - calculated_ebit).max()
        self.assertLess(diff, 0.05, "Operating Profit does not match GP - OPEX")

    def test_margin_bounds(self):
        """Verify gross and operating margins remain within logical commercial ranges."""
        self.assertTrue((self.df_fin['Gross_Margin_Pct'] >= 30.0).all(), "Gross margin fell below 30%")
        self.assertTrue((self.df_fin['Gross_Margin_Pct'] <= 50.0).all(), "Gross margin exceeded 50%")
        self.assertTrue((self.df_fin['Operating_Margin_Pct'] >= 10.0).all(), "EBIT margin fell below 10%")

    def test_scenario_ordering(self):
        """Verify Optimistic revenue > Base Case > Pessimistic revenue."""
        pess_rev = self.df_scenarios.loc[self.df_scenarios['Scenario'].str.contains('Pessimistic'), 'Gross Revenue ($)'].values[0]
        base_rev = self.df_scenarios.loc[self.df_scenarios['Scenario'].str.contains('Base'), 'Gross Revenue ($)'].values[0]
        opt_rev = self.df_scenarios.loc[self.df_scenarios['Scenario'].str.contains('Optimistic'), 'Gross Revenue ($)'].values[0]
        
        self.assertLess(pess_rev, base_rev, "Pessimistic revenue must be less than Base Case")
        self.assertLess(base_rev, opt_rev, "Base Case revenue must be less than Optimistic")

if __name__ == '__main__':
    unittest.main()
