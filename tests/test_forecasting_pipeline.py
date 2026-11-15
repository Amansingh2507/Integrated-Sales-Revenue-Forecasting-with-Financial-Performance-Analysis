"""
Unit tests for forecasting models, horizon validation, and accuracy metrics.
Compatible with both pytest and python -m unittest.
"""

import unittest
import pandas as pd
import numpy as np
from pathlib import Path

class TestForecastingPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = Path(__file__).resolve().parent.parent
        cls.models_dir = cls.root / 'models'
        cls.reports_dir = cls.root / 'reports'
        cls.df_forecasts = pd.read_csv(cls.models_dir / 'final_forecast_outputs.csv')
        cls.df_scorecard = pd.read_csv(cls.models_dir / 'master_model_scorecard.csv')

    def test_forecast_horizon(self):
        """Verify the forecast horizon is exactly 28 days."""
        self.assertEqual(len(self.df_forecasts), 28, "Forecast horizon must be exactly 28 days")

    def test_non_negative_predictions(self):
        """Assert all model forecasts satisfy retail non-negativity constraint."""
        for col in ['Random_Forest', 'SARIMAX', 'Holt_Winters', 'Naive']:
            if col in self.df_forecasts.columns:
                negative_count = (self.df_forecasts[col] < 0).sum()
                self.assertEqual(negative_count, 0, f"Model {col} predicted negative units")

    def test_scorecard_integrity(self):
        """Verify all 11 models are evaluated in the master scorecard."""
        self.assertEqual(len(self.df_scorecard), 11, "Expected 11 benchmarked models")
        required_cols = ['Rank', 'Model', 'WAPE (%)', 'RMSE (Units)', 'MAE (Units)', 'Forecast Bias (%)']
        for col in required_cols:
            self.assertIn(col, self.df_scorecard.columns, f"Missing metric column {col}")

    def test_champion_model_accuracy(self):
        """Verify champion model achieves industry-standard WAPE < 15%."""
        champion = self.df_scorecard.iloc[0]
        self.assertLess(champion['WAPE (%)'], 15.0, "Champion model WAPE must be < 15%")
        self.assertLess(abs(champion['Forecast Bias (%)']), 5.0, "Champion bias must be within +/- 5%")

if __name__ == '__main__':
    unittest.main()
