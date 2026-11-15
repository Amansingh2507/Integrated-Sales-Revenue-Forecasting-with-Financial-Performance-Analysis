"""
Unit tests for SQLite Data Warehouse schema, table records, and referential integrity.
Compatible with both pytest and python -m unittest.
"""

import unittest
import sqlite3
import pandas as pd
from pathlib import Path

class TestDatabaseIntegrity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = Path(__file__).resolve().parent.parent
        cls.db_path = cls.root / 'data' / 'processed' / 'retail_analytics.db'
        cls.conn = sqlite3.connect(cls.db_path)

    @classmethod
    def tearDownClass(cls):
        cls.conn.close()

    def test_database_exists(self):
        """Verify retail_analytics.db exists and is non-empty."""
        self.assertTrue(self.db_path.exists(), "retail_analytics.db does not exist")
        self.assertGreater(self.db_path.stat().st_size, 10000, "Database file is too small")

    def test_tables_present(self):
        """Verify all star-schema tables exist."""
        tables = pd.read_sql_query("SELECT name FROM sqlite_master WHERE type='table';", self.conn)['name'].tolist()
        expected = ['dim_calendar', 'dim_product', 'dim_store', 'fact_prices', 'fact_sales', 'fact_financials']
        for tbl in expected:
            self.assertIn(tbl, tables, f"Missing table: {tbl}")

    def test_fact_sales_record_count(self):
        """Verify fact_sales contains 57,390 transaction records."""
        count = pd.read_sql_query("SELECT COUNT(*) AS n FROM fact_sales;", self.conn)['n'].iloc[0]
        self.assertEqual(count, 57390, "Expected 57,390 transaction records in fact_sales")

    def test_referential_integrity(self):
        """Verify all sales foreign keys exist in dimension tables."""
        orphan_items = pd.read_sql_query("""
            SELECT COUNT(*) AS n FROM fact_sales s
            LEFT JOIN dim_product p ON s.item_id = p.item_id
            WHERE p.item_id IS NULL;
        """, self.conn)['n'].iloc[0]
        self.assertEqual(orphan_items, 0, "Found orphan product keys in fact_sales")

        orphan_dates = pd.read_sql_query("""
            SELECT COUNT(*) AS n FROM fact_sales s
            LEFT JOIN dim_calendar c ON s.date = c.date
            WHERE c.date IS NULL;
        """, self.conn)['n'].iloc[0]
        self.assertEqual(orphan_dates, 0, "Found orphan calendar date keys in fact_sales")

if __name__ == '__main__':
    unittest.main()
