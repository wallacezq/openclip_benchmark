"""
product_db.py
Thin wrapper around the product price CSV database.
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

DEFAULT_CSV = Path("product_prices.csv")


class ProductDatabase:
    """Loads product prices from CSV and provides price lookups by short name."""

    def __init__(self, csv_path: Path = DEFAULT_CSV) -> None:
        self._csv_path = csv_path
        df = pd.read_csv(csv_path)
        df.columns = ["Product", "Price"]
        # Index by lower-case product name for case-insensitive lookup
        self._prices: dict[str, float] = {
            row["Product"].strip(): float(row["Price"])
            for _, row in df.iterrows()
        }
        logger.info("Loaded %d products from %s", len(self._prices), csv_path)

    def get_price(self, product_name: str) -> float | None:
        """Return unit price for *product_name*, or None if not found."""
        return self._prices.get(product_name.strip())

    def list_products(self) -> list[dict]:
        """Return all products as a list of {product, price} dicts."""
        return [
            {"product": name, "price": price}
            for name, price in self._prices.items()
        ]

    def add_product(self, product_name: str, price: float) -> bool:
        """Add a new product. Returns False if it already exists."""
        key = product_name.strip()
        if key in self._prices:
            return False
        self._prices[key] = round(price, 2)
        self._save()
        logger.info("Added product '%s' with price %.2f", key, price)
        return True

    def update_price(self, product_name: str, new_price: float) -> bool:
        """Update price for *product_name*. Returns False if not found."""
        key = product_name.strip()
        if key not in self._prices:
            return False
        self._prices[key] = round(new_price, 2)
        self._save()
        logger.info("Updated price for '%s' to %.2f", key, new_price)
        return True

    def _save(self) -> None:
        """Persist current prices back to the CSV file."""
        df = pd.DataFrame(
            [(name, price) for name, price in self._prices.items()],
            columns=["Product", "Price"],
        )
        df.to_csv(self._csv_path, index=False)

    def build_bill(
        self, product_name: str, quantity: int = 1
    ) -> dict | None:
        """
        Build a bill line item dict for *product_name*.
        Returns None if the product is not in the database.
        """
        price = self.get_price(product_name)
        if price is None:
            logger.warning("Product '%s' not found in price database", product_name)
            return None
        return {
            "Product": product_name,
            "Quantity": quantity,
            "Unit_Price": price,
            "Total": round(price * quantity, 2),
        }
