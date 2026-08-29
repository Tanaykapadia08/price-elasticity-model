"""
Generates synthetic weekly sales data for a fictional consumer goods product.
This mimics the structure of real retail sales data (price, units sold, promotions,
seasonality) without using any real company or retailer information.
"""

import numpy as np
import pandas as pd

np.random.seed(42)

N_WEEKS = 104  # two years of weekly data
BASE_PRICE = 29.99
BASE_DEMAND = 500
TRUE_ELASTICITY = -1.8  # ground truth, used only to generate realistic data

dates = pd.date_range("2024-01-01", periods=N_WEEKS, freq="W-MON")

# Simulate a price that occasionally changes (promotions / price tests)
price = np.full(N_WEEKS, BASE_PRICE)
price += np.random.choice([0, -3, -5, 2], size=N_WEEKS, p=[0.6, 0.2, 0.1, 0.1])

# Seasonality: simple yearly sine wave + a spike around weeks 46-48 (holiday season)
week_of_year = dates.isocalendar().week.values
seasonality = 1 + 0.25 * np.sin(2 * np.pi * week_of_year / 52)
holiday_boost = np.where((week_of_year >= 46) & (week_of_year <= 48), 1.4, 1.0)

# Promotion flag (event variable)
on_promotion = (price < BASE_PRICE).astype(int)

# Demand via a log-log elasticity model + noise
price_ratio = price / BASE_PRICE
demand_multiplier = price_ratio ** TRUE_ELASTICITY
noise = np.random.normal(1.0, 0.07, size=N_WEEKS)

units_sold = (
    BASE_DEMAND * demand_multiplier * seasonality * holiday_boost * noise
).round().astype(int)
units_sold = np.clip(units_sold, 50, None)

df = pd.DataFrame({
    "week_start": dates,
    "price": price.round(2),
    "units_sold": units_sold,
    "on_promotion": on_promotion,
    "week_of_year": week_of_year,
})

df.to_csv("sample_sales_data.csv", index=False)
print(f"Generated {len(df)} weeks of sample sales data -> sample_sales_data.csv")
