# Price Elasticity & Pricing Optimization Model

A Python-based model that estimates product-level price elasticity from
weekly sales data and recommends a revenue-maximising price, subject to
business constraints.

> Note: This is a portfolio version built on synthetic data. It demonstrates
> the same modelling approach I use in my work as a Data Analyst, without
> using any confidential company or retailer data.

## What it does

1. **`generate_sample_data.py`** creates two years of synthetic weekly sales
   data for a fictional product, including price changes, promotions, and
   seasonality, so the model has something realistic to fit.
2. **`price_elasticity_model.py`**:
   - Fits a log-log regression model: `log(units_sold) ~ log(price) + trend + seasonality + promotion`
   - The coefficient on `log(price)` is the estimated price elasticity of demand
   - Searches a bounded price range (e.g. max 15% discount, max 10% increase)
     to find the price that maximises expected revenue
   - Exports the recommendation to a CSV, mirroring the automated stakeholder
     reports used in the real-world version of this project

## Tools used

Python, pandas, NumPy, statsmodels (OLS regression)

## How to run

```bash
pip install pandas numpy statsmodels
python generate_sample_data.py
python price_elasticity_model.py
```

## Example output

```
Pricing Recommendation
-----------------------
elasticity: -1.582
current_price: 29.99
recommended_price: 25.49
expected_units: 148101
expected_revenue: 3775318.12
current_expected_revenue: 3434872.30
```

## Background

This approach mirrors a pricing optimization model I built as part of my
work as a Working Student Data Analyst, where I used weekly sales data to
estimate product-level price elasticity and recommend optimal prices while
applying business constraints to protect revenue and demand.
