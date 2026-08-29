"""
Price Elasticity & Pricing Optimization Model
-----------------------------------------------
Estimates product-level price elasticity from weekly sales data using a
log-log regression model that controls for trend, seasonality, and
promotional events. Uses the fitted elasticity to recommend a price that
maximises revenue, subject to a simple business constraint (max discount).

This is a portfolio / demo version built on synthetic data. It reproduces
the same modelling approach used in a real pricing project, without using
any confidential company data.
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm


def load_data(path: str = "sample_sales_data.csv") -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["week_start"])
    df["log_price"] = np.log(df["price"])
    df["log_units"] = np.log(df["units_sold"])
    df["trend"] = np.arange(len(df))
    df["season_sin"] = np.sin(2 * np.pi * df["week_of_year"] / 52)
    df["season_cos"] = np.cos(2 * np.pi * df["week_of_year"] / 52)
    return df


def fit_elasticity_model(df: pd.DataFrame):
    """
    Log-log regression: log(units) ~ log(price) + trend + seasonality + promotion
    The coefficient on log(price) is the price elasticity of demand.
    """
    X = df[["log_price", "trend", "season_sin", "season_cos", "on_promotion"]]
    X = sm.add_constant(X)
    y = df["log_units"]

    model = sm.OLS(y, X).fit()
    return model


def recommend_price(model, df: pd.DataFrame, current_price: float,
                     max_discount_pct: float = 0.15, max_increase_pct: float = 0.10) -> dict:
    """
    Given a fitted elasticity model, searches over a bounded price range
    (business constraint: don't discount more than max_discount_pct or
    raise price more than max_increase_pct) for the revenue-maximising price.
    """
    elasticity = model.params["log_price"]

    low = current_price * (1 - max_discount_pct)
    high = current_price * (1 + max_increase_pct)
    candidate_prices = np.linspace(low, high, 200)

    # baseline demand at the current price (using the most recent row's other features)
    last_row = df.iloc[-1]
    baseline_log_units = (
        model.params["const"]
        + model.params["trend"] * last_row["trend"]
        + model.params["season_sin"] * last_row["season_sin"]
        + model.params["season_cos"] * last_row["season_cos"]
        + model.params["on_promotion"] * 0
    )

    predicted_units = np.exp(
        baseline_log_units + elasticity * (np.log(candidate_prices) - np.log(current_price))
    )
    predicted_revenue = candidate_prices * predicted_units

    best_idx = np.argmax(predicted_revenue)

    return {
        "elasticity": round(elasticity, 3),
        "current_price": current_price,
        "recommended_price": round(candidate_prices[best_idx], 2),
        "expected_units": round(predicted_units[best_idx]),
        "expected_revenue": round(predicted_revenue[best_idx], 2),
        "current_expected_revenue": round(current_price * np.exp(baseline_log_units), 2),
    }


def main():
    df = load_data()
    model = fit_elasticity_model(df)

    print(model.summary())
    print()

    current_price = df["price"].iloc[-1]
    result = recommend_price(model, df, current_price)

    print("Pricing Recommendation")
    print("-----------------------")
    for k, v in result.items():
        print(f"{k}: {v}")

    # Export results for stakeholder review, mirroring the automated Excel
    # output used in the real project.
    pd.DataFrame([result]).to_csv("pricing_recommendation_output.csv", index=False)
    print("\nSaved -> pricing_recommendation_output.csv")


if __name__ == "__main__":
    main()
