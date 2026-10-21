"""
analytics.py - shared analysis engine.
Imported by BOTH the Jupyter notebook and the Streamlit dashboard,
so every number you see in either one comes from the same code.
"""
import os
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm
from statsmodels.tsa.holtwinters import ExponentialSmoothing

REGIONS = ["East", "West", "North", "South"]
SEGMENTS = ["Consumer", "Corporate", "Home Office"]
CATEGORIES = ["Furniture", "Technology", "Office Supplies"]


# ----------------------------------------------------------------- DATA
def generate_data(n=3000, seed=42):
    """Synthetic Superstore-style marketing data (used if no CSV is given)."""
    rng = np.random.default_rng(seed)
    dates = pd.to_datetime("2022-01-01") + pd.to_timedelta(rng.integers(0, 1095, n), unit="D")
    df = pd.DataFrame({
        "order_id": np.arange(1, n + 1),
        "order_date": dates,
        "region": rng.choice(REGIONS, n, p=[0.3, 0.3, 0.2, 0.2]),
        "segment": rng.choice(SEGMENTS, n, p=[0.5, 0.3, 0.2]),
        "category": rng.choice(CATEGORIES, n, p=[0.25, 0.25, 0.5]),
        "group": rng.choice(["Control", "Campaign"], n),
        "ad_spend": rng.uniform(50, 1000, n).round(2),
        "discount": rng.choice([0, 0.05, 0.1, 0.2, 0.3], n, p=[0.35, 0.2, 0.2, 0.15, 0.1]),
    })
    region_eff = df["region"].map({"East": 40, "West": 25, "North": 0, "South": -20})
    seg_eff = df["segment"].map({"Consumer": 0, "Corporate": 30, "Home Office": 10})
    cat_eff = df["category"].map({"Furniture": 20, "Technology": 60, "Office Supplies": -15})
    season = 25 * np.sin((df["order_date"].dt.month - 3) / 12 * 2 * np.pi)
    trend = (df["order_date"] - df["order_date"].min()).dt.days * 0.04
    campaign = np.where(df["group"] == "Campaign", 35, 0)
    df["sales"] = (250 + region_eff + seg_eff + cat_eff + season + trend + campaign
                   + 0.12 * df["ad_spend"] - 180 * df["discount"]
                   + rng.normal(0, 45, n)).clip(lower=20).round(2)
    df["profit"] = (df["sales"] * (0.25 - 0.5 * df["discount"]) + rng.normal(0, 8, n)).round(2)
    return df.sort_values("order_date").reset_index(drop=True)


def load_data(path=None):
    """Load your own CSV (needs the same column names) or fall back to synthetic data."""
    if path and os.path.exists(path):
        df = pd.read_csv(path, parse_dates=["order_date"])
    else:
        df = generate_data()
    df["month"] = df["order_date"].dt.to_period("M").dt.to_timestamp()
    df["year"] = df["order_date"].dt.year
    return df


# ------------------------------------------------- UNIT 1: DESCRIPTIVE
def kpis(df):
    return {
        "Total Sales": df["sales"].sum(),
        "Total Profit": df["profit"].sum(),
        "Orders": len(df),
        "Avg Order Value": df["sales"].mean(),
        "Profit Margin %": 100 * df["profit"].sum() / df["sales"].sum(),
    }


def summary_by(df, col):
    return (df.groupby(col)
              .agg(orders=("sales", "size"), sales=("sales", "sum"),
                   avg_sales=("sales", "mean"), profit=("profit", "sum"))
              .round(2).sort_values("sales", ascending=False))


# ------------------------------------------- UNIT 2: SAMPLING / CLT
def sample_means(df, col="sales", n=30, reps=1000, seed=1):
    rng = np.random.default_rng(seed)
    vals = df[col].to_numpy()
    return np.array([rng.choice(vals, n, replace=False).mean() for _ in range(reps)])


def confidence_interval(x, conf=0.95):
    x = np.asarray(x)
    m, se = x.mean(), stats.sem(x)
    lo, hi = stats.t.interval(conf, len(x) - 1, loc=m, scale=se)
    return m, lo, hi


def required_sample_size(sigma, margin, conf=0.95):
    z = stats.norm.ppf(1 - (1 - conf) / 2)
    return int(np.ceil((z * sigma / margin) ** 2))


def mle_normal(x):
    """MLE for a normal distribution (mean, std) - matches Unit 2 syllabus."""
    return float(np.mean(x)), float(np.std(x, ddof=0))


# ------------------------------------------- UNIT 3: HYPOTHESIS TEST
def cohens_d(a, b):
    na, nb = len(a), len(b)
    pooled = np.sqrt(((na - 1) * np.var(a, ddof=1) + (nb - 1) * np.var(b, ddof=1)) / (na + nb - 2))
    return (np.mean(a) - np.mean(b)) / pooled


def campaign_test(df, alpha=0.05):
    a = df.loc[df["group"] == "Campaign", "sales"]
    b = df.loc[df["group"] == "Control", "sales"]
    t, p_two = stats.ttest_ind(a, b, equal_var=False)
    p_one = p_two / 2 if t > 0 else 1 - p_two / 2   # H1: Campaign > Control
    return {"campaign_mean": a.mean(), "control_mean": b.mean(), "t_stat": t,
            "p_value_one_tailed": p_one, "cohens_d": cohens_d(a, b),
            "reject_H0": bool(p_one < alpha)}


def z_test_mean(x, mu0, sigma):
    z = (np.mean(x) - mu0) / (sigma / np.sqrt(len(x)))
    return z, 2 * (1 - stats.norm.cdf(abs(z)))


def power_of_test(effect_size, n_per_group, alpha=0.05):
    from statsmodels.stats.power import TTestIndPower
    return TTestIndPower().power(effect_size=effect_size, nobs1=n_per_group, alpha=alpha)


# --------------------------------------------------- UNIT 4: ANOVA
def anova_oneway(df, by="region"):
    groups = [g["sales"].values for _, g in df.groupby(by)]
    f, p = stats.f_oneway(*groups)
    return f, p


def anova_twoway(df, a="region", b="segment"):
    model = smf.ols(f"sales ~ C({a}) * C({b})", data=df).fit()
    return anova_lm(model, typ=2).round(4)


# ------------------------------------------- UNIT 6: REGRESSION / FORECAST
def regression(df):
    model = smf.ols("sales ~ ad_spend + discount + C(region) + C(segment) + C(category)", data=df).fit()
    return model


def monthly_sales(df):
    return df.groupby("month")["sales"].sum().asfreq("MS")


def forecast_sales(df, periods=6):
    ts = monthly_sales(df)
    fit = ExponentialSmoothing(ts, trend="add", seasonal="add", seasonal_periods=12).fit()
    fc = fit.forecast(periods)
    return ts, fc
