# BAV Marketing Analytics

A marketing and sales analytics project with a Jupyter notebook and a Streamlit
dashboard. Both use `analytics.py` for the shared analysis logic.

## Features

- Synthetic Superstore-style data is generated automatically when
  `superstore.csv` is not present.
- Descriptive statistics, sampling and confidence intervals.
- Campaign hypothesis testing, ANOVA, regression, and sales forecasting.
- Interactive dashboard filters for region, segment, category, and year.

## Setup

Requires Python 3.10 or later.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Run the dashboard

```powershell
streamlit run dashboard.py
```

Open the local URL printed by Streamlit (usually `http://localhost:8501`).

## Run the notebook

```powershell
jupyter notebook BAV_Marketing_Analytics-3.ipynb
```

The notebook expects `analytics.py` to be in the same directory.

## Use your own data

Place a `superstore.csv` file beside the Python files to use it in place of the
generated data. It must contain the columns used by the analysis, including
`order_date`, `region`, `segment`, `category`, `group`, `ad_spend`, `discount`,
`sales`, and `profit`. Keep private or licensed datasets out of this repository.
