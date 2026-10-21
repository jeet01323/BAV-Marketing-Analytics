"""Streamlit dashboard. Run:  streamlit run dashboard.py
Uses the same analytics.py module as the notebook."""
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import analytics as an

st.set_page_config(page_title="Marketing & Sales Analytics", layout="wide")
st.title("📊 Marketing & Sales Analytics Dashboard")

@st.cache_data
def get_data():
    return an.load_data("superstore.csv")   # put your own CSV here, else synthetic data is used

df_all = get_data()

# ---- sidebar filters
st.sidebar.header("Filters")
regions = st.sidebar.multiselect("Region", an.REGIONS, default=an.REGIONS)
segments = st.sidebar.multiselect("Segment", an.SEGMENTS, default=an.SEGMENTS)
cats = st.sidebar.multiselect("Category", an.CATEGORIES, default=an.CATEGORIES)
years = st.sidebar.multiselect("Year", sorted(df_all["year"].unique()), default=sorted(df_all["year"].unique()))
df = df_all[df_all["region"].isin(regions) & df_all["segment"].isin(segments)
            & df_all["category"].isin(cats) & df_all["year"].isin(years)]
if df.empty:
    st.warning("No data for the selected filters.")
    st.stop()

tabs = st.tabs(["Overview", "Sampling & CLT", "Hypothesis Test", "ANOVA", "Regression", "Forecast"])

# ---- Overview (Unit 1 + 5)
with tabs[0]:
    k = an.kpis(df)
    c = st.columns(5)
    c[0].metric("Total Sales", f"{k['Total Sales']:,.0f}")
    c[1].metric("Total Profit", f"{k['Total Profit']:,.0f}")
    c[2].metric("Orders", f"{k['Orders']:,}")
    c[3].metric("Avg Order Value", f"{k['Avg Order Value']:,.1f}")
    c[4].metric("Profit Margin", f"{k['Profit Margin %']:.1f}%")
    l, r = st.columns(2)
    l.plotly_chart(px.line(df.groupby("month")["sales"].sum().reset_index(), x="month", y="sales",
                           title="Monthly Sales Trend"), use_container_width=True)
    r.plotly_chart(px.bar(an.summary_by(df, "region").reset_index(), x="region", y="sales",
                          color="region", title="Sales by Region"), use_container_width=True)
    l, r = st.columns(2)
    l.plotly_chart(px.box(df, x="category", y="sales", color="category", title="Sales Distribution by Category"),
                   use_container_width=True)
    pivot = df.pivot_table(index="region", columns="segment", values="sales", aggfunc="mean").round(1)
    r.plotly_chart(px.imshow(pivot, text_auto=True, title="Avg Sales: Region x Segment"), use_container_width=True)

# ---- Sampling (Unit 2)
with tabs[1]:
    n = st.slider("Sample size (n)", 5, 200, 30)
    reps = st.slider("Number of samples", 100, 3000, 1000, step=100)
    means = an.sample_means(df, "sales", n, reps)
    l, r = st.columns(2)
    l.plotly_chart(px.histogram(df, x="sales", nbins=40, title="Population distribution"), use_container_width=True)
    r.plotly_chart(px.histogram(x=means, nbins=40, title=f"Sampling distribution of the mean (n={n})"),
                   use_container_width=True)
    m, lo, hi = an.confidence_interval(df["sales"].sample(min(n, len(df)), random_state=1))
    st.info(f"Population mean = {df['sales'].mean():.2f} | Mean of sample means = {means.mean():.2f} | "
            f"95% CI from one sample: ({lo:.2f}, {hi:.2f})")

# ---- Hypothesis (Unit 3)
with tabs[2]:
    st.markdown("**H0:** Campaign mean sales = Control mean sales  |  **H1:** Campaign > Control")
    alpha = st.select_slider("Significance level α", [0.01, 0.05, 0.10], value=0.05)
    res = an.campaign_test(df, alpha)
    c = st.columns(4)
    c[0].metric("Campaign mean", f"{res['campaign_mean']:.2f}")
    c[1].metric("Control mean", f"{res['control_mean']:.2f}")
    c[2].metric("p-value (1-tailed)", f"{res['p_value_one_tailed']:.4g}")
    c[3].metric("Cohen's d", f"{res['cohens_d']:.2f}")
    (st.success if res["reject_H0"] else st.warning)(
        "Reject H0: the campaign significantly increases sales." if res["reject_H0"]
        else "Fail to reject H0: no significant evidence the campaign helps.")
    st.plotly_chart(px.box(df, x="group", y="sales", color="group"), use_container_width=True)

# ---- ANOVA (Unit 4)
with tabs[3]:
    by = st.selectbox("Compare mean sales across", ["region", "segment", "category"])
    f, p = an.anova_oneway(df, by)
    st.write(f"**One-way ANOVA** → F = {f:.3f}, p = {p:.4g}  "
             f"({'significant difference' if p < 0.05 else 'no significant difference'})")
    st.plotly_chart(px.violin(df, x=by, y="sales", color=by, box=True), use_container_width=True)
    st.write("**Two-way ANOVA (Region × Segment)**")
    st.dataframe(an.anova_twoway(df))

# ---- Regression (Unit 6)
with tabs[4]:
    model = an.regression(df)
    st.write(f"**R² = {model.rsquared:.3f}**, Adj. R² = {model.rsquared_adj:.3f}")
    st.dataframe(model.summary2().tables[1].round(4))
    plot_df = pd.DataFrame({"Actual": df["sales"], "Predicted": model.fittedvalues})
    st.plotly_chart(px.scatter(plot_df, x="Actual", y="Predicted", opacity=0.4, title="Actual vs Predicted Sales"),
                    use_container_width=True)

# ---- Forecast (Unit 6)
with tabs[5]:
    periods = st.slider("Months to forecast", 3, 12, 6)
    try:
        ts, fc = an.forecast_sales(df, periods)
        fig = px.line(ts.reset_index(), x="month", y="sales", title="Monthly sales & forecast")
        fig.add_scatter(x=fc.index, y=fc.values, mode="lines+markers", name="Forecast", line=dict(dash="dash"))
        st.plotly_chart(fig, use_container_width=True)
    except Exception:
        st.warning("Need at least 24 months of data (select all years) for the seasonal forecast.")
