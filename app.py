"""Task 4 - CSV to Interactive Dashboard (Streamlit + Plotly)."""
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="CSV to Dashboard", page_icon="📊", layout="wide")
st.title("📊 CSV to Interactive Dashboard")
st.caption("Upload any CSV file. Charts, filters and statistics update dynamically.")


def read_csv(file):
    try:
        return pd.read_csv(file)
    except UnicodeDecodeError:
        file.seek(0)
        return pd.read_csv(file, encoding="latin-1")


@st.cache_data
def load_sample():
    p = Path(__file__).parent / "data" / "sample_iris.csv"
    if p.exists():
        return pd.read_csv(p)
    from sklearn.datasets import load_iris
    d = load_iris(as_frame=True)
    df = d.frame.drop(columns=["target"])
    df["species"] = [d.target_names[i] for i in d.target]
    return df


up = st.sidebar.file_uploader("Upload a CSV file", type=["csv"])
df = read_csv(up) if up is not None else load_sample()
st.sidebar.success("Using uploaded file" if up is not None else "Using the built-in sample (Iris)")

# ---------- sidebar filters ----------
st.sidebar.header("Filters")
filter_cols = st.sidebar.multiselect("Filter by column(s)", list(df.columns))
view = df.copy()
for c in filter_cols:
    if pd.api.types.is_numeric_dtype(view[c]):
        lo, hi = float(view[c].min()), float(view[c].max())
        if lo < hi:
            a, b = st.sidebar.slider(c, lo, hi, (lo, hi))
            view = view[(view[c] >= a) & (view[c] <= b)]
    else:
        opts = sorted(view[c].dropna().astype(str).unique().tolist())
        chosen = st.sidebar.multiselect(c, opts, default=opts)
        view = view[view[c].astype(str).isin(chosen)]

num_cols = view.select_dtypes(include="number").columns.tolist()
cat_cols = [c for c in view.columns if c not in num_cols]

tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Charts", "Correlation", "Data"])

with tab1:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", f"{len(view):,}")
    c2.metric("Columns", view.shape[1])
    c3.metric("Missing cells", int(view.isna().sum().sum()))
    c4.metric("Duplicate rows", int(view.duplicated().sum()))
    st.subheader("Preview")
    st.dataframe(view.head(50), use_container_width=True)
    st.subheader("Summary statistics")
    st.dataframe(view.describe(include="all").T, use_container_width=True)
    miss = view.isna().sum()
    miss = miss[miss > 0]
    if len(miss):
        st.subheader("Columns with missing values")
        st.bar_chart(miss)

with tab2:
    kind = st.selectbox("Chart type", ["Histogram", "Box plot", "Scatter plot", "Line chart", "Bar chart (aggregated)", "Pie chart"])
    color = st.selectbox("Colour by (optional)", ["(none)"] + list(view.columns))
    color = None if color == "(none)" else color
    try:
        if kind == "Histogram" and num_cols:
            x = st.selectbox("Column", num_cols)
            bins = st.slider("Bins", 5, 100, 30)
            fig = px.histogram(view, x=x, color=color, nbins=bins, marginal="box")
        elif kind == "Box plot" and num_cols:
            y = st.selectbox("Numeric column", num_cols)
            x = st.selectbox("Group by (optional)", ["(none)"] + cat_cols)
            fig = px.box(view, y=y, x=None if x == "(none)" else x, color=color)
        elif kind == "Scatter plot" and len(num_cols) >= 2:
            x = st.selectbox("X axis", num_cols, index=0)
            y = st.selectbox("Y axis", num_cols, index=1)
            fig = px.scatter(view, x=x, y=y, color=color, trendline=None, hover_data=view.columns[:6].tolist())
        elif kind == "Line chart" and num_cols:
            x = st.selectbox("X axis", list(view.columns))
            y = st.selectbox("Y axis", num_cols)
            fig = px.line(view.sort_values(x), x=x, y=y, color=color)
        elif kind == "Bar chart (aggregated)" and cat_cols and num_cols:
            x = st.selectbox("Category", cat_cols)
            y = st.selectbox("Value", num_cols)
            agg = st.selectbox("Aggregation", ["sum", "mean", "median", "count", "max", "min"])
            g = view.groupby(x)[y].agg(agg).reset_index().sort_values(y, ascending=False).head(30)
            fig = px.bar(g, x=x, y=y)
        elif kind == "Pie chart" and cat_cols:
            names = st.selectbox("Category", cat_cols)
            g = view[names].value_counts().head(15).reset_index()
            g.columns = [names, "count"]
            fig = px.pie(g, names=names, values="count")
        else:
            fig = None
            st.info("This chart needs different column types than the data has. Try another chart type.")
        if fig is not None:
            st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.warning(f"Could not draw this chart: {e}")

with tab3:
    if len(num_cols) >= 2:
        corr = view[num_cols].corr()
        st.plotly_chart(px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1), use_container_width=True)
        pairs = corr.abs().where(~np.eye(len(corr), dtype=bool)).stack().sort_values(ascending=False)
        if len(pairs):
            a, b = pairs.index[0]
            st.info(f"Strongest relationship: {a} and {b} (r = {corr.loc[a, b]:.2f})")
    else:
        st.info("Need at least two numeric columns.")

with tab4:
    st.dataframe(view, use_container_width=True)
    st.download_button("Download filtered data as CSV", view.to_csv(index=False).encode("utf-8"), "filtered_data.csv", "text/csv")
