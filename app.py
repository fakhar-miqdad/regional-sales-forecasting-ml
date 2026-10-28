"""
Regional Sales Intelligence – Streamlit ML app
Run:  streamlit run app.py
"""
import io
import os

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import ml_core as mc

st.set_page_config(page_title="Regional Sales Intelligence", page_icon="📈", layout="wide")

DEFAULT_FILE = "Denormalize_regional_sales_data.xlsx"


# ---------------------------------------------------------------- helpers
def money(x: float) -> str:
    x = float(x)
    for unit, div in (("B", 1e9), ("M", 1e6), ("K", 1e3)):
        if abs(x) >= div:
            return f"{x / div:,.2f}{unit}"
    return f"{x:,.0f}"


@st.cache_data(show_spinner="Loading & cleaning data…")
def get_data(uploaded, path):
    return mc.load_data(uploaded if uploaded is not None else path)


@st.cache_data(show_spinner=False)
def get_series(df_key: str, _df: pd.DataFrame, freq: str, value: str):
    return mc.make_series(_df, freq, value)


def to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


# ---------------------------------------------------------------- sidebar: data
st.sidebar.title("📈 Sales Intelligence")
up = st.sidebar.file_uploader("Upload sales data (.xlsx / .csv)", type=["xlsx", "csv"])
path = DEFAULT_FILE if os.path.exists(DEFAULT_FILE) else None
if up is None and path is None:
    st.info("Upload the regional sales file in the sidebar to begin "
            "(or place it next to app.py as `Denormalize_regional_sales_data.xlsx`).")
    st.stop()
try:
    full = get_data(up, path)
except Exception as e:
    st.error(f"Could not read the file: {e}")
    st.stop()

# ---------------------------------------------------------------- sidebar: filters
st.sidebar.subheader("Global filters")
f = full
if "Scenario Name" in f.columns:
    scen = st.sidebar.multiselect("Scenario", sorted(f["Scenario Name"].unique()),
                                  default=sorted(f["Scenario Name"].unique()))
    f = f[f["Scenario Name"].isin(scen)]
yr_min, yr_max = int(full["Year"].min()), int(full["Year"].max())
yrs = st.sidebar.slider("Year range", yr_min, yr_max, (yr_min, yr_max)) if yr_min < yr_max else (yr_min, yr_max)
f = f[f["Year"].between(*yrs)]
for col, label in [("Category", "Category"), ("Cities", "City"), ("Brand", "Brand")]:
    if col in f.columns:
        opts = sorted(f[col].dropna().unique())
        sel = st.sidebar.multiselect(label, opts, default=opts)
        f = f[f[col].isin(sel)]

st.sidebar.caption(f"{len(f):,} of {len(full):,} rows selected")
if f.empty:
    st.warning("No rows match the current filters.")
    st.stop()

st.title("Regional Sales Intelligence")
tab_an, tab_viz, tab_ml, tab_fc, tab_pred = st.tabs(
    ["🔍 Data Analysis", "📊 Visualization", "🤖 Predictive Analysis", "🔮 Forecasting", "🎯 Predict Sales"])

# ================================================================ 1. DATA ANALYSIS
with tab_an:
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Sales", money(f["Sales"].sum()))
    c2.metric("Units Sold", f"{int(f['Units Sold'].sum()):,}")
    c3.metric("Total Profit", money(f["Profit"].sum()) if "Profit" in f else "–")
    c4.metric("Avg Margin", f"{f['Profit'].sum() / f['Sales'].sum() * 100:.1f}%" if "Profit" in f else "–")
    c5.metric("Transactions", f"{len(f):,}")

    st.subheader("Dataset preview")
    st.dataframe(f.head(200))
    st.download_button("⬇ Download filtered data (CSV)", to_csv_bytes(f), "filtered_sales.csv", "text/csv")

    a, b = st.columns(2)
    with a:
        st.subheader("Schema & data quality")
        q = pd.DataFrame({"dtype": f.dtypes.astype(str), "missing": f.isna().sum(),
                          "unique": f.nunique()})
        st.dataframe(q)
        st.caption(f"Duplicate rows: {int(f.duplicated().sum())}")
    with b:
        st.subheader("Statistical summary")
        st.dataframe(f.select_dtypes("number").drop(columns=["Index"], errors="ignore").describe().T.round(2))

    st.subheader("Group-by explorer")
    g1, g2, g3 = st.columns(3)
    cat_cols_all = [c for c in ["Category", "Brand", "Cities", "Product Name", "SalesPerson Name",
                                "Designation", "Warranty", "Scenario Name", "Year", "Quarter", "MonthName"]
                    if c in f.columns]
    gby = g1.multiselect("Group by", cat_cols_all, default=["Category"])
    metric_cols = [c for c in ["Sales", "Units Sold", "Profit", "Cost of sales"] if c in f.columns]
    gm = g2.multiselect("Metrics", metric_cols, default=["Sales", "Profit"])
    agg = g3.selectbox("Aggregation", ["sum", "mean", "median", "max", "min", "count"])
    if gby and gm:
        t = f.groupby(gby)[gm].agg(agg).reset_index().sort_values(gm[0], ascending=False)
        st.dataframe(t)

    st.subheader("Actual vs Budget")
    if "Scenario Name" in f.columns and f["Scenario Name"].nunique() > 1:
        sc = f.pivot_table(index="Year", columns="Scenario Name", values="Sales", aggfunc="sum")
        if {"Actual", "Budget"} <= set(sc.columns):
            sc["Variance"] = sc["Actual"] - sc["Budget"]
            sc["Variance %"] = (sc["Variance"] / sc["Budget"] * 100).round(2)
        st.dataframe(sc.style.format("{:,.2f}"))
    else:
        st.caption("Select both Actual and Budget scenarios in the sidebar to compare.")

    st.subheader("Correlation matrix")
    num_only = f[[c for c in ["Units Sold", "Sales", "Cost of sales", "Sale price", "Cost price",
                              "Profit", "Age", "Tenure (yrs)"] if c in f.columns]]
    st.plotly_chart(px.imshow(num_only.corr().round(2), text_auto=True, aspect="auto",
                              color_continuous_scale="RdBu_r", zmin=-1, zmax=1))

# ================================================================ 2. VISUALIZATION
with tab_viz:
    freq_label = st.radio("Time granularity", ["Monthly", "Quarterly", "Yearly"], horizontal=True)
    rule = {"Monthly": "MS", "Quarterly": "QS", "Yearly": "YS"}[freq_label]
    split = st.selectbox("Split lines by", ["None", "Scenario Name", "Category", "Cities", "Brand"])
    ts = f.set_index("Date")
    if split == "None":
        tdf = ts["Sales"].resample(rule).sum().reset_index()
        fig = px.line(tdf, x="Date", y="Sales", markers=True, title=f"{freq_label} sales trend")
    else:
        tdf = ts.groupby(split)["Sales"].resample(rule).sum().reset_index()
        fig = px.line(tdf, x="Date", y="Sales", color=split, title=f"{freq_label} sales by {split}")
    st.plotly_chart(fig)

    c1, c2 = st.columns(2)
    with c1:
        d = f.groupby("Category", as_index=False)["Sales"].sum().sort_values("Sales")
        st.plotly_chart(px.bar(d, x="Sales", y="Category", orientation="h", title="Sales by category",
                               color="Sales", color_continuous_scale="Blues"))
    with c2:
        d = f.groupby("Cities", as_index=False)["Sales"].sum()
        st.plotly_chart(px.pie(d, names="Cities", values="Sales", hole=0.45, title="Sales share by city"))

    c1, c2 = st.columns(2)
    with c1:
        d = f.groupby("Brand", as_index=False)["Sales"].sum().sort_values("Sales", ascending=False)
        st.plotly_chart(px.bar(d, x="Brand", y="Sales", title="Sales by brand", color="Sales",
                               color_continuous_scale="Teal"))
    with c2:
        d = f.groupby("Product Name", as_index=False)["Sales"].sum().nlargest(10, "Sales").sort_values("Sales")
        st.plotly_chart(px.bar(d, x="Sales", y="Product Name", orientation="h", title="Top 10 products"))

    c1, c2 = st.columns(2)
    with c1:
        hm = f.pivot_table(index="Year", columns="Month", values="Sales", aggfunc="sum")
        st.plotly_chart(px.imshow(hm, aspect="auto", color_continuous_scale="YlOrRd",
                                  labels=dict(x="Month", y="Year", color="Sales"),
                                  title="Seasonality heatmap (Year × Month)"))
    with c2:
        st.plotly_chart(px.treemap(f.groupby(["Category", "Brand"], as_index=False)["Sales"].sum(),
                                   path=["Category", "Brand"], values="Sales", title="Category → Brand"))

    c1, c2 = st.columns(2)
    with c1:
        g = f.groupby("Cities", as_index=False).agg(Sales=("Sales", "sum"), Latitude=("Latitude", "first"),
                                                    Longitude=("Longitude", "first"))
        m = px.scatter_geo(g, lat="Latitude", lon="Longitude", size="Sales", hover_name="Cities",
                           color="Sales", color_continuous_scale="Viridis", title="Sales by city (map)")
        m.update_geos(fitbounds="locations", showcountries=True, showland=True)
        st.plotly_chart(m)
    with c2:
        d = f.groupby(["SalesPerson Name", "Designation"], as_index=False)["Sales"].sum().sort_values("Sales")
        st.plotly_chart(px.bar(d, x="Sales", y="SalesPerson Name", color="Designation", orientation="h",
                               title="Salesperson performance"))

    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(px.box(f, x="Category", y="Sales", log_y=True, title="Sales distribution by category (log)"))
    with c2:
        d = f.groupby("Category", as_index=False).agg(Sales=("Sales", "sum"), Profit=("Profit", "sum"))
        d["Margin %"] = d["Profit"] / d["Sales"] * 100
        st.plotly_chart(px.bar(d.sort_values("Margin %"), x="Margin %", y="Category", orientation="h",
                               title="Profit margin by category", color="Margin %"))

    st.subheader("🎛 Custom chart builder")
    k1, k2, k3, k4 = st.columns(4)
    kind = k1.selectbox("Chart", ["Bar", "Line", "Scatter", "Box", "Histogram"])
    cats = [c for c in cat_cols_all]
    nums = [c for c in ["Sales", "Units Sold", "Profit", "Cost of sales", "Sale price", "Age"] if c in f.columns]
    xcol = k2.selectbox("X", cats + nums, index=0)
    ycol = k3.selectbox("Y", nums, index=0)
    ccol = k4.selectbox("Colour", ["None"] + cats)
    colour = None if ccol == "None" else ccol
    if kind == "Bar":
        d = f.groupby([xcol] + ([colour] if colour and colour != xcol else []), as_index=False)[ycol].sum()
        fig = px.bar(d, x=xcol, y=ycol, color=colour)
    elif kind == "Line":
        d = f.groupby([xcol] + ([colour] if colour and colour != xcol else []), as_index=False)[ycol].sum()
        fig = px.line(d.sort_values(xcol), x=xcol, y=ycol, color=colour, markers=True)
    elif kind == "Scatter":
        fig = px.scatter(f.sample(min(5000, len(f)), random_state=1), x=xcol, y=ycol, color=colour, opacity=0.5)
    elif kind == "Box":
        fig = px.box(f, x=xcol, y=ycol, color=colour)
    else:
        fig = px.histogram(f, x=ycol, color=colour, nbins=50)
    st.plotly_chart(fig)

# ================================================================ 3. PREDICTIVE ANALYSIS
with tab_ml:
    st.subheader("Train a model to predict the Sales value of a transaction")
    st.caption("Models are trained on the **full** dataset (global filters don't affect training). "
               "Hold-out uses the most recent rows when 'time-based split' is on – a realistic test of "
               "predicting the future.")

    inc_price = st.checkbox("Include 'Sale price' as a feature (⚠ leakage)", value=False,
                            help="Sales = Units Sold × Sale price exactly, so including price makes the "
                                 "task trivial. Leave off to learn from product/region/time/people instead.")
    def_num, def_cat = mc.default_feature_lists(full, inc_price)
    all_num = def_num + [c for c in ["Latitude", "Longitude"] if c in full.columns and c not in def_num]
    c1, c2 = st.columns(2)
    num_cols = c1.multiselect("Numeric features", all_num, default=def_num)
    cat_cols = c2.multiselect("Categorical features", mc.CANDIDATE_CATEGORICAL, default=def_cat)
    c1, c2, c3 = st.columns(3)
    model_name = c1.selectbox("Model", list(mc.MODEL_ZOO), index=1)
    test_size = c2.slider("Hold-out size", 0.1, 0.4, 0.2, 0.05)
    time_split = c3.checkbox("Time-based split", value=True)

    b1, b2 = st.columns(2)
    if b1.button("🚀 Train model", type="primary"):
        if not (num_cols or cat_cols):
            st.error("Pick at least one feature.")
        else:
            with st.spinner("Training…"):
                st.session_state["tr"] = mc.train_regressor(full, model_name, num_cols, cat_cols,
                                                            test_size, time_split)
                st.session_state["tr_name"] = model_name
    if b2.button("⚖ Compare all models"):
        with st.spinner("Training 3 models (subsampled for speed)…"):
            st.session_state["cmp"] = mc.compare_models(full, num_cols, cat_cols, test_size, time_split)

    if "cmp" in st.session_state:
        st.markdown("**Model comparison (lower RMSE is better)**")
        st.dataframe(st.session_state["cmp"].style.format(
            {"MAE": "{:,.0f}", "RMSE": "{:,.0f}", "R²": "{:.4f}", "MAPE %": "{:.2f}"}))

    if "tr" in st.session_state:
        tr = st.session_state["tr"]
        st.success(f"Trained: {st.session_state['tr_name']}")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("R²", f"{tr.metrics['R²']:.4f}")
        m2.metric("MAE", f"{tr.metrics['MAE']:,.0f}")
        m3.metric("RMSE", f"{tr.metrics['RMSE']:,.0f}")
        m4.metric("MAPE", f"{tr.metrics['MAPE %']:.2f}%")

        c1, c2 = st.columns(2)
        with c1:
            s = tr.test_df.sample(min(3000, len(tr.test_df)), random_state=1)
            fig = px.scatter(s, x="Actual", y="Predicted", opacity=0.4, title="Actual vs predicted (hold-out)")
            mx = float(max(s["Actual"].max(), s["Predicted"].max()))
            fig.add_trace(go.Scatter(x=[0, mx], y=[0, mx], mode="lines", name="Perfect", line=dict(dash="dash")))
            st.plotly_chart(fig)
        with c2:
            res = tr.test_df["Actual"] - tr.test_df["Predicted"]
            st.plotly_chart(px.histogram(res, nbins=60, title="Residual distribution",
                                         labels={"value": "Actual − Predicted"}))
        imp = tr.importance.head(15).sort_values("importance")
        st.plotly_chart(px.bar(imp, x="importance", y="feature", orientation="h",
                               title="Feature importance (grouped by original column)"))
        st.download_button("⬇ Download hold-out predictions", to_csv_bytes(tr.test_df),
                           "holdout_predictions.csv", "text/csv")

# ================================================================ 4. FORECASTING
with tab_fc:
    st.subheader("Forecast future sales")
    c1, c2, c3, c4 = st.columns(4)
    scope_col = c1.selectbox("Forecast for", ["All data", "Category", "Cities", "Brand", "Scenario Name"])
    scope_val = None
    base = full
    if scope_col != "All data":
        scope_val = c2.selectbox(scope_col, sorted(full[scope_col].unique()))
        base = full[full[scope_col] == scope_val]
    else:
        # for the totals, default to the Actual scenario only (Budget would double count)
        if "Scenario Name" in full.columns:
            sc_pick = c2.selectbox("Scenario", ["Actual", "Budget", "Both"])
            if sc_pick != "Both":
                base = full[full["Scenario Name"] == sc_pick]
    freq_label = c3.selectbox("Frequency", list(mc.FREQS), index=0)
    freq = mc.FREQS[freq_label]
    metric = c4.selectbox("Target", ["Sales", "Units Sold", "Profit"])
    default_h = {"MS": 12, "QS": 4, "W-SUN": 26}[freq]
    max_h = {"MS": 36, "QS": 12, "W-SUN": 104}[freq]
    c1, c2 = st.columns(2)
    horizon = c1.slider(f"Horizon ({freq_label.lower()} periods)", 1, max_h, default_h)
    method_pick = c2.selectbox("Method", ["Auto (best on backtest)"] + mc.FORECAST_METHODS)

    y = mc.make_series(base, freq, metric)
    test_n = max(2, min({"MS": 12, "QS": 4, "W-SUN": 26}[freq], len(y) // 4))
    if len(y) < 2 * test_n + 4:
        st.warning("Not enough history for this selection/frequency. Try broader filters or a coarser frequency.")
    else:
        bt, bt_preds = mc.backtest(y, freq, test_n)
        best = bt.iloc[0]["Method"]
        method = best if method_pick.startswith("Auto") else method_pick
        rmse = float(bt.set_index("Method").loc[method, "RMSE"]) if method in set(bt["Method"]) else None
        fc = mc.forecast(y, freq, horizon, method, rmse)

        k1, k2, k3 = st.columns(3)
        k1.metric("Method used", method)
        k2.metric(f"Forecast total ({horizon} periods)", money(fc["Forecast"].sum()))
        last_n = y.iloc[-horizon:].sum()
        k3.metric("vs. last equal-length period", f"{(fc['Forecast'].sum() / last_n - 1) * 100:+.1f}%")

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=y.index, y=y.values, name="History", mode="lines"))
        if method in bt_preds:
            fig.add_trace(go.Scatter(x=bt_preds[method].index, y=bt_preds[method].values,
                                     name="Backtest prediction", mode="lines", line=dict(dash="dot")))
        if "Upper" in fc:
            fig.add_trace(go.Scatter(x=list(fc.index) + list(fc.index[::-1]),
                                     y=list(fc["Upper"]) + list(fc["Lower"][::-1]),
                                     fill="toself", fillcolor="rgba(255,127,14,0.18)",
                                     line=dict(width=0), name="~95% interval", hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=fc.index, y=fc["Forecast"], name="Forecast", mode="lines+markers",
                                 line=dict(color="#ff7f0e")))
        fig.update_layout(title=f"{metric} forecast – {scope_val or 'all data'} ({freq_label})",
                          yaxis_title=metric, hovermode="x unified")
        st.plotly_chart(fig)

        st.markdown(f"**Backtest** – each method trained without the last {test_n} periods, "
                    f"then scored on them:")
        st.dataframe(bt.style.format({"MAE": "{:,.0f}", "RMSE": "{:,.0f}", "MAPE %": "{:.2f}"}))
        out = fc.reset_index().rename(columns={"index": "Period"})
        st.dataframe(out.style.format({c: "{:,.0f}" for c in fc.columns}))
        st.download_button("⬇ Download forecast (CSV)", to_csv_bytes(out), "sales_forecast.csv", "text/csv")
        st.caption("Intervals are approximate: ±1.96 × backtest RMSE, widening slightly with horizon.")

# ================================================================ 5. PREDICT SALES
with tab_pred:
    st.subheader("Predict the sales value of a single order")
    if "tr" not in st.session_state:
        st.info("No model trained yet – train one in the **Predictive Analysis** tab, or use the button below "
                "(Random Forest with default settings).")
        if st.button("Quick-train Random Forest"):
            dn, dc = mc.default_feature_lists(full, False)
            with st.spinner("Training…"):
                st.session_state["tr"] = mc.train_regressor(full, "Random Forest", dn, dc)
                st.session_state["tr_name"] = "Random Forest"
            st.rerun()
    else:
        tr = st.session_state["tr"]
        st.caption(f"Using trained model: **{st.session_state['tr_name']}** "
                   f"(hold-out R² {tr.metrics['R²']:.3f}). Features: "
                   f"{', '.join(tr.num_cols + tr.cat_cols)}")
        prod = (full.drop_duplicates("Product Name").set_index("Product Name")[["Category", "Brand", "Warranty"]])
        rep = (full.drop_duplicates("SalesPerson Name").set_index("SalesPerson Name")[["Age", "DOJ", "Designation"]])

        c1, c2, c3 = st.columns(3)
        pname = c1.selectbox("Product", sorted(prod.index))
        city = c2.selectbox("City", sorted(full["Cities"].unique()))
        rname = c3.selectbox("Salesperson", sorted(rep.index))
        c1, c2, c3 = st.columns(3)
        units = c1.number_input("Units", 1, 50, 3)
        dt = c2.date_input("Order date", value=(full["Date"].max() + pd.Timedelta(days=30)).date())
        scen_in = c3.selectbox("Scenario", sorted(full["Scenario Name"].unique())) if "Scenario Name" in full else None
        price_in = None
        if "Sale price" in tr.num_cols:
            price_in = st.number_input("Sale price", 0, value=int(full[full["Product Name"] == pname]["Sale price"].median()))

        d = pd.Timestamp(dt)
        inputs = {
            "Units Sold": units, "Year": d.year, "Month": d.month, "Quarter": d.quarter,
            "DayOfWeek": d.dayofweek, "WeekOfYear": int(d.isocalendar().week),
            "Age": rep.loc[rname, "Age"],
            "Tenure (yrs)": (d - rep.loc[rname, "DOJ"]).days / 365.25,
            "Category": prod.loc[pname, "Category"], "Brand": prod.loc[pname, "Brand"],
            "Warranty": prod.loc[pname, "Warranty"], "Cities": city, "Designation": rep.loc[rname, "Designation"],
            "Product Name": pname, "SalesPerson Name": rname, "Scenario Name": scen_in,
        }
        if price_in is not None:
            inputs["Sale price"] = price_in
        row = mc.row_template(full, tr.num_cols, tr.cat_cols, inputs)
        pred = float(max(tr.pipeline.predict(row)[0], 0))
        ref = full[(full["Product Name"] == pname) & (full["Units Sold"] == units)]["Sales"]

        st.metric("Predicted sales value", f"{pred:,.0f}")
        if len(ref):
            st.caption(f"Historical: same product & units – median {ref.median():,.0f} "
                       f"(range {ref.min():,.0f} – {ref.max():,.0f}, n={len(ref):,}).")

        st.markdown("**What-if: predicted sales vs. units**")
        wi = []
        for u in range(1, 11):
            r2 = row.copy()
            if "Units Sold" in r2:
                r2["Units Sold"] = u
            wi.append({"Units": u, "Predicted Sales": float(max(tr.pipeline.predict(r2)[0], 0))})
        st.plotly_chart(px.line(pd.DataFrame(wi), x="Units", y="Predicted Sales", markers=True))
        if "Year" in tr.num_cols and d.year > int(full["Year"].max()):
            st.caption("ℹ Tree models can't extrapolate beyond the years they have seen; for future *totals* "
                       "use the Forecasting tab, which models trend explicitly.")

        st.divider()
        st.subheader("Batch prediction")
        st.caption("Upload a CSV with the same columns as the model's features to score many rows at once.")
        bf = st.file_uploader("Batch CSV", type=["csv"], key="batch")
        if bf is not None:
            bdf = pd.read_csv(bf)
            miss = [c for c in tr.num_cols + tr.cat_cols if c not in bdf.columns]
            if miss:
                st.error(f"Missing columns: {miss}")
            else:
                bdf["Predicted Sales"] = np.clip(tr.pipeline.predict(bdf[tr.num_cols + tr.cat_cols]), 0, None)
                st.dataframe(bdf.head(100))
                st.download_button("⬇ Download predictions", to_csv_bytes(bdf), "batch_predictions.csv", "text/csv")
