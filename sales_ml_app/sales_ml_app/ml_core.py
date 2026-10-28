"""
ml_core.py – data loading, feature engineering, regression and forecasting logic.
No Streamlit / Plotly imports here so it can be unit-tested on its own.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

try:  # optional dependency
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    HAS_STATSMODELS = True
except Exception:  # pragma: no cover
    HAS_STATSMODELS = False

warnings.filterwarnings("ignore")

# ----------------------------------------------------------------------------
# 1. LOADING & CLEANING
# ----------------------------------------------------------------------------
EXPECTED_COLS = ["Date", "Sales", "Units Sold"]


def load_data(file_or_path) -> pd.DataFrame:
    """Read the Excel/CSV file. The sample file has 2 blank rows above the header,
    so we auto-detect the header row (first row containing 'Date' and 'Sales')."""
    name = getattr(file_or_path, "name", str(file_or_path)).lower()
    if name.endswith(".csv"):
        raw = pd.read_csv(file_or_path, header=None)
    else:
        raw = pd.read_excel(file_or_path, header=None)

    header_row = None
    for i in range(min(15, len(raw))):
        vals = raw.iloc[i].astype(str).str.strip().tolist()
        if "Date" in vals and "Sales" in vals:
            header_row = i
            break
    if header_row is None:
        raise ValueError("Could not find a header row containing 'Date' and 'Sales'.")

    df = raw.iloc[header_row + 1:].copy()
    df.columns = [str(c).strip() for c in raw.iloc[header_row]]
    df = df.reset_index(drop=True)
    return clean_data(df)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.dropna(how="all")
    for c in ["Date", "DOJ"]:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")
    num_cols = ["Index", "ProductID", "SalesrepID", "CityID", "ScenarioID", "Units Sold",
                "Sales", "Cost of sales", "Sale price", "Cost price", "Latitude",
                "Longitude", "Age"]
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["Date", "Sales"]).drop_duplicates()

    # Derived business columns
    if "Cost of sales" in df.columns:
        df["Profit"] = df["Sales"] - df["Cost of sales"]
        df["Margin %"] = np.where(df["Sales"] > 0, df["Profit"] / df["Sales"] * 100, np.nan)
    df["Year"] = df["Date"].dt.year
    df["Quarter"] = df["Date"].dt.quarter
    df["Month"] = df["Date"].dt.month
    df["MonthName"] = df["Date"].dt.strftime("%b")
    df["DayOfWeek"] = df["Date"].dt.dayofweek
    df["WeekOfYear"] = df["Date"].dt.isocalendar().week.astype(int)
    df["YearMonth"] = df["Date"].dt.to_period("M").dt.to_timestamp()
    if "DOJ" in df.columns:
        df["Tenure (yrs)"] = ((df["Date"] - df["DOJ"]).dt.days / 365.25).round(2)
    return df.reset_index(drop=True)


# ----------------------------------------------------------------------------
# 2. TRANSACTION-LEVEL SALES PREDICTION (regression)
# ----------------------------------------------------------------------------
# Columns that are (near-)deterministic functions of the target: Sales = Units × Sale price.
LEAKY = ["Sale price", "Cost of sales", "Cost price", "Profit", "Margin %"]

CANDIDATE_NUMERIC = ["Units Sold", "Age", "Tenure (yrs)", "Year", "Month", "Quarter",
                     "DayOfWeek", "WeekOfYear", "ProductID", "SalesrepID", "CityID",
                     "Latitude", "Longitude"]
CANDIDATE_CATEGORICAL = ["Category", "Brand", "Warranty", "Cities", "Designation",
                         "Scenario Name", "Product Name", "SalesPerson Name"]


def default_feature_lists(df: pd.DataFrame, include_price: bool = False):
    num = [c for c in CANDIDATE_NUMERIC if c in df.columns]
    cat = [c for c in CANDIDATE_CATEGORICAL if c in df.columns]
    # IDs duplicate their name columns; keep names (categorical) only
    for dup in ["ProductID", "SalesrepID", "CityID", "Latitude", "Longitude"]:
        if dup in num:
            num.remove(dup)
    for dup in ["SalesPerson Name"]:
        if dup in cat:
            cat.remove(dup)
    if include_price and "Sale price" in df.columns:
        num.append("Sale price")
    return num, cat


MODEL_ZOO = {
    "Ridge Regression": lambda: Ridge(alpha=1.0),
    "Random Forest": lambda: RandomForestRegressor(
        n_estimators=150, max_depth=14, min_samples_leaf=3, n_jobs=-1, random_state=42),
    "Gradient Boosting": lambda: GradientBoostingRegressor(
        n_estimators=250, learning_rate=0.08, max_depth=4, subsample=0.8, random_state=42),
}


def build_pipeline(model_name: str, num_cols, cat_cols) -> Pipeline:
    scale = model_name == "Ridge Regression"
    pre = ColumnTransformer([
        ("num", StandardScaler() if scale else "passthrough", num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
    ])
    return Pipeline([("pre", pre), ("model", MODEL_ZOO[model_name]())])


def regression_metrics(y_true, y_pred) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mape = np.mean(np.abs((y_true - y_pred) / np.where(y_true == 0, np.nan, y_true))) * 100
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R²": r2_score(y_true, y_pred),
        "MAPE %": float(mape),
    }


@dataclass
class TrainResult:
    pipeline: Pipeline
    metrics: dict
    test_df: pd.DataFrame        # actual vs predicted on the hold-out
    num_cols: list
    cat_cols: list
    importance: pd.DataFrame


def train_regressor(df, model_name, num_cols, cat_cols, test_size=0.2,
                    time_split=True, max_rows=60000, random_state=42) -> TrainResult:
    d = df.sort_values("Date").reset_index(drop=True)
    if len(d) > max_rows and model_name != "Ridge Regression":
        # subsample (keeps time order) so tree models stay quick on the free tier
        d = d.sample(max_rows, random_state=random_state).sort_values("Date").reset_index(drop=True)
    X, y = d[num_cols + cat_cols], d["Sales"]
    if time_split:
        cut = int(len(d) * (1 - test_size))
        Xtr, Xte, ytr, yte = X.iloc[:cut], X.iloc[cut:], y.iloc[:cut], y.iloc[cut:]
    else:
        from sklearn.model_selection import train_test_split
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=test_size, random_state=random_state)

    pipe = build_pipeline(model_name, num_cols, cat_cols)
    pipe.fit(Xtr, ytr)
    pred = np.clip(pipe.predict(Xte), 0, None)
    metrics = regression_metrics(yte, pred)

    test_df = Xte.copy()
    test_df["Actual"], test_df["Predicted"] = yte.values, pred
    test_df["Date"] = d.loc[Xte.index, "Date"].values

    # feature importance, aggregated back to original columns
    names = pipe.named_steps["pre"].get_feature_names_out()
    model = pipe.named_steps["model"]
    raw = getattr(model, "feature_importances_", None)
    if raw is None:
        raw = np.abs(getattr(model, "coef_", np.zeros(len(names))))
    imp = pd.DataFrame({"feature": names, "importance": raw})
    imp["feature"] = imp["feature"].str.replace(r"^(num|cat)__", "", regex=True)
    for c in cat_cols:  # collapse one-hot columns
        imp.loc[imp["feature"].str.startswith(c + "_"), "feature"] = c
    imp = imp.groupby("feature", as_index=False)["importance"].sum()
    imp["importance"] = imp["importance"] / imp["importance"].sum()
    imp = imp.sort_values("importance", ascending=False).reset_index(drop=True)
    return TrainResult(pipe, metrics, test_df, num_cols, cat_cols, imp)


def compare_models(df, num_cols, cat_cols, test_size=0.2, time_split=True, max_rows=40000):
    rows = []
    for name in MODEL_ZOO:
        r = train_regressor(df, name, num_cols, cat_cols, test_size, time_split, max_rows)
        rows.append({"Model": name, **r.metrics})
    return pd.DataFrame(rows).sort_values("RMSE").reset_index(drop=True)


# ----------------------------------------------------------------------------
# 3. TIME-SERIES FORECASTING
# ----------------------------------------------------------------------------
FREQS = {"Monthly": "MS", "Weekly": "W-SUN", "Quarterly": "QS"}
SEASON = {"MS": 12, "W-SUN": 52, "QS": 4}
FORECAST_METHODS = ["Trend + Seasonality (Ridge)", "Gradient Boosting (lags)", "Seasonal Naive"]
if HAS_STATSMODELS:
    FORECAST_METHODS.insert(0, "Holt-Winters")


def make_series(df: pd.DataFrame, freq: str, value: str = "Sales") -> pd.Series:
    """Aggregate to a regular, gap-free time series."""
    s = df.set_index("Date")[value].resample(freq).sum()
    # drop incomplete first/last bucket (partial periods bias the series)
    first, last = df["Date"].min(), df["Date"].max()
    if freq == "MS":
        if first.day != 1:
            s = s.iloc[1:]
        if (last + pd.Timedelta(days=1)).day != 1:
            s = s.iloc[:-1]
    elif freq == "W-SUN":
        s = s.iloc[1:-1] if len(s) > 2 else s
    elif freq == "QS":
        if not (first.month in (1, 4, 7, 10) and first.day == 1):
            s = s.iloc[1:]
        if not ((last + pd.Timedelta(days=1)).day == 1 and (last + pd.Timedelta(days=1)).month in (1, 4, 7, 10)):
            s = s.iloc[:-1]
    return s.astype(float)


def _future_index(y: pd.Series, freq: str, h: int) -> pd.DatetimeIndex:
    return pd.date_range(y.index[-1], periods=h + 1, freq=freq)[1:]


def _fourier(idx: pd.DatetimeIndex, freq: str, K: int) -> np.ndarray:
    if freq == "MS":
        phase = (idx.month - 1) / 12
    elif freq == "QS":
        phase = (idx.quarter - 1) / 4
    else:
        phase = (idx.dayofyear - 1) / 365.25
    cols = []
    for k in range(1, K + 1):
        cols += [np.sin(2 * np.pi * k * phase), np.cos(2 * np.pi * k * phase)]
    return np.column_stack(cols)


def _trend_seasonal_design(idx, t0, freq):
    t = np.arange(len(idx)) + t0
    K = {"MS": 4, "QS": 1, "W-SUN": 4}[freq]
    return np.column_stack([t, _fourier(idx, freq, K)])


def _fit_predict(y_train: pd.Series, h: int, freq: str, method: str) -> np.ndarray:
    m = SEASON[freq]
    if method == "Seasonal Naive":
        # last season repeated + average growth between the last two seasons
        if len(y_train) < m:
            return np.repeat(y_train.iloc[-1], h)
        last = y_train.iloc[-m:].values
        growth = 1.0
        if len(y_train) >= 2 * m:
            prev = y_train.iloc[-2 * m:-m].sum()
            growth = last.sum() / prev if prev > 0 else 1.0
        return np.array([last[i % m] * growth for i in range(h)])

    if method == "Holt-Winters":
        if len(y_train) >= 2 * m and freq != "W-SUN":
            fit = ExponentialSmoothing(y_train, trend="add", damped_trend=True,
                                       seasonal="add", seasonal_periods=m,
                                       initialization_method="estimated").fit()
        else:
            fit = ExponentialSmoothing(y_train, trend="add", damped_trend=True,
                                       initialization_method="estimated").fit()
        return np.asarray(fit.forecast(h))

    if method == "Trend + Seasonality (Ridge)":
        Xtr = _trend_seasonal_design(y_train.index, 0, freq)
        mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
        model = Ridge(alpha=0.5).fit((Xtr - mu) / sd, y_train.values)
        fidx = _future_index(y_train, freq, h)
        Xf = _trend_seasonal_design(fidx, len(y_train), freq)
        return model.predict((Xf - mu) / sd)

    if method == "Gradient Boosting (lags)":
        return _gbm_forecast(y_train, h, freq)
    raise ValueError(method)


def _gbm_forecast(y: pd.Series, h: int, freq: str) -> np.ndarray:
    """Hybrid: linear trend (extrapolates) + GradientBoosting on residual lags/season."""
    lags = {"MS": [1, 2, 3, 6, 12], "QS": [1, 2, 4], "W-SUN": [1, 2, 4, 13, 52]}[freq]
    max_lag = max(lags)
    n = len(y)
    if n <= max_lag + 8:
        lags = [l for l in lags if l <= max(1, n // 3)] or [1]
        max_lag = max(lags)

    t = np.arange(n)
    slope, intercept = np.polyfit(t, y.values, 1)
    trend = lambda tt: intercept + slope * tt
    resid = pd.Series(y.values - trend(t), index=y.index)

    def feats(res_hist: list, idx: pd.Timestamp, tt: int):
        row = [res_hist[-l] for l in lags]
        row += [np.mean(res_hist[-3:]), tt]
        row += list(_fourier(pd.DatetimeIndex([idx]), freq, 2)[0])
        return row

    X, tgt = [], []
    hist = resid.tolist()
    for i in range(max_lag, n):
        X.append(feats(hist[:i], y.index[i], i))
        tgt.append(hist[i])
    model = GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, max_depth=2,
                                      subsample=0.8, random_state=42).fit(np.array(X), np.array(tgt))
    fidx = _future_index(y, freq, h)
    out = []
    for j in range(h):
        r = model.predict(np.array([feats(hist, fidx[j], n + j)]))[0]
        hist.append(r)
        out.append(trend(n + j) + r)
    return np.array(out)


def backtest(y: pd.Series, freq: str, test_periods: int, methods=None) -> tuple[pd.DataFrame, dict]:
    methods = methods or FORECAST_METHODS
    train, test = y.iloc[:-test_periods], y.iloc[-test_periods:]
    rows, preds = [], {}
    for m in methods:
        try:
            p = np.clip(_fit_predict(train, test_periods, freq, m), 0, None)
        except Exception:
            continue
        preds[m] = pd.Series(p, index=test.index)
        mt = regression_metrics(test.values, p)
        rows.append({"Method": m, "MAE": mt["MAE"], "RMSE": mt["RMSE"], "MAPE %": mt["MAPE %"]})
    return pd.DataFrame(rows).sort_values("RMSE").reset_index(drop=True), preds


def forecast(y: pd.Series, freq: str, horizon: int, method: str, rmse: float | None = None) -> pd.DataFrame:
    p = np.clip(_fit_predict(y, horizon, freq, method), 0, None)
    idx = _future_index(y, freq, horizon)
    out = pd.DataFrame({"Forecast": p}, index=idx)
    if rmse is not None:
        w = 1.96 * rmse * np.sqrt(1 + 0.08 * np.arange(horizon))   # interval widens with horizon
        out["Lower"] = np.clip(p - w, 0, None)
        out["Upper"] = p + w
    return out


# ----------------------------------------------------------------------------
# 4. SCENARIO PREDICTION HELPERS
# ----------------------------------------------------------------------------
def row_template(df: pd.DataFrame, num_cols, cat_cols, inputs: dict) -> pd.DataFrame:
    """Build a single-row frame for the trained pipeline; unspecified columns get median/mode."""
    row = {}
    for c in num_cols:
        row[c] = inputs.get(c, float(df[c].median()))
    for c in cat_cols:
        row[c] = inputs.get(c, df[c].mode().iloc[0])
    return pd.DataFrame([row])
