import numpy as np
import pandas as pd
from scipy.integrate import odeint
from scipy.optimize import curve_fit
import plotly.graph_objects as go
import streamlit as st

# -----------------------------------
# Streamlit page config
# -----------------------------------
st.set_page_config(
    page_title="Market Share LV Dashboard",
    layout="wide"
)

st.title("📈 Lotka–Volterra Market Competition Dashboard")
st.write(
    "Apple vs Samsung are modeled using a Lotka–Volterra competition system "
    "on their *joint* market (Apple + Samsung). "
    "Huawei is included as a third competitor (actual data only)."
)

# -----------------------------------
# Load data
# -----------------------------------
@st.cache_data
def load_data():
    # Replace with your own data source if needed
    url = "https://drive.google.com/uc?id=1h6hO-0fWiIDKShToIZIXnJMVVVjLGIw5"
    df = pd.read_csv(url, parse_dates=["Date"])
    return df

df = load_data()

required_cols = {"Date", "scenario_id", "Apple_US", "Samsung_US", "Huawei_US"}
missing = required_cols - set(df.columns)
if missing:
    st.error(f"Missing columns in data: {missing}")
    st.stop()

# -----------------------------------
# Sidebar controls
# -----------------------------------
st.sidebar.header("Controls")

scenario_ids = sorted(df["scenario_id"].unique())
selected_scenario = st.sidebar.selectbox("Scenario ID", scenario_ids)

use = df[df["scenario_id"] == selected_scenario].copy()
use = use.sort_values("Date")

if use.empty:
    st.error("No data for the selected scenario.")
    st.stop()

company_options = ["Apple", "Samsung", "Huawei"]
selected_companies = st.sidebar.multiselect(
    "Companies to display",
    company_options,
    default=company_options
)

show_model = st.sidebar.checkbox(
    "Show Lotka–Volterra model for Apple & Samsung",
    value=True
)

smooth_window = st.sidebar.slider(
    "Smoothing window (days) for Apple & Samsung",
    min_value=3, max_value=60, value=7, step=2
)

# -----------------------------------
# Extract raw series
# -----------------------------------
A_raw = use["Apple_US"].values.astype(float)
S_raw = use["Samsung_US"].values.astype(float)
H_raw = use["Huawei_US"].values.astype(float)
dates = use["Date"]

t = np.arange(len(use))  # time index for ODE solver & fitting

# -----------------------------------
# Build Apple/Samsung *share-of-two* + smoothing
# -----------------------------------
total_AS = A_raw + S_raw
# avoid division by zero
A_share = np.divide(A_raw, total_AS, out=np.zeros_like(A_raw), where=total_AS != 0)
S_share = np.divide(S_raw, total_AS, out=np.zeros_like(S_raw), where=total_AS != 0)

A_smooth = (
    pd.Series(A_share)
    .rolling(window=smooth_window, center=True, min_periods=1)
    .mean()
    .bfill()
    .ffill()
    .values
)
S_smooth = (
    pd.Series(S_share)
    .rolling(window=smooth_window, center=True, min_periods=1)
    .mean()
    .bfill()
    .ffill()
    .values
)

# These are the series the LV model will try to match
A_obs = A_smooth
S_obs = S_smooth

# Huawei: just normalized for plotting (not in LV model)
H_n = H_raw / np.max(H_raw) if np.max(H_raw) != 0 else H_raw

# -----------------------------------
# Lotka–Volterra model (2-species)
# -----------------------------------
def lv(y, t, a, b, c, d):
    """
    Lotka–Volterra competition model for Apple (A) and Samsung (S).
    dA/dt = a*A - b*A*S
    dS/dt = -c*S + d*A*S
    """
    A, S = y
    dA = a * A - b * A * S
    dS = -c * S + d * A * S
    return [dA, dS]

def simulate(t, a, b, c, d, A0, S0):
    y0 = [A0, S0]
    sol = odeint(lv, y0, t, args=(a, b, c, d))
    return sol[:, 0], sol[:, 1]

# Initial conditions from smoothed series
A_init = A_obs[0]
S_init = S_obs[0]

def lv_concat(t, a, b, c, d):
    """
    Helper for curve_fit: concatenated A and S predictions.
    """
    A_pred, S_pred = simulate(t, a, b, c, d, A_init, S_init)
    return np.concatenate([A_pred, S_pred])

# -----------------------------------
# Fit LV model to smoothed Apple & Samsung shares
# -----------------------------------
with st.spinner("Fitting Lotka–Volterra model for Apple & Samsung..."):
    ydata = np.concatenate([A_obs, S_obs])

    try:
        p0 = [0.01, 0.01, 0.01, 0.01]  # initial guesses
        params, _ = curve_fit(
            lv_concat,
            t,
            ydata,
            p0=p0,
            bounds=(0, np.inf),        # non-negative parameters
            maxfev=50000
        )

        A_pred, S_pred = simulate(t, *params, A_init, S_init)
        fit_success = True
    except Exception as e:
        st.warning(f"Model fit failed: {e}")
        A_pred = np.full_like(A_obs, np.nan)
        S_pred = np.full_like(S_obs, np.nan)
        fit_success = False

# -----------------------------------
# Display fitted parameters + simple error metrics
# -----------------------------------
st.subheader("Fitted Lotka–Volterra Parameters (Apple & Samsung)")

if fit_success:
    a, b, c, d = params

    # basic RMSE for info
    rmse_A = np.sqrt(np.mean((A_obs - A_pred) ** 2))
    rmse_S = np.sqrt(np.mean((S_obs - S_pred) ** 2))

    col1, col2, col3, col4, col5, col6 = st.columns(6)
    col1.metric("a (Apple growth)", f"{a:.4f}")
    col2.metric("b (Apple loss to Samsung)", f"{b:.4f}")
    col3.metric("c (Samsung decay)", f"{c:.4f}")
    col4.metric("d (Samsung gain from Apple)", f"{d:.4f}")
    col5.metric("Apple RMSE", f"{rmse_A:.4f}")
    col6.metric("Samsung RMSE", f"{rmse_S:.4f}")
else:
    st.write("Model parameters unavailable due to fit failure.")

# -----------------------------------
# Build interactive Plotly figure
# -----------------------------------
fig = go.Figure()

# Apple traces (smoothed share-of-two + model)
if "Apple" in selected_companies:
    fig.add_trace(go.Scatter(
        x=dates,
        y=A_obs,
        mode="lines",
        name=f"Apple (actual, {smooth_window}d avg share of A+S)"
    ))
    if show_model and fit_success:
        fig.add_trace(go.Scatter(
            x=dates,
            y=A_pred,
            mode="lines",
            name="Apple (LV model)",
            line=dict(dash="dash")
        ))

# Samsung traces (smoothed share-of-two + model)
if "Samsung" in selected_companies:
    fig.add_trace(go.Scatter(
        x=dates,
        y=S_obs,
        mode="lines",
        name=f"Samsung (actual, {smooth_window}d avg share of A+S)"
    ))
    if show_model and fit_success:
        fig.add_trace(go.Scatter(
            x=dates,
            y=S_pred,
            mode="lines",
            name="Samsung (LV model)",
            line=dict(dash="dash")
        ))

# Huawei (normalized only, not part of LV system)
if "Huawei" in selected_companies:
    fig.add_trace(go.Scatter(
        x=dates,
        y=H_n,
        mode="lines",
        name="Huawei (actual, normalized)"
    ))

fig.update_layout(
    title=(
        f"Lotka–Volterra Fit — Apple vs Samsung (Scenario {selected_scenario})"
        "<br><sup>Apple & Samsung: smoothed share of Apple+Samsung; "
        "Huawei: normalized volume (not in LV model)</sup>"
    ),
    xaxis_title="Date",
    yaxis_title="Scaled share (0–1)",
    hovermode="x unified",
    legend=dict(orientation="h", yanchor="bottom", y=1.02,
                xanchor="left", x=0)
)

st.plotly_chart(fig, use_container_width=True)

# -----------------------------------
# Data preview
# -----------------------------------
with st.expander("🔍 Show raw data for this scenario"):
    st.dataframe(use.reset_index(drop=True))