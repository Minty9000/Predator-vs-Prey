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
    "Apple vs Samsung are modeled using a Lotka–Volterra predator–prey system. "
    "Huawei is included as a third competitor (actual data only)."
)

# -----------------------------------
# Load data
# -----------------------------------
@st.cache_data
def load_data():
    # 🔁 Replace this with how you actually load df
    # Example: CSV with columns: Date, scenario_id, Apple_US, Samsung_US, Huawei_US
    url = "https://drive.google.com/uc?id=1h6hO-0fWiIDKShToIZIXnJMVVVjLGIw5"
    df = pd.read_csv(url, parse_dates=["Date"])
    return df

df = load_data()

# Basic sanity check
required_cols = {"Date", "scenario_id", "Apple_US", "Samsung_US", "Huawei_US"}
missing = required_cols - set(df.columns)
if missing:
    st.error(f"Missing columns in data: {missing}")
    st.stop()

# -----------------------------------
# Sidebar controls
# -----------------------------------
st.sidebar.header("Controls")

# Scenario selector
scenario_ids = sorted(df["scenario_id"].unique())
selected_scenario = st.sidebar.selectbox("Scenario ID", scenario_ids)

# Filter data by scenario
use = df[df["scenario_id"] == selected_scenario].copy()
use = use.sort_values("Date")

if use.empty:
    st.error("No data for the selected scenario.")
    st.stop()

# Company visibility selector
company_options = ["Apple", "Samsung", "Huawei"]
selected_companies = st.sidebar.multiselect(
    "Companies to display",
    company_options,
    default=company_options
)

# Option to show/hide model lines
show_model = st.sidebar.checkbox(
    "Show Lotka–Volterra model for Apple & Samsung",
    value=True
)

# -----------------------------------
# Extract and normalize series
# -----------------------------------
A = use["Apple_US"].values.astype(float)
S = use["Samsung_US"].values.astype(float)
H = use["Huawei_US"].values.astype(float)
dates = use["Date"]

# Avoid division by zero if any series is constant or zero
A_n = A / np.max(A) if np.max(A) != 0 else A
S_n = S / np.max(S) if np.max(S) != 0 else S
H_n = H / np.max(H) if np.max(H) != 0 else H

t = np.arange(len(use))  # time index for ODE solver & fitting

# -----------------------------------
# Lotka–Volterra model (2-species)
# -----------------------------------
def lv(y, t, a, b, c, d):
    """
    Lotka–Volterra system for Apple (A) and Samsung (S).
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

# Initial conditions from normalized series
A_init = A_n[0]
S_init = S_n[0]

# -----------------------------------
# Fit LV model to Apple & Samsung
# -----------------------------------
with st.spinner("Fitting Lotka–Volterra model for Apple & Samsung..."):
    ydata = np.concatenate([A_n, S_n])

    try:
        params, _ = curve_fit(
            lambda t, a, b, c, d: np.concatenate(
                simulate(t, a, b, c, d, A_init, S_init)
            ),
            t,
            ydata,
            p0=[0.05, 0.01, 0.05, 0.01],  # initial guesses
            maxfev=20000
        )

        A_pred, S_pred = simulate(t, *params, A_init, S_init)
        fit_success = True
    except Exception as e:
        st.warning(f"Model fit failed: {e}")
        A_pred = np.full_like(A_n, np.nan)
        S_pred = np.full_like(S_n, np.nan)
        fit_success = False

# -----------------------------------
# Display fitted parameters
# -----------------------------------
st.subheader("Fitted Lotka–Volterra Parameters (Apple & Samsung)")
if fit_success:
    a, b, c, d = params
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("a (Apple growth)", f"{a:.4f}")
    col2.metric("b (Apple loss to Samsung)", f"{b:.4f}")
    col3.metric("c (Samsung decay)", f"{c:.4f}")
    col4.metric("d (Samsung gain from Apple)", f"{d:.4f}")
else:
    st.write("Model parameters unavailable due to fit failure.")

# -----------------------------------
# Build interactive Plotly figure
# -----------------------------------
fig = go.Figure()

# Apple traces
if "Apple" in selected_companies:
    fig.add_trace(go.Scatter(
        x=dates,
        y=A_n,
        mode="lines",
        name="Apple (actual)"
    ))
    if show_model and fit_success:
        fig.add_trace(go.Scatter(
            x=dates,
            y=A_pred,
            mode="lines",
            name="Apple (model)",
            line=dict(dash="dash")
        ))

# Samsung traces
if "Samsung" in selected_companies:
    fig.add_trace(go.Scatter(
        x=dates,
        y=S_n,
        mode="lines",
        name="Samsung (actual)"
    ))
    if show_model and fit_success:
        fig.add_trace(go.Scatter(
            x=dates,
            y=S_pred,
            mode="lines",
            name="Samsung (model)",
            line=dict(dash="dash")
        ))

# Huawei trace (actual only)
if "Huawei" in selected_companies:
    fig.add_trace(go.Scatter(
        x=dates,
        y=H_n,
        mode="lines",
        name="Huawei (actual)"
    ))

fig.update_layout(
    title=f"Lotka–Volterra Fit — Apple vs Samsung (US, Scenario {selected_scenario})"
           "<br><sup>Huawei shown as third competitor (actual only)</sup>",
    xaxis_title="Date",
    yaxis_title="Normalized share",
    hovermode="x unified",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0)
)

st.plotly_chart(fig, use_container_width=True)

# -----------------------------------
# Data preview
# -----------------------------------
with st.expander("🔍 Show raw data for this scenario"):
    st.dataframe(use.reset_index(drop=True))