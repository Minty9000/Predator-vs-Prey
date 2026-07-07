# Lotka–Volterra Market Competition Dashboard

Models **Apple vs Samsung** US market share as a predator–prey style **Lotka–Volterra** competition system, with Huawei as a third competitor on actual data. Includes parameter fitting, residual analysis, and an interactive **Streamlit** dashboard.

**Authors:** [Kareena Doda](https://github.com/kareenadoda) · [Minty9000](https://github.com/Minty9000)

## Features

- Fit Lotka–Volterra parameters to normalized Apple & Samsung market share time series
- Residual analysis (time-series and histogram) to evaluate model fit
- Regime detection: Apple dominance, Samsung dominance, or equilibrium phases
- **Streamlit dashboard** with scenario selection, company toggles, smoothing, and model overlay
- Plotly interactive charts for actual vs modeled shares

## Project structure

```
script.ipynb                    # Analysis notebook (fitting + residuals)
dashboard.py                    # Streamlit interactive dashboard
vendor_augmented_large.csv      # Vendor market share data
lotka_volterra_100_2000_lag3.csv
data.csv
```

## Run locally

```bash
pip install pandas numpy scipy matplotlib plotly streamlit jupyter
streamlit run dashboard.py
```

Or open `script.ipynb` in Jupyter for the offline analysis workflow.

## Tech stack

Python · Lotka–Volterra · SciPy · Streamlit · Plotly · Pandas · Jupyter
