# StatLab — Statistical Analysis Engine

A Streamlit app for statistical analysis. Upload a CSV, Excel or JSON file, or enter a regression function, and get the tests, models and charts with an automatic plain-language interpretation.

## Who it is for

- Students and researchers who need standard statistical tests without writing code.
- Analysts in finance and marketing who want quick, interpretable models on their own data.

## Why I built it

I wanted one tool that covers the statistics I use in finance and data work: regression, hypothesis tests, time series, panel data and A/B testing, all in one place and explained in plain language.

## What it covers

39 tabs, including:

- Descriptive statistics, normality tests and correlation (Pearson, Spearman, Kendall)
- OLS regression with diagnostics and a prediction tool
- Binary, multinomial and ordinal logistic regression
- One-way, two-way, Welch and repeated-measures ANOVA with post-hoc tests
- Non-parametric and classic hypothesis tests, including chi-square and Fisher
- Time series: stationarity tests, decomposition, ACF/PACF, Holt-Winters, ARIMA and auto-ARIMA
- Panel data: pooled OLS, fixed and random effects, Hausman test
- Clustering (K-Means, hierarchical) and PCA
- Count models (Poisson, negative binomial)
- Power analysis, Cronbach's alpha and multiple-comparison corrections
- A/B test design and analysis, including a Bayesian view

Optionally, a Claude API key adds a deeper natural-language interpretation of the results. Without it, the built-in interpretation is used.

## Run

On macOS:

```bash
./run.sh
```

The script creates a virtual environment if needed and opens the app at http://localhost:8501.

Manually:

```bash
pip install -r requirements.txt
streamlit run app.py
```

A packaged macOS app can be built with `build_app.sh` (see `packaging/`).

## Tech

Python, Streamlit, pandas, NumPy, SciPy, statsmodels, linearmodels, lifelines, pingouin, scikit-learn, Plotly, Matplotlib, Anthropic API (optional).

The original Turkish documentation is in [README.tr.md](README.tr.md).
