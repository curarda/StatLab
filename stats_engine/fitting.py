"""Veriye eğri/model uydurma: polinom, üstel, logaritmik, güç yasası."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy import stats

from . import interpret as I


def _r2(y, yhat):
    y = np.asarray(y, float); yhat = np.asarray(yhat, float)
    ss_res = np.sum((y - yhat) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    return float(1 - ss_res / ss_tot) if ss_tot else np.nan


def fit_curve(x, y, model: str = "poly", degree: int = 2) -> dict:
    x = np.asarray(pd.to_numeric(pd.Series(x), errors="coerce"), float)
    y = np.asarray(pd.to_numeric(pd.Series(y), errors="coerce"), float)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if len(x) < degree + 2:
        return {"error": "Uydurma için yeterli veri yok."}
    order = np.argsort(x)
    x_s = x[order]
    try:
        if model == "poly":
            coef = np.polyfit(x, y, degree)
            p = np.poly1d(coef)
            yhat = p(x)
            eq = _poly_str(coef)
            func = p
            params = {f"a{degree-i}": float(c) for i, c in enumerate(coef)}
        elif model == "exp":  # y = a*exp(b*x)+c
            p0 = [1.0, 0.01, float(np.min(y))]
            popt, _ = curve_fit(lambda t, a, b, c: a * np.exp(b * t) + c, x, y, p0=p0, maxfev=10000)
            func = lambda t, a=popt[0], b=popt[1], c=popt[2]: a * np.exp(b * t) + c
            yhat = func(x)
            eq = f"y = {popt[0]:.4g}·e^({popt[1]:.4g}·x) + {popt[2]:.4g}"
            params = {"a": float(popt[0]), "b": float(popt[1]), "c": float(popt[2])}
        elif model == "log":  # y = a*ln(x)+b
            if (x <= 0).any():
                return {"error": "Logaritmik model için x > 0 olmalı."}
            popt, _ = curve_fit(lambda t, a, b: a * np.log(t) + b, x, y, maxfev=10000)
            func = lambda t, a=popt[0], b=popt[1]: a * np.log(t) + b
            yhat = func(x)
            eq = f"y = {popt[0]:.4g}·ln(x) + {popt[1]:.4g}"
            params = {"a": float(popt[0]), "b": float(popt[1])}
        elif model == "power":  # y = a*x^b
            if (x <= 0).any() or (y <= 0).any():
                return {"error": "Güç yasası için x > 0 ve y > 0 olmalı."}
            popt, _ = curve_fit(lambda t, a, b: a * np.power(t, b), x, y, p0=[1, 1], maxfev=10000)
            func = lambda t, a=popt[0], b=popt[1]: a * np.power(t, b)
            yhat = func(x)
            eq = f"y = {popt[0]:.4g}·x^{popt[1]:.4g}"
            params = {"a": float(popt[0]), "b": float(popt[1])}
        else:
            return {"error": f"Bilinmeyen model: {model}"}
    except Exception as e:
        return {"error": f"Uydurma başarısız: {e}"}

    r2 = _r2(y, yhat)
    n, k = len(x), len(params)
    adj_r2 = 1 - (1 - r2) * (n - 1) / (n - k - 1) if n - k - 1 > 0 else np.nan
    rmse = float(np.sqrt(np.mean((y - yhat) ** 2)))
    x_line = np.linspace(x.min(), x.max(), 200)
    y_line = func(x_line)
    return {"model": model, "degree": degree, "params": params, "equation": eq,
            "r2": r2, "adj_r2": float(adj_r2), "rmse": rmse, "n": n,
            "x": x, "y": y, "x_line": x_line, "y_line": np.asarray(y_line), "func": func}


def _poly_str(coef) -> str:
    d = len(coef) - 1
    terms = []
    for i, c in enumerate(coef):
        power = d - i
        if power == 0:
            terms.append(f"{c:.4g}")
        elif power == 1:
            terms.append(f"{c:.4g}·x")
        else:
            terms.append(f"{c:.4g}·x^{power}")
    return "y = " + " + ".join(terms).replace("+ -", "- ")


def compare_fits(x, y, models: list[tuple[str, int]]) -> pd.DataFrame:
    rows = []
    for model, deg in models:
        r = fit_curve(x, y, model, deg)
        if "error" in r:
            continue
        label = f"Polinom (derece {deg})" if model == "poly" else \
            {"exp": "Üstel", "log": "Logaritmik", "power": "Güç yasası"}[model]
        rows.append({"Model": label, "Denklem": r["equation"], "R²": r["r2"],
                     "Düz. R²": r["adj_r2"], "RMSE": r["rmse"]})
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values("R²", ascending=False).reset_index(drop=True)
    return out


def interpret_fit(res: dict) -> str:
    if "error" in res:
        return res["error"]
    lines = [f"### Eğri uydurma sonucu", I.bullet(f"**Denklem:** {res['equation']}")]
    lines.append(I.bullet(f"R² = {res['r2']:.4f} → veri değişkenliğinin %{res['r2']*100:.1f}'i model tarafından açıklanıyor "
                          f"({I.r2_quality(res['r2'])})."))
    lines.append(I.bullet(f"Düzeltilmiş R² = {res['adj_r2']:.4f}, RMSE = {res['rmse']:.4f}, n = {res['n']}."))
    lines.append(I.bullet("Parametreler: " + ", ".join(f"{k} = {v:.4g}" for k, v in res["params"].items())))
    if res["r2"] > 0.9:
        lines.append(I.bullet("Model veriye **çok iyi** uyuyor. Yine de aşırı uyum (overfitting) için düzeltilmiş R²'ye bakın."))
    elif res["r2"] < 0.4:
        lines.append(I.bullet("Uyum **zayıf**; farklı bir model türü (üstel/logaritmik/daha yüksek derece) deneyin."))
    return I.joinlines(lines)
