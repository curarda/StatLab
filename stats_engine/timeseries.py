"""Zaman serisi analizi: trend, mevsimsel ayrıştırma, durağanlık, ACF/PACF, tahmin."""
from __future__ import annotations

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller, kpss, acf, pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA

from . import interpret as I


def build_series(df: pd.DataFrame, value: str, time: str | None = None) -> pd.Series:
    """Sayısal değer serisini (isteğe bağlı tarih indeksiyle) oluştur."""
    s = pd.to_numeric(df[value], errors="coerce")
    if time and time in df.columns:
        idx = pd.to_datetime(df[time], errors="coerce", dayfirst=True)
        s = pd.Series(s.values, index=idx, name=value).dropna()
        s = s.sort_index()
    else:
        s = pd.Series(s.values, name=value).dropna().reset_index(drop=True)
    return s


def overview(s: pd.Series) -> dict:
    n = len(s)
    x = np.arange(n)
    slope, intercept = np.polyfit(x, s.values, 1) if n > 1 else (np.nan, np.nan)
    return {"n": n, "start": str(s.index[0]) if n else None, "end": str(s.index[-1]) if n else None,
            "mean": float(s.mean()), "slope": float(slope), "intercept": float(intercept),
            "trend_line": intercept + slope * x}


def stationarity(s: pd.Series, alpha: float = 0.05) -> dict:
    out = {}
    try:
        adf = adfuller(s.values, autolag="AIC")
        out["adf_stat"] = float(adf[0]); out["adf_p"] = float(adf[1])
        out["adf_stationary"] = bool(adf[1] < alpha)
    except Exception as e:
        out["adf_error"] = str(e)
    try:
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            k = kpss(s.values, regression="c", nlags="auto")
        out["kpss_stat"] = float(k[0]); out["kpss_p"] = float(k[1])
        out["kpss_stationary"] = bool(k[1] >= alpha)  # KPSS H0 = durağan
    except Exception:
        pass
    return out


def decompose(s: pd.Series, period: int, model: str = "additive") -> dict:
    if len(s) < 2 * period:
        return {"error": f"Ayrıştırma için en az {2*period} gözlem gerekir (period={period})."}
    try:
        res = seasonal_decompose(s.values, period=period, model=model, extrapolate_trend="freq")
    except Exception as e:
        return {"error": f"Ayrıştırma hatası: {e}"}
    trend = res.trend; seasonal = res.seasonal; resid = res.resid
    var_resid = np.nanvar(resid)
    var_detrend = np.nanvar(seasonal + resid)
    seasonal_strength = max(0.0, 1 - var_resid / var_detrend) if var_detrend else np.nan
    return {"trend": trend, "seasonal": seasonal, "resid": resid, "period": period,
            "model": model, "seasonal_strength": float(seasonal_strength)}


def autocorr(s: pd.Series, nlags: int = 20) -> dict:
    nlags = min(nlags, len(s) // 2 - 1) if len(s) > 4 else 1
    return {"acf": acf(s.values, nlags=nlags), "pacf": pacf(s.values, nlags=nlags), "nlags": nlags}


def forecast_holt(s: pd.Series, steps: int = 12, seasonal_periods: int | None = None,
                  trend: str = "add", seasonal: str | None = None) -> dict:
    try:
        kwargs = dict(trend=trend)
        if seasonal and seasonal_periods and len(s) >= 2 * seasonal_periods:
            kwargs.update(seasonal=seasonal, seasonal_periods=seasonal_periods)
        fit = ExponentialSmoothing(s.values.astype(float), **kwargs).fit()
        fc = fit.forecast(steps)
        return {"method": "Holt-Winters (üstel düzeltme)", "fitted": fit.fittedvalues,
                "forecast": np.asarray(fc), "steps": steps, "aic": float(getattr(fit, "aic", np.nan))}
    except Exception as e:
        return {"error": f"Holt-Winters tahmini başarısız: {e}"}


def _adf_stationary(x, alpha=0.05) -> bool:
    try:
        return adfuller(x, autolag="AIC")[1] < alpha
    except Exception:
        return True


def _ndiffs(x, max_d=2, alpha=0.05) -> int:
    """ADF testine göre gereken fark alma (differencing) derecesi."""
    d = 0
    cur = np.asarray(x, float)
    while d < max_d and not _adf_stationary(cur, alpha):
        cur = np.diff(cur)
        d += 1
    return d


def auto_arima(s: pd.Series, max_p: int = 3, max_q: int = 3, seasonal: bool = False,
               m: int = 12, steps: int = 12, alpha: float = 0.05) -> dict:
    """AIC ızgara aramasıyla en iyi (S)ARIMA derecelerini otomatik seçer."""
    import warnings
    x = s.values.astype(float)
    d = _ndiffs(x, 2, alpha)
    D, best = 0, None
    seas_orders = [(0, 0, 0, 0)]
    if seasonal and len(s) >= 2 * m:
        D = 1
        seas_orders = [(P, D, Q, m) for P in range(2) for Q in range(2)]
    results = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for p in range(max_p + 1):
            for q in range(max_q + 1):
                if p == 0 and q == 0 and d == 0:
                    continue
                for so in seas_orders:
                    try:
                        mod = ARIMA(x, order=(p, d, q),
                                    seasonal_order=so if seasonal else (0, 0, 0, 0)).fit()
                        results.append(((p, d, q), so, float(mod.aic), mod))
                    except Exception:
                        continue
    if not results:
        return {"error": "Uygun ARIMA modeli bulunamadı."}
    results.sort(key=lambda r: r[2])
    order, sorder, aic, fit = results[0]
    pred = fit.get_forecast(steps)
    ci = pred.conf_int(alpha=alpha)
    top = pd.DataFrame([{"order": str(o), "seasonal": str(so) if seasonal else "—", "AIC": round(a, 2)}
                        for o, so, a, _ in results[:5]])
    label = f"auto-ARIMA{order}" + (f"×{sorder}" if seasonal else "")
    return {"method": label, "order": order, "seasonal_order": sorder if seasonal else None,
            "aic": aic, "bic": float(fit.bic), "d_selected": d,
            "fitted": np.asarray(fit.fittedvalues), "forecast": np.asarray(pred.predicted_mean),
            "ci_low": np.asarray(ci[:, 0]), "ci_high": np.asarray(ci[:, 1]),
            "steps": steps, "candidates": top, "n_tried": len(results)}


def forecast_arima(s: pd.Series, order=(1, 1, 1), steps: int = 12, alpha: float = 0.05) -> dict:
    try:
        fit = ARIMA(s.values.astype(float), order=order).fit()
        pred = fit.get_forecast(steps)
        mean = np.asarray(pred.predicted_mean)
        ci = pred.conf_int(alpha=alpha)
        return {"method": f"ARIMA{order}", "fitted": np.asarray(fit.fittedvalues),
                "forecast": mean, "ci_low": np.asarray(ci[:, 0]), "ci_high": np.asarray(ci[:, 1]),
                "steps": steps, "aic": float(fit.aic), "bic": float(fit.bic)}
    except Exception as e:
        return {"error": f"ARIMA tahmini başarısız: {e}"}


def interpret_overview(ov: dict, stat: dict, dec: dict | None, alpha: float = 0.05) -> str:
    lines = [f"### Zaman serisi özeti (n = {ov['n']})"]
    trend_dir = ("yükselen" if ov["slope"] > 0 else "düşen" if ov["slope"] < 0 else "yatay")
    lines.append(I.bullet(f"Doğrusal eğim = {ov['slope']:.4f} → seri genel olarak **{trend_dir} trend** gösteriyor "
                          f"(dönem başına ortalama {ov['slope']:+.4f} birim)."))
    # Durağanlık
    lines.append("\n**Durağanlık testleri**")
    if "adf_p" in stat:
        lines.append(I.bullet(
            f"ADF: istatistik = {stat['adf_stat']:.3f}, p = {stat['adf_p']:.4g} → "
            + ("seri **durağan** (birim kök yok)." if stat.get("adf_stationary")
               else "seri **durağan değil** (birim kök var); fark alma (differencing) gerekebilir.")))
    if "kpss_p" in stat:
        lines.append(I.bullet(
            f"KPSS: istatistik = {stat['kpss_stat']:.3f}, p = {stat['kpss_p']:.4g} → "
            + ("durağanlık korunuyor." if stat.get("kpss_stationary") else "durağanlıktan sapma var.")))
    if not stat.get("adf_stationary", True):
        lines.append(I.bullet("Öneri: ARIMA'da d≥1 (fark alma) kullanın ya da seriyi log/fark ile dönüştürün."))
    # Mevsimsellik
    if dec and "error" not in dec:
        ss = dec["seasonal_strength"]
        q = ("çok güçlü" if ss > 0.6 else "belirgin" if ss > 0.3 else "zayıf/ihmal edilebilir")
        lines.append("\n**Mevsimsellik**")
        lines.append(I.bullet(f"Mevsimsel güç ≈ {ss:.2f} ({q}), period = {dec['period']} ({dec['model']} model)."))
    return I.joinlines(lines)


def interpret_forecast(fc: dict, last_value: float) -> str:
    if "error" in fc:
        return fc["error"]
    f = fc["forecast"]
    change = f[-1] - last_value
    lines = [f"### Tahmin — {fc['method']} ({fc['steps']} dönem)"]
    lines.append(I.bullet(f"Son gözlem = {last_value:.3f} → {fc['steps']}. dönem tahmini = {f[-1]:.3f} "
                          f"({change:+.3f} birim, %{(change/last_value*100 if last_value else float('nan')):+.1f})."))
    lines.append(I.bullet(f"Tahmin aralığı: min {np.min(f):.3f} – max {np.max(f):.3f}."))
    if "aic" in fc and fc["aic"] == fc["aic"]:
        lines.append(I.bullet(f"Model AIC = {fc['aic']:.1f} (daha küçük = daha iyi uyum)."))
    if "ci_low" in fc:
        lines.append(I.bullet(f"Son dönem %95 güven aralığı: [{fc['ci_low'][-1]:.3f}, {fc['ci_high'][-1]:.3f}]."))
    lines.append(I.bullet("Not: Tahminler geçmiş kalıbın süreceği varsayımına dayanır; ufuk uzadıkça belirsizlik artar."))
    return I.joinlines(lines)
