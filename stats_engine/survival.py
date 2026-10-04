"""Sağkalım analizi: Kaplan-Meier, log-rank testi, Cox orantılı tehlike modeli."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import interpret as I


def kaplan_meier(df: pd.DataFrame, duration: str, event: str, group: str | None = None) -> dict:
    from lifelines import KaplanMeierFitter
    cols = [duration, event] + ([group] if group else [])
    data = df[cols].copy()
    data[duration] = pd.to_numeric(data[duration], errors="coerce")
    data[event] = pd.to_numeric(data[event], errors="coerce")
    data = data.dropna()
    if data.empty:
        return {"error": "Geçerli veri yok. Olay sütunu 1=olay, 0=sansür olmalı."}
    curves = {}
    medians = {}
    if group:
        for gval, gdf in data.groupby(group):
            kmf = KaplanMeierFitter().fit(gdf[duration], gdf[event], label=str(gval))
            curves[str(gval)] = kmf.survival_function_.reset_index()
            medians[str(gval)] = float(kmf.median_survival_time_)
    else:
        kmf = KaplanMeierFitter().fit(data[duration], data[event], label="Tümü")
        curves["Tümü"] = kmf.survival_function_.reset_index()
        medians["Tümü"] = float(kmf.median_survival_time_)

    out = {"curves": curves, "medians": medians, "group": group, "n": int(len(data)),
           "n_events": int(data[event].sum())}
    # log-rank (2+ grup)
    if group and data[group].nunique() >= 2:
        out["logrank"] = _logrank(data, duration, event, group)
    return out


def _logrank(data, duration, event, group) -> dict:
    from lifelines.statistics import multivariate_logrank_test
    res = multivariate_logrank_test(data[duration], data[group], data[event])
    return {"stat": float(res.test_statistic), "p": float(res.p_value)}


def cox_ph(df: pd.DataFrame, duration: str, event: str, covariates: list[str],
           alpha: float = 0.05) -> dict:
    from lifelines import CoxPHFitter
    cols = [duration, event] + covariates
    data = df[cols].apply(pd.to_numeric, errors="coerce").dropna()
    if data.empty or len(data) <= len(covariates) + 1:
        return {"error": "Cox modeli için yeterli veri yok (sayısal ortak değişkenler gerekir)."}
    try:
        cph = CoxPHFitter()
        cph.fit(data, duration_col=duration, event_col=event)
    except Exception as e:
        return {"error": f"Cox modeli uyduramadı: {e}"}
    s = cph.summary
    table = pd.DataFrame({
        "Değişken": s.index,
        "Katsayı (β)": s["coef"].values,
        "Tehlike Oranı (HR)": s["exp(coef)"].values,
        "HR CI alt": s["exp(coef) lower 95%"].values,
        "HR CI üst": s["exp(coef) upper 95%"].values,
        "p": s["p"].values,
        "Anlam": [I.stars(p) or "—" for p in s["p"].values],
    })
    return {"table": table, "concordance": float(cph.concordance_index_),
            "n": int(cph._n_examples), "alpha": alpha,
            "log_likelihood_p": float(cph.log_likelihood_ratio_test().p_value)}


def interpret_km(res: dict, alpha: float = 0.05) -> str:
    if "error" in res:
        return res["error"]
    lines = [f"### Kaplan-Meier sağkalım — n={res['n']}, olay sayısı={res['n_events']}"]
    for g, med in res["medians"].items():
        mtxt = "ulaşılmadı (>%50 hayatta)" if (med != med or np.isinf(med)) else f"{med:.2f}"
        lines.append(I.bullet(f"**{g}**: medyan sağkalım süresi = {mtxt}."))
    if "logrank" in res:
        lr = res["logrank"]
        lines.append("\n**Log-rank testi (gruplar arası fark)**")
        lines.append(I.bullet(f"χ² = {lr['stat']:.3f}, p = {lr['p']:.4g}"))
        if lr["p"] < alpha:
            lines.append(I.bullet("p < α → gruplar arasında sağkalım eğrileri **anlamlı biçimde farklı.**"))
        else:
            lines.append(I.bullet("p ≥ α → gruplar arası sağkalım farkı anlamlı değil."))
    lines.append(I.bullet("Sağkalım eğrisi zamanla olayı yaşamamış (hayatta kalan) birim oranını gösterir; "
                          "sansürlü gözlemler (olayı yaşamadan çıkanlar) hesaba katılır."))
    return I.joinlines(lines)


def interpret_cox(res: dict) -> str:
    if "error" in res:
        return res["error"]
    a = res["alpha"]
    lines = [f"### Cox orantılı tehlike modeli — n={res['n']}"]
    lines.append(I.bullet(f"Uyum: C-indeksi (concordance) = {res['concordance']:.3f} "
                          f"({'zayıf' if res['concordance']<0.6 else 'orta' if res['concordance']<0.7 else 'iyi'}). "
                          f"Model LR testi p = {res['log_likelihood_p']:.4g}."))
    lines.append("\n**Tehlike oranları (HR)**")
    for _, r in res["table"].iterrows():
        hr, p = r["Tehlike Oranı (HR)"], r["p"]
        if hr >= 1:
            eff = f"olay riskini **{hr:.3f} katına** çıkarır (~%{(hr-1)*100:.1f} artış)"
        else:
            eff = f"olay riskini **{hr:.3f} katına** düşürür (~%{(1-hr)*100:.1f} azalış)"
        sig = "**anlamlı**" if p < a else "anlamlı değil"
        lines.append(I.bullet(f"**{r['Değişken']}**: HR = {hr:.3f} {r['Anlam']} (p={p:.4g}) → 1 birim artış {eff}. Etki {sig}."))
    lines.append(I.bullet("HR > 1 riski artırır (kötü prognoz), HR < 1 koruyucudur."))
    return I.joinlines(lines)
