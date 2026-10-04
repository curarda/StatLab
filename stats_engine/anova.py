"""Tek ve çift yönlü ANOVA + post-hoc (Tukey HSD)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd

from . import interpret as I


def one_way(df: pd.DataFrame, value: str, group: str, alpha: float = 0.05) -> dict:
    data = df[[value, group]].copy()
    data[value] = pd.to_numeric(data[value], errors="coerce")
    data = data.dropna()
    groups = [g[value].values for _, g in data.groupby(group)]
    labels = [str(k) for k, _ in data.groupby(group)]
    if len(groups) < 2:
        return {"error": "ANOVA için en az 2 grup gerekir."}

    f, p = stats.f_oneway(*groups)
    # Levene (varyans homojenliği)
    try:
        lev_stat, lev_p = stats.levene(*groups, center="median")
    except Exception:
        lev_stat, lev_p = np.nan, np.nan
    # Etki büyüklüğü (eta kare) — statsmodels ile
    model = smf.ols(f"Q('{value}') ~ C(Q('{group}'))", data=data).fit()
    aov = sm.stats.anova_lm(model, typ=2)
    ss_between = aov["sum_sq"].iloc[0]
    ss_total = aov["sum_sq"].sum()
    eta2 = float(ss_between / ss_total) if ss_total else np.nan
    omega2 = _omega_squared(aov)

    group_stats = data.groupby(group)[value].agg(["count", "mean", "std"]).reset_index()
    group_stats.columns = [group, "N", "Ortalama", "Std"]

    out = {
        "value": value, "group": group, "labels": labels,
        "k": len(groups), "n": int(len(data)),
        "f": float(f), "p": float(p),
        "df_between": int(aov["df"].iloc[0]), "df_within": int(aov["df"].iloc[1]),
        "eta2": eta2, "omega2": omega2,
        "levene_p": float(lev_p) if lev_p == lev_p else np.nan,
        "anova_table": aov, "group_stats": group_stats, "alpha": alpha, "data": data,
    }
    # Post-hoc yalnızca anlamlıysa
    if p < alpha and len(groups) > 2:
        try:
            tuk = pairwise_tukeyhsd(data[value].values, data[group].astype(str).values, alpha=alpha)
            out["tukey"] = pd.DataFrame(tuk.summary().data[1:], columns=tuk.summary().data[0])
        except Exception:
            pass
    return out


def two_way(df: pd.DataFrame, value: str, factor_a: str, factor_b: str, alpha: float = 0.05) -> dict:
    if factor_a == factor_b:
        return {"error": "İki faktör farklı olmalıdır."}
    data = df[[value, factor_a, factor_b]].copy()
    data[value] = pd.to_numeric(data[value], errors="coerce")
    data = data.dropna()
    if data.empty:
        return {"error": "Geçerli veri yok."}
    formula = f"Q('{value}') ~ C(Q('{factor_a}')) + C(Q('{factor_b}')) + C(Q('{factor_a}')):C(Q('{factor_b}'))"
    model = smf.ols(formula, data=data).fit()
    aov = sm.stats.anova_lm(model, typ=2)
    ss_total = aov["sum_sq"].sum()
    aov = aov.copy()
    aov["eta_sq"] = aov["sum_sq"] / ss_total
    labels = {0: factor_a, 1: factor_b, 2: f"{factor_a} × {factor_b} (etkileşim)"}
    aov.index = [labels.get(i, idx) for i, idx in enumerate(aov.index[:-1])] + ["Artık"]
    return {"value": value, "factor_a": factor_a, "factor_b": factor_b,
            "anova_table": aov, "alpha": alpha, "r2": float(model.rsquared), "n": int(model.nobs)}


def _omega_squared(aov) -> float:
    try:
        ss_b = aov["sum_sq"].iloc[0]
        df_b = aov["df"].iloc[0]
        ms_w = aov["sum_sq"].iloc[1] / aov["df"].iloc[1]
        ss_t = aov["sum_sq"].sum()
        return float((ss_b - df_b * ms_w) / (ss_t + ms_w))
    except Exception:
        return np.nan


def interpret_one_way(res: dict) -> str:
    if "error" in res:
        return res["error"]
    a = res["alpha"]
    lines = [f"### Tek yönlü ANOVA — **{res['value']}** ~ **{res['group']}**"]
    lines.append(I.bullet(f"{res['k']} grup, toplam n = {res['n']}."))
    lines.append(I.bullet(f"F({res['df_between']}, {res['df_within']}) = {res['f']:.3f}, p = {res['p']:.4g} {I.stars(res['p'])}"))
    lines.append(I.bullet(f"Etki büyüklüğü: η² = {res['eta2']:.3f} ({I.eta_sq_label(res['eta2'])}), ω² = {res['omega2']:.3f}"))
    # Levene
    if res["levene_p"] == res["levene_p"]:
        lev_ok = res["levene_p"] >= a
        lines.append(I.bullet(
            f"Levene varyans homojenliği: p = {res['levene_p']:.4g} → "
            + ("varyanslar homojen (ANOVA varsayımı sağlanıyor)." if lev_ok
               else "**varyanslar homojen değil**; Welch ANOVA veya Games-Howell düşünülmeli.")))
    lines.append("")
    lines.append("**Karar:** " + I.p_sentence(res["p"], a))
    if res["p"] < a:
        lines.append(I.bullet(
            f"Gruplardan en az birinin {res['value']} ortalaması diğerlerinden **anlamlı biçimde farklı.** "
            f"Grup farkı {res['value']} değişkenindeki değişimin ~%{res['eta2']*100:.1f}'ini açıklıyor."))
        if "tukey" in res:
            lines.append(I.bullet("Hangi grupların farklı olduğu için aşağıdaki Tukey HSD ikili karşılaştırma tablosuna bakın "
                                  "(`reject=True` → o ikili anlamlı farklı)."))
    else:
        lines.append(I.bullet(f"Grup ortalamaları arasında anlamlı fark yok; {res['group']} düzeyleri {res['value']} üzerinde ayrışmıyor."))
    return I.joinlines(lines)


def interpret_two_way(res: dict) -> str:
    if "error" in res:
        return res["error"]
    a = res["alpha"]
    aov = res["anova_table"]
    lines = [f"### Çift yönlü ANOVA — **{res['value']}** ~ {res['factor_a']} + {res['factor_b']} + etkileşim",
             I.bullet(f"n = {res['n']}, model R² = {res['r2']:.3f}"), ""]
    for idx, row in aov.iterrows():
        if idx == "Artık":
            continue
        p = row.get("PR(>F)", np.nan)
        f = row.get("F", np.nan)
        eta = row.get("eta_sq", np.nan)
        verdict = "**anlamlı**" if (p == p and p < a) else "anlamlı değil"
        lines.append(I.bullet(
            f"**{idx}**: F = {f:.3f}, p = {p:.4g} {I.stars(p)}, η² = {eta:.3f} → {verdict}."))
    # etkileşim yorumu
    inter_row = [r for i, r in aov.iterrows() if "etkileşim" in str(i)]
    if inter_row:
        pi = inter_row[0].get("PR(>F)", np.nan)
        if pi == pi and pi < a:
            lines.append("")
            lines.append(I.bullet("Etkileşim anlamlı: bir faktörün etkisi diğerinin düzeyine göre **değişiyor**; "
                                  "ana etkiler tek başına dikkatli yorumlanmalı."))
        elif pi == pi:
            lines.append("")
            lines.append(I.bullet("Etkileşim anlamlı değil: faktörlerin etkileri birbirinden bağımsız (toplanabilir) kabul edilebilir."))
    return I.joinlines(lines)
