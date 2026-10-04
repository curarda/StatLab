"""Nedensellik: fark-içinde-fark (DiD), eğilim skoru eşleştirme, karıştırıcı kontrolü.

Ayrım: ETKİ TAHMİNLERİ (ham gerçekler) ile YORUM (nedensel çıkarım) ayrı döner.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from . import interpret as I


def difference_in_differences(df: pd.DataFrame, outcome: str, group: str, time: str,
                              treated_label, post_label, alpha: float = 0.05) -> dict:
    """DiD: outcome ~ treated + post + treated:post. Etkileşim katsayısı = nedensel etki."""
    data = df[[outcome, group, time]].copy()
    data[outcome] = pd.to_numeric(data[outcome], errors="coerce")
    data = data.dropna()
    data["treated"] = (data[group].astype(str) == str(treated_label)).astype(int)
    data["post"] = (data[time].astype(str) == str(post_label)).astype(int)
    if data["treated"].nunique() < 2 or data["post"].nunique() < 2:
        return {"error": "Treated/kontrol ve öncesi/sonrası düzeyleri ayrıştırılamadı."}
    model = smf.ols(f"Q('{outcome}') ~ treated + post + treated:post", data).fit()
    did = model.params.get("treated:post", np.nan)
    p = model.pvalues.get("treated:post", np.nan)
    ci = model.conf_int(alpha=alpha).loc["treated:post"] if "treated:post" in model.params.index else [np.nan, np.nan]
    # 2x2 ortalama tablosu
    cell = data.groupby(["treated", "post"])[outcome].mean().unstack()
    return {"did": float(did), "p": float(p), "ci": (float(ci[0]), float(ci[1])),
            "means": cell, "outcome": outcome, "alpha": alpha, "n": int(model.nobs)}


def propensity_matching(df: pd.DataFrame, treatment: str, outcome: str, covariates: list[str],
                        alpha: float = 0.05) -> dict:
    """Eğilim skoru (logistic) + en yakın komşu eşleştirme ile ATT tahmini."""
    from sklearn.linear_model import LogisticRegression
    cols = [treatment, outcome] + covariates
    data = df[cols].apply(pd.to_numeric, errors="coerce").dropna().reset_index(drop=True)
    if data[treatment].nunique() != 2:
        return {"error": "Tedavi değişkeni tam 2 düzeyli (0/1) olmalı."}
    t = data[treatment].astype(int).values
    y = data[outcome].values
    X = data[covariates].values
    ps = LogisticRegression(max_iter=1000).fit(X, t).predict_proba(X)[:, 1]
    treated_idx = np.where(t == 1)[0]
    control_idx = np.where(t == 0)[0]
    if len(treated_idx) == 0 or len(control_idx) == 0:
        return {"error": "Her iki grupta da gözlem gerekir."}
    ps_c = ps[control_idx]
    # en yakın komşu (PS farkı)
    matched_effects = []
    for ti in treated_idx:
        j = control_idx[np.argmin(np.abs(ps_c - ps[ti]))]
        matched_effects.append(y[ti] - y[j])
    att = float(np.mean(matched_effects))
    naive = float(y[t == 1].mean() - y[t == 0].mean())
    from scipy import stats as _st
    se = np.std(matched_effects, ddof=1) / np.sqrt(len(matched_effects))
    p = float(2 * (1 - _st.norm.cdf(abs(att / se)))) if se else np.nan
    return {"att": att, "naive": naive, "p": p, "n_treated": len(treated_idx),
            "n_control": len(control_idx), "outcome": outcome, "treatment": treatment,
            "covariates": covariates, "alpha": alpha}


def confounding_check(df: pd.DataFrame, x: str, y: str, confounders: list[str], alpha: float = 0.05) -> dict:
    """X→Y ilişkisinin karıştırıcılar kontrol edilince değişip değişmediği."""
    data = df[[x, y] + confounders].apply(pd.to_numeric, errors="coerce").dropna()
    raw = smf.ols(f"Q('{y}') ~ Q('{x}')", data).fit()
    adj = smf.ols(f"Q('{y}') ~ Q('{x}') + " + " + ".join(f"Q('{c}')" for c in confounders), data).fit()
    b_raw = raw.params.iloc[1]
    b_adj = adj.params.iloc[1]
    p_adj = adj.pvalues.iloc[1]
    change = (b_raw - b_adj) / b_raw * 100 if b_raw else np.nan
    return {"b_raw": float(b_raw), "b_adj": float(b_adj), "p_adj": float(p_adj),
            "change_pct": float(change), "x": x, "y": y, "confounders": confounders, "alpha": alpha}


# ---------------- Yorum katmanı ----------------

def interpret_did(res: dict) -> str:
    if "error" in res:
        return res["error"]
    a = res["alpha"]
    lines = ["### Fark-içinde-fark (DiD) — nedensel yorum katmanı"]
    lines.append(I.bullet(f"**Nedensel etki tahmini = {res['did']:.4f}** (p={res['p']:.4g}, "
                          f"%{int((1-a)*100)} GA [{res['ci'][0]:.3f}, {res['ci'][1]:.3f}])."))
    if res["p"] < a:
        lines.append(I.bullet(f"Müdahalenin, kontrol grubuna kıyasla {res['outcome']} üzerinde **anlamlı nedensel etkisi** var "
                              f"({'artış' if res['did']>0 else 'azalış'})."))
    else:
        lines.append(I.bullet("Etkileşim anlamlı değil; müdahalenin ayrı bir nedensel etkisine dair yeterli kanıt yok."))
    lines.append(I.bullet("DiD, iki grubun müdahale olmasaydı **paralel gideceği** varsayımına dayanır — bu varsayımı "
                          "öncesi dönem trendleriyle kontrol edin."))
    return I.joinlines(lines)


def interpret_psm(res: dict) -> str:
    if "error" in res:
        return res["error"]
    a = res["alpha"]
    lines = ["### Eğilim skoru eşleştirme — nedensel yorum katmanı"]
    lines.append(I.bullet(f"Ham fark: {res['naive']:.4f} → **eşleştirme sonrası etki (ATT): {res['att']:.4f}** "
                          f"(p={res['p']:.4g})."))
    diff = abs(res["naive"] - res["att"])
    if diff > abs(res["naive"]) * 0.2:
        lines.append(I.bullet("Eşleştirme etkiyi **belirgin değiştirdi** → ham karşılaştırma karıştırıcılardan etkilenmiş; "
                              "eşleştirilmiş tahmin daha güvenilir."))
    else:
        lines.append(I.bullet("Eşleştirme etkiyi pek değiştirmedi → gözlenen karıştırıcılar ilişkiyi az saptırmış."))
    lines.append(I.bullet(f"{res['n_treated']} tedavi, {res['n_control']} kontrol gözlemi eşleştirildi."))
    lines.append(I.bullet("Yalnızca **gözlenen** karıştırıcıları dengeler; ölçülmemiş karıştırıcı hâlâ yanlılık yaratabilir."))
    return I.joinlines(lines)


def interpret_confounding(res: dict) -> str:
    a = res["alpha"]
    lines = ["### Karıştırıcı kontrolü — nedensel yorum katmanı"]
    lines.append(I.bullet(f"{res['x']}→{res['y']} katsayısı: ham {res['b_raw']:.4f} → "
                          f"kontrol sonrası **{res['b_adj']:.4f}** (%{res['change_pct']:.0f} değişim, p={res['p_adj']:.4g})."))
    if abs(res["change_pct"]) > 20 and res["p_adj"] >= a:
        lines.append(I.bullet(f"Karıştırıcı(lar) eklenince ilişki **çöktü/anlamsızlaştı** → {res['x']} ile {res['y']} "
                              "arasındaki görünen ilişki büyük ölçüde **sahte (spurious)**; karıştırıcılar üzerinden."))
    elif abs(res["change_pct"]) > 20:
        lines.append(I.bullet("İlişki zayıfladı ama hâlâ anlamlı → kısmen karıştırıcı, kısmen doğrudan etki."))
    else:
        lines.append(I.bullet(f"İlişki kontrol sonrası büyük ölçüde korundu → {res['x']}'in {res['y']} üzerindeki etkisi "
                              "bu karıştırıcılarla açıklanamıyor (nedensellik için daha güçlü kanıt)."))
    lines.append(I.bullet("Not: Yalnızca listelenen karıştırıcılar kontrol edildi; gerçek nedensellik için deneysel tasarım en güçlüsüdür."))
    return I.joinlines(lines)
