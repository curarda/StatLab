"""Deney kesinliğini artıran yöntemler: ANCOVA ve CUPED varyans azaltma."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from . import interpret as I


def ancova(df: pd.DataFrame, dv: str, group: str, covariates: list[str], alpha: float = 0.05) -> dict:
    """Ortak değişken(ler) kontrol edilerek grup etkisi (ANCOVA)."""
    import statsmodels.formula.api as smf
    import statsmodels.api as sm
    data = df[[dv, group] + covariates].copy()
    data[dv] = pd.to_numeric(data[dv], errors="coerce")
    for c in covariates:
        data[c] = pd.to_numeric(data[c], errors="coerce")
    data = data.dropna()
    terms = [f"C(Q('{group}'))"] + [f"Q('{c}')" for c in covariates]
    model = smf.ols(f"Q('{dv}') ~ " + " + ".join(terms), data).fit()
    aov = sm.stats.anova_lm(model, typ=2)
    grp_row = aov.iloc[0]
    ss_total = aov["sum_sq"].sum()
    eta2 = grp_row["sum_sq"] / ss_total if ss_total else np.nan
    # düzeltilmiş ortalamalar (kovaryasları ortalamada sabitleyerek)
    adj = {}
    means_cov = {c: data[c].mean() for c in covariates}
    for lvl in data[group].dropna().unique():
        row = {group: lvl, **means_cov}
        pred = model.predict(pd.DataFrame([row]))
        adj[str(lvl)] = float(pred.iloc[0])
    # ham ortalamalar
    raw = data.groupby(group)[dv].mean().to_dict()
    return {"dv": dv, "group": group, "covariates": covariates,
            "F": float(grp_row["F"]), "p": float(grp_row["PR(>F)"]), "eta2": float(eta2),
            "adj_means": adj, "raw_means": {str(k): float(v) for k, v in raw.items()},
            "r2": float(model.rsquared), "n": int(model.nobs), "alpha": alpha}


def cuped(pre: pd.Series, post_a: pd.Series, post_b: pd.Series, alpha: float = 0.05) -> dict:
    """CUPED: deney öncesi ölçümle (pre) varyans azaltma → daha güçlü A/B.
    pre, post aynı sırada; a=kontrol, b=varyant maskeleri değil, ayrı seriler beklenir.
    Burada basitlik için: pre ve post tek seri (tüm birimler), varyant etiketiyle birlikte kullanılır.
    """
    return {"note": "CUPED için cuped_ab kullanın."}


def cuped_ab(df: pd.DataFrame, metric: str, pre_metric: str, variant: str,
             control_label, variant_label, alpha: float = 0.05) -> dict:
    """CUPED ile A/B: pre_metric kullanarak metric'in varyansını azaltıp t-testi."""
    data = df[[metric, pre_metric, variant]].copy()
    data[metric] = pd.to_numeric(data[metric], errors="coerce")
    data[pre_metric] = pd.to_numeric(data[pre_metric], errors="coerce")
    data = data.dropna()
    y = data[metric].values
    x = data[pre_metric].values
    theta = np.cov(y, x)[0, 1] / np.var(x, ddof=1) if np.var(x, ddof=1) else 0.0
    y_cuped = y - theta * (x - x.mean())
    data = data.assign(_cuped=y_cuped)
    a = data.loc[data[variant] == control_label]
    b = data.loc[data[variant] == variant_label]
    # ham t-testi
    t_raw, p_raw = stats.ttest_ind(a[metric], b[metric], equal_var=False)
    # CUPED t-testi
    t_c, p_c = stats.ttest_ind(a["_cuped"], b["_cuped"], equal_var=False)
    var_reduction = 1 - np.var(y_cuped, ddof=1) / np.var(y, ddof=1) if np.var(y, ddof=1) else 0
    return {"theta": float(theta), "var_reduction": float(var_reduction),
            "p_raw": float(p_raw), "p_cuped": float(p_c),
            "mean_a": float(a[metric].mean()), "mean_b": float(b[metric].mean()),
            "lift": float(b[metric].mean() - a[metric].mean()), "alpha": alpha,
            "n_a": len(a), "n_b": len(b)}


def tost(a: pd.Series, b: pd.Series, low: float, upp: float, alpha: float = 0.05) -> dict:
    """Eşdeğerlik testi (TOST): iki grubun ortalaması [low, upp] bandında pratikte eşit mi?"""
    from statsmodels.stats.weightstats import ttost_ind
    sa = pd.to_numeric(a, errors="coerce").dropna()
    sb = pd.to_numeric(b, errors="coerce").dropna()
    p, (t1, p1, df1), (t2, p2, df2) = ttost_ind(sa, sb, low, upp, usevar="unequal")
    diff = sa.mean() - sb.mean()
    return {"p": float(p), "diff": float(diff), "low": low, "upp": upp,
            "p1": float(p1), "p2": float(p2), "equivalent": float(p) < alpha,
            "mean_a": float(sa.mean()), "mean_b": float(sb.mean()), "alpha": alpha}


def interpret_tost(res: dict) -> str:
    a = res["alpha"]
    lines = ["### Eşdeğerlik testi (TOST)"]
    lines.append(I.bullet(f"Ortalama fark = {res['diff']:.4f}; eşdeğerlik bandı [{res['low']}, {res['upp']}]."))
    lines.append(I.bullet(f"TOST p = {res['p']:.4g} (iki tek-yönlü testin büyüğü)."))
    if res["equivalent"]:
        lines.append(I.bullet(f"p < {a:g} → **iki grup pratik olarak EŞDEĞER**: fark, önemsiz kabul edilen bandın içinde. "
                              "Bu, klasik testin yapamadığı şeydir — 'anlamlı fark yok' değil, **'fark yok'** kanıtı."))
    else:
        lines.append(I.bullet(f"p ≥ {a:g} → eşdeğerlik **kanıtlanamadı**; fark önemsiz banttan taşabilir "
                              "(ya gerçekten farklılar ya da örneklem yetersiz)."))
    lines.append(I.bullet("Eşdeğerlik bandını iş/klinik olarak 'önemsiz' saydığınız fark aralığına göre belirleyin."))
    return I.joinlines(lines)


def interpret_ancova(res: dict) -> str:
    a = res["alpha"]
    lines = [f"### ANCOVA — {res['dv']} ~ {res['group']} | kontrol: {', '.join(res['covariates'])}"]
    lines.append(I.bullet(f"Grup etkisi (kovaryas kontrol edilmiş): F = {res['F']:.3f}, p = {res['p']:.4g} "
                          f"{I.stars(res['p'])}, η² = {res['eta2']:.3f}, model R² = {res['r2']:.3f}."))
    lines.append("**Ortalamalar (ham → düzeltilmiş):**")
    for k in res["adj_means"]:
        lines.append(I.bullet(f"{k}: {res['raw_means'].get(k, float('nan')):.3f} → **{res['adj_means'][k]:.3f}** (düzeltilmiş)"))
    lines.append("")
    if res["p"] < a:
        lines.append(I.bullet("Kovaryas(lar) sabitlendiğinde bile gruplar arasında **anlamlı fark var** — "
                              "etki ortak değişkenle açıklanamıyor."))
    else:
        lines.append(I.bullet("Kovaryas kontrol edilince grup farkı anlamlı değil — ham fark büyük olasılıkla "
                              "ortak değişkendeki dengesizlikten kaynaklanıyordu."))
    lines.append(I.bullet("ANCOVA, ilgili bir ortak değişken ekleyerek hatayı azaltır → aynı örneklemle **daha yüksek güç.**"))
    return I.joinlines(lines)


def interpret_cuped(res: dict) -> str:
    lines = ["### CUPED — varyans azaltmalı A/B"]
    lines.append(I.bullet(f"Öncül ölçümle **varyans %{res['var_reduction']*100:.1f} azaltıldı** (θ={res['theta']:.3f})."))
    lines.append(I.bullet(f"Ortalamalar: A={res['mean_a']:.4f}, B={res['mean_b']:.4f} (fark {res['lift']:+.4f})."))
    lines.append(I.bullet(f"p-değeri: ham {res['p_raw']:.4g} → **CUPED {res['p_cuped']:.4g}**."))
    if res["p_cuped"] < res["p_raw"]:
        lines.append(I.bullet("CUPED daha küçük p verdi → aynı veriyle **daha yüksek istatistiksel güç**; "
                              "etkiyi daha erken/net saptar."))
    lines.append(I.bullet("CUPED, deney öncesi davranışla ilişkili bir metrik varsa A/B testlerinde standarttır "
                          "(etkiyi değiştirmez, gürültüyü azaltır)."))
    return I.joinlines(lines)
