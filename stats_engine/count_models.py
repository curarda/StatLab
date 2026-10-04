"""Sayım verisi modelleri: Poisson ve Negatif Binom GLM."""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

from . import interpret as I


def fit_count(df: pd.DataFrame, y: str, x_vars: list[str], family: str = "poisson",
              categorical: list[str] | None = None, alpha: float = 0.05,
              exposure: str | None = None) -> dict:
    """Poisson veya Negatif Binom regresyon (log bağlantı)."""
    categorical = categorical or []
    cols = [y] + x_vars + ([exposure] if exposure else [])
    data = df[cols].copy()
    for v in [y] + [v for v in x_vars if v not in categorical] + ([exposure] if exposure else []):
        data[v] = pd.to_numeric(data[v], errors="coerce")
    data = data.dropna()
    if (data[y] < 0).any():
        return {"error": "Sayım modeli negatif değer içeremez (y ≥ 0 olmalı)."}
    terms = [f"C(Q('{v}'))" if v in categorical else f"Q('{v}')" for v in x_vars]
    formula = f"Q('{y}') ~ " + " + ".join(terms)
    fam = sm.families.NegativeBinomial() if family == "negbin" else sm.families.Poisson()
    kwargs = {}
    if exposure:
        kwargs["exposure"] = data[exposure].values
    try:
        model = smf.glm(formula, data=data, family=fam, **kwargs).fit()
    except Exception as e:
        return {"error": f"Model uyduramadı: {e}"}

    params = model.params
    conf = model.conf_int(alpha=alpha)
    rows = []
    for name in params.index:
        b = float(params[name])
        rows.append({"Terim": _pretty(name), "Katsayı (β)": b, "IRR (oran oranı)": float(np.exp(b)),
                     "IRR CI alt": float(np.exp(conf.loc[name, 0])), "IRR CI üst": float(np.exp(conf.loc[name, 1])),
                     "z": float(model.tvalues[name]), "p": float(model.pvalues[name]),
                     "Anlam": I.stars(float(model.pvalues[name])) or "—"})
    coef_df = pd.DataFrame(rows)

    # Aşırı yayılım (overdispersion) kontrolü — Pearson chi2 / df
    pearson_chi2 = float(model.pearson_chi2)
    df_resid = int(model.df_resid)
    dispersion = pearson_chi2 / df_resid if df_resid else np.nan
    # McFadden sözde-R²
    ll = model.llf
    ll0 = smf.glm(f"Q('{y}') ~ 1", data=data, family=fam, **kwargs).fit().llf
    pseudo_r2 = 1 - ll / ll0 if ll0 else np.nan

    return {"y": y, "x_vars": x_vars, "family": family, "n": int(model.nobs),
            "coef_df": coef_df, "coef_params": params, "aic": float(model.aic),
            "bic": float(model.bic_llf) if hasattr(model, "bic_llf") else float(model.bic),
            "dispersion": float(dispersion), "pseudo_r2": float(pseudo_r2),
            "alpha": alpha, "summary_text": model.summary().as_text()}


def _pretty(name: str) -> str:
    return (name.replace("Q('", "").replace("')", "").replace("C(", "")
                .replace(")", "").replace("[T.", " = ").replace("]", ""))


def interpret_count(res: dict) -> str:
    if "error" in res:
        return res["error"]
    a = res["alpha"]
    fam = "Poisson" if res["family"] == "poisson" else "Negatif Binom"
    lines = [f"### {fam} regresyon — sayım değişkeni: **{res['y']}**"]
    lines.append(I.bullet(f"n = {res['n']}, sözde-R² = {res['pseudo_r2']:.3f}, AIC = {res['aic']:.1f}."))

    lines.append("\n**1) Aşırı yayılım (overdispersion) kontrolü**")
    d = res["dispersion"]
    lines.append(I.bullet(f"Yayılım oranı (Pearson χ²/sd) = {d:.2f}."))
    if res["family"] == "poisson" and d > 1.5:
        lines.append(I.bullet("**> 1.5**: Poisson için aşırı yayılım var (varyans > ortalama). "
                              "**Negatif Binom modeli daha uygun** — aileyi değiştirin."))
    elif res["family"] == "poisson":
        lines.append(I.bullet("~1 civarı: Poisson varsayımı (ortalama = varyans) makul."))
    else:
        lines.append(I.bullet("Negatif Binom, aşırı yayılımı zaten hesaba katar."))

    lines.append("\n**2) Katsayılar ve oran oranları (IRR)**")
    for _, row in res["coef_df"].iterrows():
        term, irr, p = row["Terim"], row["IRR (oran oranı)"], row["p"]
        if term.lower() == "intercept":
            lines.append(I.bullet(f"**Sabit**: baz sayım oranı = exp({row['Katsayı (β)']:.3f}) = {irr:.3f}."))
            continue
        if irr >= 1:
            eff = f"beklenen sayıyı **{irr:.3f} katına** çıkarır (~%{(irr-1)*100:.1f} artış)"
        else:
            eff = f"beklenen sayıyı **{irr:.3f} katına** düşürür (~%{(1-irr)*100:.1f} azalış)"
        sig = "**anlamlı**" if p < a else "anlamlı değil"
        lines.append(I.bullet(f"**{term}**: IRR = {irr:.3f} {row['Anlam']} (p={p:.4g}) → 1 birim artış {eff}. Etki {sig}."))
    return I.joinlines(lines)
