"""OLS regresyon, katsayı anlamlılığı ve tanı (diagnostic) testleri."""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.stattools import durbin_watson, jarque_bera

from . import interpret as I


def fit_ols(df: pd.DataFrame, y: str, x_vars: list[str], categorical: list[str] | None = None,
            alpha: float = 0.05) -> dict:
    """Çoklu doğrusal regresyon (OLS). Kategorik değişkenler C() ile sarılır."""
    categorical = categorical or []
    terms = []
    for v in x_vars:
        terms.append(f"C(Q('{v}'))" if v in categorical else f"Q('{v}')")
    formula = f"Q('{y}') ~ " + " + ".join(terms)
    data = df[[y] + x_vars].copy()
    # sayısal olması gerekenleri çevir
    for v in [y] + [v for v in x_vars if v not in categorical]:
        data[v] = pd.to_numeric(data[v], errors="coerce")
    data = data.dropna()
    if len(data) <= len(x_vars) + 1:
        return {"error": "Yeterli gözlem yok (parametre sayısından fazla satır gerekir)."}

    model = smf.ols(formula, data=data).fit()
    return _package(model, y, x_vars, data, alpha, categorical)


def _package(model, y, x_vars, data, alpha, categorical) -> dict:
    params = model.params
    bse = model.bse
    tvals = model.tvalues
    pvals = model.pvalues
    conf = model.conf_int(alpha=alpha)

    coef_rows = []
    for name in params.index:
        coef_rows.append({
            "Terim": _pretty_term(name),
            "Katsayı (β)": float(params[name]),
            "Std. Hata": float(bse[name]),
            "t": float(tvals[name]),
            "p": float(pvals[name]),
            "Anlam": I.stars(float(pvals[name])) or "—",
            f"CI alt": float(conf.loc[name, 0]),
            f"CI üst": float(conf.loc[name, 1]),
        })
    coef_df = pd.DataFrame(coef_rows)

    # Tanı testleri
    resid = model.resid
    diag = {}
    diag["durbin_watson"] = float(durbin_watson(resid))
    try:
        jb, jbp, skew, kurt = jarque_bera(resid)
        diag["jarque_bera"] = float(jb)
        diag["jarque_bera_p"] = float(jbp)
    except Exception:
        diag["jarque_bera_p"] = np.nan
    try:
        bp = het_breuschpagan(resid, model.model.exog)
        diag["breusch_pagan_p"] = float(bp[1])
    except Exception:
        diag["breusch_pagan_p"] = np.nan

    # VIF (çoklu bağıntı) — yalnızca sayısal öngörücüler
    vif_df = _vif(data, [v for v in x_vars if v not in categorical])

    out = {
        "y": y,
        "x_vars": x_vars,
        "formula": model.model.formula,
        "n": int(model.nobs),
        "df_model": int(model.df_model),
        "df_resid": int(model.df_resid),
        "r2": float(model.rsquared),
        "adj_r2": float(model.rsquared_adj),
        "f_stat": float(model.fvalue),
        "f_pvalue": float(model.f_pvalue),
        "aic": float(model.aic),
        "bic": float(model.bic),
        "rmse": float(np.sqrt(model.mse_resid)),
        "cond_no": float(model.condition_number),
        "coef_df": coef_df,
        "coef_params": params,
        "diag": diag,
        "vif_df": vif_df,
        "alpha": alpha,
        "resid": resid,
        "fitted": model.fittedvalues,
        "model": model,
        "summary_text": model.summary().as_text(),
    }
    return out


def _pretty_term(name: str) -> str:
    return (name.replace("Q('", "").replace("')", "")
                .replace("C(", "").replace(")", "")
                .replace("[T.", " = ").replace("]", ""))


def _vif(data: pd.DataFrame, num_vars: list[str]) -> pd.DataFrame:
    num_vars = [v for v in num_vars if v in data.columns]
    if len(num_vars) < 2:
        return pd.DataFrame(columns=["Değişken", "VIF"])
    X = data[num_vars].apply(pd.to_numeric, errors="coerce").dropna()
    X = sm.add_constant(X)
    rows = []
    for i, col in enumerate(X.columns):
        if col == "const":
            continue
        try:
            rows.append({"Değişken": col, "VIF": float(variance_inflation_factor(X.values, i))})
        except Exception:
            rows.append({"Değişken": col, "VIF": np.nan})
    return pd.DataFrame(rows)


def predict(res: dict, values: dict) -> float:
    """Verilen öngörücü değerleri için tahmin (yalnızca sayısal terimler)."""
    params = res["coef_params"]
    yhat = float(params.get("Intercept", 0.0))
    for name, coef in params.items():
        if name == "Intercept":
            continue
        var = _pretty_term(name)
        if var in values:
            yhat += float(coef) * float(values[var])
    return yhat


def interpret_regression(res: dict) -> str:
    if "error" in res:
        return res["error"]
    a = res["alpha"]
    lines = []
    lines.append(f"### Model özeti — bağımlı değişken: **{res['y']}**")
    lines.append(f"Denklem türü: {len(res['x_vars'])} öngörücülü OLS regresyon (n={res['n']}).")

    # Model bütünsel anlamlılığı
    lines.append("\n**1) Modelin genel anlamlılığı (F-testi)**")
    lines.append(I.bullet(f"F({res['df_model']}, {res['df_resid']}) = {res['f_stat']:.3f}, p = {res['f_pvalue']:.4g}"))
    if res["f_pvalue"] < a:
        lines.append(I.bullet(
            f"Model bir bütün olarak **anlamlıdır** (p < {a:g}): en az bir öngörücü {res['y']} "
            "değişkenini anlamlı biçimde açıklıyor. H₀ (tüm eğim katsayıları = 0) reddedilir."))
    else:
        lines.append(I.bullet(
            f"Model bir bütün olarak **anlamlı değil** (p ≥ {a:g}); öngörücüler topluca {res['y']} "
            "değişkenini açıklamıyor."))

    # Açıklayıcılık
    lines.append("\n**2) Açıklayıcı güç (R²)**")
    lines.append(I.bullet(
        f"R² = {res['r2']:.4f} → bağımlı değişkendeki değişkenliğin **%{res['r2']*100:.1f}'i** model "
        f"tarafından açıklanıyor. Düzeltilmiş R² = {res['adj_r2']:.4f}. RMSE = {res['rmse']:.4f}."))
    lines.append(I.bullet(f"Açıklayıcılık düzeyi: {I.r2_quality(res['r2'])}."))
    lines.append(I.bullet(f"Model seçimi ölçütleri — AIC = {res['aic']:.1f}, BIC = {res['bic']:.1f} (küçük olan daha iyi)."))

    # Katsayılar
    lines.append("\n**3) Katsayıların anlamlılığı (t-testleri)**")
    for _, row in res["coef_df"].iterrows():
        term = row["Terim"]
        b = row["Katsayı (β)"]
        p = row["p"]
        if term.lower() == "intercept":
            lines.append(I.bullet(
                f"**Sabit (Intercept)** = {b:.4f} — tüm öngörücüler 0 iken {res['y']} beklenen değeri. "
                f"{I.p_label(p, a)} (p={p:.4g})."))
        else:
            verb = "arttıkça" if b > 0 else "arttıkça"
            eff = "artar" if b > 0 else "azalır"
            sig = ("**anlamlı** katkı sağlıyor" if p < a else "anlamlı katkı **sağlamıyor**")
            lines.append(I.bullet(
                f"**{term}**: β = {b:.4f} {row['Anlam']} (p = {p:.4g}). Diğer değişkenler sabitken bu "
                f"öngörücü 1 birim {verb} {res['y']} ortalama {abs(b):.4f} birim {eff}. "
                f"Bu öngörücü modele {sig}."))

    # Tanı testleri
    d = res["diag"]
    lines.append("\n**4) Varsayım/tanı kontrolleri**")
    dw = d.get("durbin_watson")
    if dw is not None:
        dw_txt = ("otokorelasyon yok (ideal ~2)" if 1.5 <= dw <= 2.5 else
                  "pozitif otokorelasyon şüphesi" if dw < 1.5 else "negatif otokorelasyon şüphesi")
        lines.append(I.bullet(f"Durbin-Watson = {dw:.3f} → {dw_txt}."))
    jbp = d.get("jarque_bera_p")
    if jbp == jbp:
        lines.append(I.bullet(
            f"Artık normalliği (Jarque-Bera) p = {jbp:.4g} → "
            + ("artıklar normal dağılım varsayımını karşılıyor." if jbp >= a
               else "artıklar normal dağılımdan sapıyor (küçük örneklemde çıkarımları etkileyebilir).")))
    bpp = d.get("breusch_pagan_p")
    if bpp == bpp:
        lines.append(I.bullet(
            f"Değişen varyans (Breusch-Pagan) p = {bpp:.4g} → "
            + ("sabit varyans (homoskedasti) varsayımı korunuyor." if bpp >= a
               else "**değişen varyans (heteroskedasti) var**; robust standart hatalar düşünülmeli.")))
    if not res["vif_df"].empty:
        maxvif = res["vif_df"]["VIF"].max()
        vtxt = ("çoklu bağıntı sorunu yok" if maxvif < 5 else
                "orta düzey çoklu bağıntı (5–10)" if maxvif < 10 else "**ciddi çoklu bağıntı (>10)**")
        lines.append(I.bullet(f"En yüksek VIF = {maxvif:.2f} → {vtxt}."))
    if res["cond_no"] > 1000:
        lines.append(I.bullet(f"Koşul sayısı = {res['cond_no']:.0f} yüksek; çoklu bağıntı veya ölçek sorunu olabilir."))

    return I.joinlines(lines)
