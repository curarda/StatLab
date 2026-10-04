"""Regresyon tanıları+ ve alternatif regresyonlar: etki, robust, kantil, değişken seçimi."""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import OLSInfluence

from . import interpret as I


def _fit_ols(df, y, x_vars):
    data = df[[y] + x_vars].apply(pd.to_numeric, errors="coerce").dropna()
    X = sm.add_constant(data[x_vars])
    model = sm.OLS(data[y], X).fit()
    return model, data


def influence(df: pd.DataFrame, y: str, x_vars: list[str], alpha: float = 0.05) -> dict:
    """Etkili gözlemler: Cook uzaklığı, kaldıraç (leverage), standart artıklar."""
    try:
        model, data = _fit_ols(df, y, x_vars)
    except Exception as e:
        return {"error": f"Model uyduramadı: {e}"}
    infl = OLSInfluence(model)
    n, k = int(model.nobs), model.df_model + 1
    cook = infl.cooks_distance[0]
    lev = infl.hat_matrix_diag
    stud = infl.resid_studentized_internal
    cook_thr = 4 / n
    lev_thr = 2 * k / n
    tbl = pd.DataFrame({"Gözlem": data.index, "Cook D": cook, "Kaldıraç": lev,
                        "Std. Artık": stud})
    tbl["Etkili?"] = ((tbl["Cook D"] > cook_thr) | (tbl["Kaldıraç"] > lev_thr) |
                      (tbl["Std. Artık"].abs() > 3)).map({True: "⚠️", False: ""})
    flagged = tbl[tbl["Etkili?"] == "⚠️"].sort_values("Cook D", ascending=False)
    return {"table": tbl, "flagged": flagged, "cook_thr": cook_thr, "lev_thr": lev_thr,
            "n": n, "fitted": model.fittedvalues.values, "y": y, "x_vars": x_vars}


def interpret_influence(res: dict) -> str:
    if "error" in res:
        return res["error"]
    lines = ["### Etkili gözlem tanıları"]
    lines.append(I.bullet(f"Eşikler — Cook D > {res['cook_thr']:.4f} (4/n), kaldıraç > {res['lev_thr']:.4f} (2k/n), "
                          "|std. artık| > 3."))
    nf = len(res["flagged"])
    if nf == 0:
        lines.append(I.bullet("Eşiği aşan **etkili/aykırı gözlem yok**; model bireysel noktalara duyarlı değil."))
    else:
        lines.append(I.bullet(f"**{nf} etkili gözlem** işaretlendi (en yüksek Cook D olanlar tabloda ⚠️). "
                              "Bunlar katsayıları orantısız etkileyebilir; veri hatası mı yoksa gerçek uç değer mi "
                              "kontrol edin, dışlayıp modeli tekrar deneyin."))
    return I.joinlines(lines)


def robust_se(df: pd.DataFrame, y: str, x_vars: list[str], cov_type: str = "HC3", alpha: float = 0.05) -> dict:
    """OLS katsayıları + değişen-varyansa dayanıklı (heteroskedasti-tutarlı) standart hatalar."""
    try:
        data = df[[y] + x_vars].apply(pd.to_numeric, errors="coerce").dropna()
        X = sm.add_constant(data[x_vars])
        ols = sm.OLS(data[y], X).fit()
        rob = sm.OLS(data[y], X).fit(cov_type=cov_type)
    except Exception as e:
        return {"error": f"Hesaplanamadı: {e}"}
    rows = []
    for name in ols.params.index:
        rows.append({"Terim": name, "Katsayı (β)": float(ols.params[name]),
                     "OLS std. hata": float(ols.bse[name]), "OLS p": float(ols.pvalues[name]),
                     f"{cov_type} std. hata": float(rob.bse[name]), f"{cov_type} p": float(rob.pvalues[name])})
    df_out = pd.DataFrame(rows)
    # anlamlılığı değişen terimler
    changed = df_out[(df_out["OLS p"] < alpha) != (df_out[f"{cov_type} p"] < alpha)]
    return {"table": df_out, "cov_type": cov_type, "y": y, "changed": changed, "alpha": alpha}


def interpret_robust_se(res: dict) -> str:
    if "error" in res:
        return res["error"]
    ct = res["cov_type"]
    lines = [f"### Değişen varyansa dayanıklı standart hatalar ({ct})"]
    lines.append(I.bullet("Katsayılar aynı kalır; yalnızca standart hatalar/p-değerleri düzeltilir. "
                          "Heteroskedasti varsa OLS p-değerleri yanıltıcıdır — bunlar doğru çıkarımı verir."))
    if res["changed"].empty:
        lines.append(I.bullet("İyi haber: hiçbir terimin **anlamlılık kararı değişmedi** — sonuçlar sağlam."))
    else:
        terms = ", ".join(res["changed"]["Terim"])
        lines.append(I.bullet(f"⚠️ Şu terim(ler)in anlamlılığı düzeltmeyle **değişti**: {terms}. "
                              "Robust p-değerlerini esas alın (OLS heteroskedasti nedeniyle yanıltmış)."))
    return I.joinlines(lines)


def robust_regression(df: pd.DataFrame, y: str, x_vars: list[str], alpha: float = 0.05) -> dict:
    """Aykırı değerlere dayanıklı regresyon (Huber M-tahmincisi)."""
    data = df[[y] + x_vars].apply(pd.to_numeric, errors="coerce").dropna()
    X = sm.add_constant(data[x_vars])
    try:
        model = sm.RLM(data[y], X, M=sm.robust.norms.HuberT()).fit()
    except Exception as e:
        return {"error": f"Robust model uyduramadı: {e}"}
    coef = _coef_table(model, alpha)
    # OLS ile karşılaştırma
    ols = sm.OLS(data[y], X).fit()
    return {"type": "Robust (Huber)", "coef_df": coef, "ols_params": ols.params,
            "y": y, "x_vars": x_vars, "alpha": alpha}


def quantile_regression(df: pd.DataFrame, y: str, x_vars: list[str], q: float = 0.5,
                        alpha: float = 0.05) -> dict:
    """Kantil regresyon (varsayılan medyan). q ile farklı kantiller."""
    data = df[[y] + x_vars].apply(pd.to_numeric, errors="coerce").dropna()
    formula = f"Q('{y}') ~ " + " + ".join(f"Q('{v}')" for v in x_vars)
    try:
        model = smf.quantreg(formula, data).fit(q=q)
    except Exception as e:
        return {"error": f"Kantil model uyduramadı: {e}"}
    coef = _coef_table(model, alpha, pretty=True)
    return {"type": f"Kantil regresyon (q={q})", "coef_df": coef, "q": q,
            "pseudo_r2": float(model.prsquared), "y": y, "x_vars": x_vars, "alpha": alpha}


def stepwise(df: pd.DataFrame, y: str, x_vars: list[str], alpha_in: float = 0.05,
             alpha_out: float = 0.10) -> dict:
    """İleri-geri adımsal değişken seçimi (p-değeri ölçütlü)."""
    data = df[[y] + x_vars].apply(pd.to_numeric, errors="coerce").dropna()
    included = []
    steps = []
    remaining = list(x_vars)
    while True:
        changed = False
        # ileri adım
        best_p, best_v = 1.0, None
        for v in remaining:
            X = sm.add_constant(data[included + [v]])
            try:
                m = sm.OLS(data[y], X).fit()
                pval = m.pvalues[v]
            except Exception:
                continue
            if pval < best_p:
                best_p, best_v = pval, v
        if best_v is not None and best_p < alpha_in:
            included.append(best_v); remaining.remove(best_v)
            steps.append(f"+ {best_v} eklendi (p={best_p:.4g})")
            changed = True
        # geri adım
        if included:
            X = sm.add_constant(data[included])
            m = sm.OLS(data[y], X).fit()
            pvals = m.pvalues.drop("const", errors="ignore")
            worst_p = pvals.max()
            if worst_p > alpha_out:
                worst_v = pvals.idxmax()
                included.remove(worst_v); remaining.append(worst_v)
                steps.append(f"− {worst_v} çıkarıldı (p={worst_p:.4g})")
                changed = True
        if not changed:
            break
    if included:
        Xf = sm.add_constant(data[included])
        final = sm.OLS(data[y], Xf).fit()
        r2, adj = float(final.rsquared), float(final.rsquared_adj)
    else:
        r2 = adj = np.nan
    return {"selected": included, "steps": steps, "r2": r2, "adj_r2": adj,
            "y": y, "candidates": x_vars}


def _coef_table(model, alpha, pretty=False) -> pd.DataFrame:
    conf = model.conf_int(alpha=alpha)
    rows = []
    for name in model.params.index:
        term = name
        if pretty:
            term = (str(name).replace("Q('", "").replace("')", ""))
        rows.append({"Terim": term, "Katsayı (β)": float(model.params[name]),
                     "Std. Hata": float(model.bse[name]), "t/z": float(model.tvalues[name]),
                     "p": float(model.pvalues[name]),
                     "Anlam": I.stars(float(model.pvalues[name])) or "—"})
    return pd.DataFrame(rows)


def interpret_robust(res: dict) -> str:
    if "error" in res:
        return res["error"]
    lines = [f"### {res['type']} — {res['y']}"]
    lines.append(I.bullet("Aykırı gözlemlere OLS'ten daha az duyarlıdır. OLS ile karşılaştırın: "
                          "katsayılar çok farklıysa OLS uç değerlerden etkilenmiş demektir."))
    for _, r in res["coef_df"].iterrows():
        if r["Terim"] == "const":
            continue
        ols_b = float(res["ols_params"].get(r["Terim"], np.nan))
        lines.append(I.bullet(f"{r['Terim']}: robust β = {r['Katsayı (β)']:.4f} (OLS β = {ols_b:.4f}) "
                              f"{r['Anlam']} (p={r['p']:.4g})"))
    return I.joinlines(lines)


def interpret_quantile(res: dict) -> str:
    if "error" in res:
        return res["error"]
    lines = [f"### {res['type']} — {res['y']}"]
    lines.append(I.bullet(f"Sözde-R² = {res['pseudo_r2']:.3f}. Kantil regresyon, koşullu ortalama yerine "
                          f"koşullu %{int(res['q']*100)}. kantili modeller — dağılımın farklı bölgelerinde etkiyi gösterir."))
    for _, r in res["coef_df"].iterrows():
        if r["Terim"].lower() == "intercept":
            continue
        sig = "**anlamlı**" if r["p"] < res["alpha"] else "anlamlı değil"
        lines.append(I.bullet(f"{r['Terim']}: β = {r['Katsayı (β)']:.4f} {r['Anlam']} (p={r['p']:.4g}) → {sig}."))
    return I.joinlines(lines)


def interpret_stepwise(res: dict) -> str:
    lines = ["### Adımsal değişken seçimi"]
    if not res["selected"]:
        lines.append(I.bullet("Hiçbir değişken ölçütü karşılamadı; anlamlı öngörücü bulunamadı."))
        return I.joinlines(lines)
    lines.append(I.bullet(f"**Seçilen model:** {res['y']} ~ {' + '.join(res['selected'])}"))
    lines.append(I.bullet(f"R² = {res['r2']:.3f}, düzeltilmiş R² = {res['adj_r2']:.3f}."))
    lines.append(I.bullet(f"{len(res['candidates'])} aday değişkenden {len(res['selected'])} tanesi seçildi."))
    if res["steps"]:
        lines.append("\n**Adımlar:** " + " · ".join(res["steps"]))
    lines.append(I.bullet("Not: Adımsal seçim keşifseldir; p-değerlerini iyimser gösterebilir. "
                          "Nihai modeli teoriyle ve AIC/BIC ile doğrulayın."))
    return I.joinlines(lines)
