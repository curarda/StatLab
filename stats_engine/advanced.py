"""İleri modeller: Karışık etkiler (MixedLM), sıfır-şişirilmiş sayım, faktör analizi (EFA)."""
from __future__ import annotations

import warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

from . import interpret as I


# ---------------- Karışık etkiler (multilevel) ----------------

def fit_mixedlm(df: pd.DataFrame, y: str, x_vars: list[str], group: str,
                random_slope: str | None = None, alpha: float = 0.05) -> dict:
    """Rassal kesişimli (ve isteğe bağlı rassal eğimli) karışık etki modeli."""
    cols = [y] + x_vars + [group]
    data = df[cols].copy()
    for v in [y] + x_vars:
        data[v] = pd.to_numeric(data[v], errors="coerce")
    data = data.dropna()
    formula = f"Q('{y}') ~ " + " + ".join(f"Q('{v}')" for v in x_vars)
    kwargs = {"groups": data[group]}
    if random_slope:
        kwargs["re_formula"] = f"~Q('{random_slope}')"
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = smf.mixedlm(formula, data, **kwargs).fit()
    except Exception as e:
        return {"error": f"Karışık model uyduramadı: {e}"}
    fe = pd.DataFrame({"Terim": [_pretty(i) for i in model.fe_params.index],
                       "Katsayı (β)": model.fe_params.values,
                       "Std. Hata": model.bse_fe.values,
                       "z": (model.fe_params / model.bse_fe).values,
                       "p": model.pvalues[model.fe_params.index].values})
    fe["Anlam"] = fe["p"].map(lambda p: I.stars(p) or "—")
    group_var = float(model.cov_re.iloc[0, 0]) if model.cov_re.size else np.nan
    resid_var = float(model.scale)
    icc = group_var / (group_var + resid_var) if (group_var + resid_var) else np.nan
    return {"y": y, "x_vars": x_vars, "group": group, "n": int(model.nobs),
            "n_groups": int(data[group].nunique()), "fe": fe, "group_var": group_var,
            "resid_var": resid_var, "icc": float(icc), "alpha": alpha,
            "summary_text": model.summary().as_text()}


def interpret_mixedlm(res: dict) -> str:
    if "error" in res:
        return res["error"]
    a = res["alpha"]
    lines = [f"### Karışık etki modeli — {res['y']} ~ {', '.join(res['x_vars'])} + (1|{res['group']})"]
    lines.append(I.bullet(f"{res['n']} gözlem, {res['n_groups']} grup ({res['group']})."))
    lines.append(I.bullet(f"**ICC = {res['icc']:.3f}** → toplam değişkenliğin %{res['icc']*100:.1f}'i gruplar arası. "
                          + ("Gruplama önemli, çok düzeyli model gerekli." if res['icc'] > 0.05
                             else "Gruplama etkisi küçük.")))
    lines.append("\n**Sabit etkiler (fixed effects)**")
    for _, r in res["fe"].iterrows():
        if r["Terim"].lower() in ("intercept", "group var"):
            continue
        sig = "**anlamlı**" if r["p"] < a else "anlamlı değil"
        lines.append(I.bullet(f"{r['Terim']}: β = {r['Katsayı (β)']:.4f} {r['Anlam']} (p={r['p']:.4g}) → {sig}."))
    lines.append(I.bullet(f"Grup (rassal kesişim) varyansı = {res['group_var']:.3f}, artık varyans = {res['resid_var']:.3f}."))
    return I.joinlines(lines)


# ---------------- Sıfır-şişirilmiş sayım ----------------

def fit_zero_inflated(df: pd.DataFrame, y: str, x_vars: list[str], family: str = "poisson",
                      alpha: float = 0.05) -> dict:
    """Çok sıfırlı sayım verisi için ZIP / ZINB."""
    from statsmodels.discrete.count_model import ZeroInflatedPoisson, ZeroInflatedNegativeBinomialP
    data = df[[y] + x_vars].apply(pd.to_numeric, errors="coerce").dropna()
    if (data[y] < 0).any():
        return {"error": "Sayım verisi negatif olamaz."}
    yv = data[y].values
    X = sm.add_constant(data[x_vars])
    Model = ZeroInflatedNegativeBinomialP if family == "negbin" else ZeroInflatedPoisson
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = Model(yv, X, exog_infl=np.ones((len(yv), 1))).fit(disp=False, maxiter=200)
    except Exception as e:
        return {"error": f"Sıfır-şişirilmiş model uyduramadı: {e}"}
    zero_obs = float(np.mean(yv == 0))
    params = model.params
    rows = []
    for name in X.columns:
        key = name
        if key in params.index:
            b = float(params[key])
            rows.append({"Terim": key, "Katsayı (β)": b, "IRR": float(np.exp(b)),
                         "p": float(model.pvalues.get(key, np.nan)),
                         "Anlam": I.stars(float(model.pvalues.get(key, np.nan))) or "—"})
    return {"y": y, "x_vars": x_vars, "family": family, "n": int(model.nobs),
            "zero_obs": zero_obs, "aic": float(model.aic), "coef_df": pd.DataFrame(rows),
            "alpha": alpha, "summary_text": str(model.summary())}


def interpret_zero_inflated(res: dict) -> str:
    if "error" in res:
        return res["error"]
    a = res["alpha"]
    fam = "ZIP (Poisson)" if res["family"] == "poisson" else "ZINB (Negatif Binom)"
    lines = [f"### Sıfır-şişirilmiş {fam} — {res['y']}"]
    lines.append(I.bullet(f"n = {res['n']}, gözlemlerin **%{res['zero_obs']*100:.1f}'i sıfır**, AIC = {res['aic']:.1f}."))
    lines.append(I.bullet("Model iki süreci ayırır: (1) 'yapısal sıfır' olma olasılığı, (2) sayım süreci. "
                          "Fazla sıfır içeren verilerde düz Poisson'dan daha uygundur."))
    lines.append("\n**Sayım kısmı katsayıları (IRR)**")
    for _, r in res["coef_df"].iterrows():
        if r["Terim"] == "const":
            continue
        sig = "**anlamlı**" if r["p"] < a else "anlamlı değil"
        lines.append(I.bullet(f"{r['Terim']}: IRR = {r['IRR']:.3f} {r['Anlam']} (p={r['p']:.4g}) → {sig}."))
    return I.joinlines(lines)


# ---------------- Faktör analizi (EFA) ----------------

def _kmo(R: np.ndarray) -> float:
    """Kaiser-Meyer-Olkin örnekleme yeterliliği (korelasyon matrisinden)."""
    try:
        R_inv = np.linalg.inv(R)
    except np.linalg.LinAlgError:
        R_inv = np.linalg.pinv(R)
    d = np.sqrt(np.diag(R_inv))
    P = -R_inv / np.outer(d, d)  # kısmi korelasyon
    R2 = R.copy(); np.fill_diagonal(R2, 0.0)
    P2 = P.copy(); np.fill_diagonal(P2, 0.0)
    sr, sp = np.sum(R2 ** 2), np.sum(P2 ** 2)
    return float(sr / (sr + sp)) if (sr + sp) else np.nan


def _bartlett_sphericity(R: np.ndarray, n: int) -> float:
    from scipy import stats
    p = R.shape[0]
    det = np.linalg.det(R)
    det = max(det, 1e-12)
    chi2 = -((n - 1) - (2 * p + 5) / 6) * np.log(det)
    dfree = p * (p - 1) / 2
    return float(1 - stats.chi2.cdf(chi2, dfree))


def factor_analysis(df: pd.DataFrame, items: list[str], n_factors: int = 2,
                    rotation: str = "varimax") -> dict:
    from sklearn.decomposition import FactorAnalysis
    from sklearn.preprocessing import StandardScaler
    data = df[items].apply(pd.to_numeric, errors="coerce").dropna()
    if data.shape[1] < 3:
        return {"error": "Faktör analizi için en az 3 madde gerekir."}
    Xs = StandardScaler().fit_transform(data.values)
    R = np.corrcoef(Xs, rowvar=False)
    kmo = _kmo(R)
    bart_p = _bartlett_sphericity(R, len(data))
    rot = rotation if rotation in ("varimax", "quartimax") else None
    fa = FactorAnalysis(n_components=n_factors, rotation=rot, random_state=0).fit(Xs)
    loadings = fa.components_.T  # p x n_factors
    load = pd.DataFrame(loadings, index=items, columns=[f"F{i+1}" for i in range(n_factors)])
    comm = pd.Series(np.sum(loadings ** 2, axis=1), index=items, name="Ortak varyans (h²)")
    ev = np.sort(np.linalg.eigvalsh(R))[::-1]
    prop = np.sum(loadings ** 2, axis=0) / len(items)
    var_df = pd.DataFrame({"Faktör": [f"F{i+1}" for i in range(n_factors)],
                           "Açıklanan Varyans %": prop * 100,
                           "Kümülatif %": np.cumsum(prop) * 100})
    return {"loadings": load, "communalities": comm, "kmo": float(kmo),
            "bartlett_p": float(bart_p), "eigenvalues": ev, "var_df": var_df,
            "n_factors": n_factors, "items": items, "n": int(len(data))}


def interpret_factor(res: dict) -> str:
    if "error" in res:
        return res["error"]
    lines = ["### Açımlayıcı Faktör Analizi (EFA)"]
    kmo = res["kmo"]
    kq = ("kabul edilemez (<0.5)" if kmo < 0.5 else "zayıf" if kmo < 0.6 else "orta" if kmo < 0.7
          else "iyi" if kmo < 0.8 else "çok iyi")
    lines.append(I.bullet(f"**KMO = {kmo:.3f}** ({kq}) — veri faktör analizine {'uygun' if kmo>=0.6 else 'pek uygun değil'}."))
    lines.append(I.bullet(f"Bartlett küresellik testi p = {res['bartlett_p']:.4g} → "
                          + ("değişkenler ilişkili, faktörleştirilebilir." if res['bartlett_p'] < 0.05
                             else "**değişkenler ilişkisiz; faktör analizi uygun olmayabilir.**")))
    n_ev = int(np.sum(np.asarray(res["eigenvalues"]) > 1))
    lines.append(I.bullet(f"Kaiser ölçütü (özdeğer > 1): **{n_ev} faktör** öneriliyor. "
                          f"Seçilen {res['n_factors']} faktör toplam varyansın "
                          f"%{res['var_df']['Kümülatif %'].iloc[-1]:.1f}'ini açıklıyor."))
    lines.append(I.bullet("Aşağıdaki yük tablosunda her maddenin en yüksek yüke sahip olduğu faktör, "
                          "o maddenin ait olduğu boyutu gösterir (genelde |yük| > 0.40 anlamlı sayılır)."))
    low = res["communalities"][res["communalities"] < 0.3]
    if not low.empty:
        lines.append(I.bullet("Ortak varyansı (h²) düşük maddeler (<0.30): " + ", ".join(low.index) +
                              " → faktör yapısına zayıf katkı."))
    return I.joinlines(lines)


def _pretty(name: str) -> str:
    return (str(name).replace("Q('", "").replace("')", "").replace("C(", "")
            .replace(")", "").replace("[T.", " = ").replace("]", ""))
