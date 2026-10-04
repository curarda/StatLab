"""Panel veri analizi: Pooled OLS, Sabit Etkiler (FE), Rassal Etkiler (RE) + Hausman."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import interpret as I


def fit_panel(df: pd.DataFrame, y: str, x_vars: list[str], entity: str, time: str,
              alpha: float = 0.05) -> dict:
    """Aynı Y/X için Pooled, FE (entity) ve RE modellerini uydurur ve karşılaştırır."""
    try:
        from linearmodels.panel import PanelOLS, RandomEffects, PooledOLS
    except ImportError:
        return {"error": "linearmodels kurulu değil. `pip install linearmodels` çalıştırın."}

    cols = [y, entity, time] + [v for v in x_vars if v not in (entity, time)]
    data = df[cols].copy()
    for v in [y] + x_vars:
        data[v] = pd.to_numeric(data[v], errors="coerce")
    data = data.dropna(subset=[y] + x_vars)
    if data.empty:
        return {"error": "Geçerli veri yok."}
    # MultiIndex: (entity, time)
    data = data.set_index([entity, time])
    Y = data[y]
    X = data[x_vars]

    out = {"y": y, "x_vars": x_vars, "entity": entity, "time": time, "alpha": alpha,
           "n": int(len(data)), "n_entities": int(data.index.get_level_values(0).nunique()),
           "n_periods": int(data.index.get_level_values(1).nunique())}
    models = {}
    try:
        models["Pooled OLS"] = PooledOLS(Y, _const(X)).fit()
        models["Sabit Etkiler (FE)"] = PanelOLS(Y, X, entity_effects=True).fit()
        models["Rassal Etkiler (RE)"] = RandomEffects(Y, _const(X)).fit()
    except Exception as e:
        return {"error": f"Panel modeli uyduramadı: {e}"}

    rows = []
    coef_tables = {}
    for name, res in models.items():
        rows.append({"Model": name, "R² (overall)": float(getattr(res, "rsquared_overall", res.rsquared)),
                     "R² (within)": float(getattr(res, "rsquared_within", np.nan)),
                     "Gözlem": int(res.nobs)})
        ct = pd.DataFrame({"Katsayı (β)": res.params, "Std. Hata": res.std_errors,
                           "t": res.tstats, "p": res.pvalues})
        ct["Anlam"] = ct["p"].map(lambda p: I.stars(p) or "—")
        coef_tables[name] = ct.reset_index().rename(columns={"index": "Terim", "parameter": "Terim"})
    out["summary_table"] = pd.DataFrame(rows)
    out["coef_tables"] = coef_tables

    # Hausman testi (FE vs RE)
    out["hausman"] = _hausman(models["Sabit Etkiler (FE)"], models["Rassal Etkiler (RE)"])
    return out


def _const(X):
    X = X.copy()
    X.insert(0, "const", 1.0)
    return X


def _hausman(fe, re) -> dict:
    try:
        common = [c for c in fe.params.index if c in re.params.index and c != "const"]
        b = fe.params[common].values
        B = re.params[common].values
        v_b = fe.cov.loc[common, common].values
        v_B = re.cov.loc[common, common].values
        diff = b - B
        var = v_b - v_B
        stat = float(diff @ np.linalg.pinv(var) @ diff.T)
        from scipy import stats
        dof = len(common)
        p = float(1 - stats.chi2.cdf(stat, dof))
        return {"stat": stat, "dof": dof, "p": p}
    except Exception as e:
        return {"error": str(e)}


def interpret_panel(res: dict) -> str:
    if "error" in res:
        return res["error"]
    a = res["alpha"]
    lines = [f"### Panel veri analizi — **{res['y']}** ~ {', '.join(res['x_vars'])}"]
    lines.append(I.bullet(f"{res['n_entities']} birim × {res['n_periods']} dönem = {res['n']} gözlem "
                          f"(birim: {res['entity']}, zaman: {res['time']})."))
    lines.append("\n**Model karşılaştırması** (aşağıdaki tablo). En yaygın seçim FE veya RE'dir; "
                 "Pooled OLS birim etkilerini yok sayar.")
    h = res.get("hausman", {})
    if "p" in h:
        lines.append("\n**Hausman testi (FE vs RE)**")
        lines.append(I.bullet(f"χ² = {h['stat']:.3f}, sd = {h['dof']}, p = {h['p']:.4g}"))
        if h["p"] < a:
            lines.append(I.bullet("p < α → **Sabit Etkiler (FE)** tercih edilmeli. RE tahmincisi tutarsız; "
                                  "birim etkileri açıklayıcılarla ilişkili."))
        else:
            lines.append(I.bullet("p ≥ α → **Rassal Etkiler (RE)** kullanılabilir (daha etkin); "
                                  "birim etkileri açıklayıcılarla ilişkisiz varsayılabilir."))
    # FE katsayı yorumu
    fe_ct = res["coef_tables"].get("Sabit Etkiler (FE)")
    if fe_ct is not None:
        lines.append("\n**Sabit Etkiler modelinde anlamlı öngörücüler**")
        sig = fe_ct[(fe_ct["Terim"] != "const") & (fe_ct["p"] < a)]
        if sig.empty:
            lines.append(I.bullet("FE modelinde anlamlı öngörücü yok."))
        for _, r in sig.iterrows():
            b = r["Katsayı (β)"]
            lines.append(I.bullet(f"{r['Terim']}: β = {b:.4f} (p={r['p']:.4g}) → birim içi 1 birim değişim "
                                  f"{res['y']} değişkenini {b:+.4f} birim değiştirir."))
    return I.joinlines(lines)
