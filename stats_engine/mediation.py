"""Aracılık (mediation) ve düzenleyicilik (moderation) analizi.

Aracılık: X → M → Y (dolaylı yol var mı?)
Düzenleyicilik: X'in Y üzerindeki etkisi W'ye göre değişiyor mu? (etkileşim)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import interpret as I


def mediation(df: pd.DataFrame, x: str, m: str, y: str, alpha: float = 0.05, n_boot: int = 2000) -> dict:
    """Basit aracılık analizi (Baron-Kenny + bootstrap dolaylı etki)."""
    import pingouin as pg
    data = df[[x, m, y]].apply(pd.to_numeric, errors="coerce").dropna()
    if len(data) < 10:
        return {"error": "Aracılık analizi için yeterli veri yok."}
    try:
        res = pg.mediation_analysis(data=data, x=x, m=m, y=y, alpha=alpha, n_boot=n_boot, seed=42)
    except Exception as e:
        return {"error": f"Aracılık hesaplanamadı: {e}"}
    # pingouin satırları: 'Total', 'Direct', 'Indirect', ('M ~ X', 'Y ~ M')
    def _row(name):
        r = res[res["path"] == name]
        return r.iloc[0] if len(r) else None
    total = _row("Total"); direct = _row("Direct"); indirect = _row("Indirect")
    out = {"x": x, "m": m, "y": y, "table": res, "alpha": alpha}
    if total is not None:
        out["total"] = float(total["coef"]); out["total_p"] = float(total["pval"])
    if direct is not None:
        out["direct"] = float(direct["coef"]); out["direct_p"] = float(direct["pval"])
    if indirect is not None:
        out["indirect"] = float(indirect["coef"]); out["indirect_p"] = float(indirect["pval"])
        out["indirect_sig"] = float(indirect["pval"]) < alpha
        if out.get("total"):
            out["prop_mediated"] = out["indirect"] / out["total"] if out["total"] else np.nan
    return out


def moderation(df: pd.DataFrame, x: str, w: str, y: str, alpha: float = 0.05) -> dict:
    """Düzenleyicilik: Y ~ X + W + X:W. Etkileşim anlamlıysa W, X→Y etkisini düzenler."""
    import statsmodels.formula.api as smf
    data = df[[x, w, y]].apply(pd.to_numeric, errors="coerce").dropna()
    if len(data) < 10:
        return {"error": "Yeterli veri yok."}
    # merkezleme (çoklu bağıntıyı azaltır, yorumu netleştirir)
    data["_x"] = data[x] - data[x].mean()
    data["_w"] = data[w] - data[w].mean()
    model = smf.ols(f"Q('{y}') ~ _x * _w", data).fit()
    inter = model.params.get("_x:_w", np.nan)
    inter_p = model.pvalues.get("_x:_w", np.nan)
    # basit eğimler: düşük/orta/yüksek W (ort ± 1 SD)
    sd_w = data[w].std()
    b_x = model.params.get("_x", np.nan)
    slopes = {"W düşük (−1SD)": b_x + inter * (-sd_w),
              "W ortalama": b_x,
              "W yüksek (+1SD)": b_x + inter * (sd_w)}
    return {"x": x, "w": w, "y": y, "inter": float(inter), "inter_p": float(inter_p),
            "b_x": float(b_x), "slopes": {k: float(v) for k, v in slopes.items()},
            "r2": float(model.rsquared), "alpha": alpha, "n": int(model.nobs)}


def interpret_mediation(res: dict) -> str:
    if "error" in res:
        return res["error"]
    a = res["alpha"]
    lines = [f"### Aracılık analizi — {res['x']} → {res['m']} → {res['y']}"]
    if "total" in res:
        lines.append(I.bullet(f"Toplam etki (X→Y): {res['total']:.4f} (p={res['total_p']:.4g})"))
    if "direct" in res:
        lines.append(I.bullet(f"Doğrudan etki (X→Y, M kontrollü): {res['direct']:.4f} (p={res['direct_p']:.4g})"))
    if "indirect" in res:
        lines.append(I.bullet(f"**Dolaylı etki (X→M→Y): {res['indirect']:.4f} (p={res['indirect_p']:.4g})** "
                              f"{'✅ anlamlı' if res.get('indirect_sig') else 'anlamlı değil'}"))
    if res.get("indirect_sig"):
        pm = res.get("prop_mediated")
        dir_sig = res.get("direct_p", 1) < a
        typ = "kısmi aracılık" if dir_sig else "tam aracılık"
        lines.append(I.bullet(f"**{typ.capitalize()} var:** {res['x']}'in {res['y']} üzerindeki etkisinin "
                              + (f"~%{pm*100:.0f}'i " if pm == pm else "") +
                              f"{res['m']} üzerinden geçiyor. "
                              + ("Doğrudan etki hâlâ anlamlı (kısmi)." if dir_sig
                                 else "Doğrudan etki anlamsız → etki tamamen aracı üzerinden (tam).")))
    else:
        lines.append(I.bullet("Dolaylı yol anlamlı değil → M bir aracı olarak işlev görmüyor."))
    lines.append(I.bullet("Aracılık, 'neden/nasıl' sorusunu yanıtlar: mekanizmayı gösterir. Bootstrap güven aralığı kullanılır."))
    return I.joinlines(lines)


def interpret_moderation(res: dict) -> str:
    if "error" in res:
        return res["error"]
    a = res["alpha"]
    lines = [f"### Düzenleyicilik — {res['w']}, {res['x']}→{res['y']} etkisini düzenliyor mu?"]
    lines.append(I.bullet(f"Etkileşim (X×W): β = {res['inter']:.4f} (p={res['inter_p']:.4g}) {I.stars(res['inter_p'])}, "
                          f"model R²={res['r2']:.3f}."))
    if res["inter_p"] < a:
        lines.append(I.bullet(f"**Anlamlı düzenleyicilik var:** {res['x']}'in {res['y']} üzerindeki etkisi "
                              f"{res['w']} düzeyine göre **değişiyor**."))
        lines.append("**Basit eğimler (X→Y):**")
        for k, v in res["slopes"].items():
            lines.append(I.bullet(f"{k}: eğim = {v:.4f}"))
        lines.append(I.bullet("Yani etkiyi tek bir sayıyla özetlemek yanıltıcı; W'ye göre koşullu yorumlayın."))
    else:
        lines.append(I.bullet(f"Etkileşim anlamlı değil → {res['x']}'in etkisi {res['w']}'den bağımsız (düzenleyicilik yok)."))
    return I.joinlines(lines)
