"""Birden çok regresyon modelini AIC/BIC/R² ile karşılaştırma."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import regression, interpret as I


def compare_models(df: pd.DataFrame, y: str, model_specs: list[dict], alpha: float = 0.05) -> dict:
    """model_specs: [{'name','x_vars','categorical'}]. Aynı Y için karşılaştırır."""
    rows = []
    details = {}
    for spec in model_specs:
        xv = spec.get("x_vars", [])
        if not xv:
            continue
        res = regression.fit_ols(df, y, xv, categorical=spec.get("categorical", []), alpha=alpha)
        name = spec.get("name") or " + ".join(xv)
        if "error" in res:
            rows.append({"Model": name, "Hata": res["error"]})
            continue
        details[name] = res
        rows.append({
            "Model": name,
            "Öngörücü sayısı": len(xv),
            "R²": res["r2"], "Düz. R²": res["adj_r2"],
            "AIC": res["aic"], "BIC": res["bic"], "RMSE": res["rmse"],
            "F p-değeri": res["f_pvalue"], "n": res["n"],
        })
    table = pd.DataFrame(rows)
    best = {}
    if not table.empty and "AIC" in table.columns:
        valid = table.dropna(subset=["AIC"]) if "AIC" in table else table
        if not valid.empty:
            best["aic"] = valid.loc[valid["AIC"].idxmin(), "Model"]
            best["bic"] = valid.loc[valid["BIC"].idxmin(), "Model"]
            best["adj_r2"] = valid.loc[valid["Düz. R²"].idxmax(), "Model"]
    return {"table": table, "best": best, "details": details, "y": y}


def interpret_comparison(res: dict) -> str:
    b = res["best"]
    if not b:
        return "Karşılaştırılacak geçerli model yok."
    lines = [f"### Model karşılaştırması — bağımlı değişken: **{res['y']}**"]
    lines.append(I.bullet(f"**AIC'e göre en iyi:** {b.get('aic','—')} (en düşük AIC = bilgi kaybı en az)."))
    lines.append(I.bullet(f"**BIC'e göre en iyi:** {b.get('bic','—')} (BIC karmaşıklığı daha sert cezalandırır)."))
    lines.append(I.bullet(f"**Düzeltilmiş R²'ye göre en iyi:** {b.get('adj_r2','—')} (öngörücü sayısına göre düzeltilmiş açıklayıcılık)."))
    if b.get("aic") == b.get("bic") == b.get("adj_r2"):
        lines.append(I.bullet(f"Üç ölçüt de **{b.get('aic')}** modelinde birleşiyor → net tercih."))
    else:
        lines.append(I.bullet("Ölçütler farklı modelleri işaret ediyor: AIC/BIC daha küçük, "
                              "düzeltilmiş R² daha büyük olanı tercih edin; genelde **BIC daha sade** modeli seçer."))
    lines.append(I.bullet("Kural: AIC/BIC farkı < 2 ise modeller pratikte denk sayılır; sade olanı yeğleyin."))
    return I.joinlines(lines)
