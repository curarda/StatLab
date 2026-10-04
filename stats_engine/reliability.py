"""Güvenilirlik (Cronbach α) ve çoklu karşılaştırma düzeltmesi (Bonferroni/FDR)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests

from . import interpret as I


def cronbach_alpha(df: pd.DataFrame, items: list[str]) -> dict:
    """Ölçek maddeleri için Cronbach α + madde analizi."""
    data = df[items].apply(pd.to_numeric, errors="coerce").dropna()
    if data.shape[1] < 2:
        return {"error": "Cronbach α için en az 2 madde gerekir."}
    if data.shape[0] < 3:
        return {"error": "Yeterli gözlem yok."}
    k = data.shape[1]
    item_var = data.var(axis=0, ddof=1)
    total = data.sum(axis=1)
    total_var = total.var(ddof=1)
    alpha = (k / (k - 1)) * (1 - item_var.sum() / total_var) if total_var else np.nan

    # Madde-toplam korelasyonu (düzeltilmiş) ve madde silinirse α
    rows = []
    for it in items:
        rest = data.drop(columns=[it]).sum(axis=1)
        corr = data[it].corr(rest)
        sub = data.drop(columns=[it])
        k2 = sub.shape[1]
        if k2 >= 2:
            tv = sub.sum(axis=1).var(ddof=1)
            a_del = (k2 / (k2 - 1)) * (1 - sub.var(ddof=1).sum() / tv) if tv else np.nan
        else:
            a_del = np.nan
        rows.append({"Madde": it, "Madde-Toplam r": float(corr), "α (madde silinirse)": float(a_del)})
    return {"alpha": float(alpha), "k": k, "n": int(data.shape[0]),
            "item_stats": pd.DataFrame(rows)}


def interpret_cronbach(res: dict) -> str:
    if "error" in res:
        return res["error"]
    a = res["alpha"]
    q = ("kabul edilemez" if a < 0.5 else "zayıf" if a < 0.6 else "sorgulanabilir" if a < 0.7
         else "kabul edilebilir" if a < 0.8 else "iyi" if a < 0.9 else "mükemmel (maddeler fazla benzer olabilir)")
    lines = [f"### Cronbach α güvenilirlik — {res['k']} madde, n={res['n']}"]
    lines.append(I.bullet(f"**Cronbach α = {a:.3f}** → iç tutarlılık **{q}**."))
    lines.append(I.bullet("Genel eşik: α ≥ 0.70 kabul edilebilir güvenilirlik."))
    low = res["item_stats"][res["item_stats"]["Madde-Toplam r"] < 0.30]
    if not low.empty:
        lines.append(I.bullet("Madde-toplam korelasyonu < 0.30 olan maddeler ölçeğe zayıf katkı sağlıyor: "
                              + ", ".join(low["Madde"]) + ". Çıkarılması düşünülebilir."))
    better = res["item_stats"][res["item_stats"]["α (madde silinirse)"] > a + 1e-9]
    if not better.empty:
        lines.append(I.bullet("Şu madde(ler) silinirse α **artar**: "
                              + ", ".join(better["Madde"]) + " → gözden geçirin."))
    else:
        lines.append(I.bullet("Hiçbir maddenin silinmesi α'yı artırmıyor; ölçek dengeli."))
    return I.joinlines(lines)


def correct_pvalues(pvalues: list[float], labels: list[str] | None = None,
                    method: str = "fdr_bh", alpha: float = 0.05) -> dict:
    """Çoklu karşılaştırma düzeltmesi (Bonferroni / Holm / FDR-BH)."""
    p = np.asarray([x for x in pvalues if x is not None and x == x], float)
    if len(p) == 0:
        return {"error": "Geçerli p-değeri yok."}
    reject, p_adj, _, _ = multipletests(p, alpha=alpha, method=method)
    labels = labels or [f"Test {i+1}" for i in range(len(p))]
    table = pd.DataFrame({"Karşılaştırma": labels[:len(p)], "p (ham)": p,
                          "p (düzeltilmiş)": p_adj,
                          "Anlamlı?": ["Evet" if r else "Hayır" for r in reject]})
    return {"table": table, "method": method, "n_tests": len(p),
            "n_sig_raw": int(np.sum(p < alpha)), "n_sig_adj": int(np.sum(reject)), "alpha": alpha}


def interpret_correction(res: dict) -> str:
    if "error" in res:
        return res["error"]
    names = {"bonferroni": "Bonferroni", "holm": "Holm", "fdr_bh": "Benjamini-Hochberg (FDR)"}
    lines = [f"### Çoklu karşılaştırma düzeltmesi — {names.get(res['method'], res['method'])}"]
    lines.append(I.bullet(f"{res['n_tests']} test yapıldı. Düzeltme öncesi anlamlı: {res['n_sig_raw']}, "
                          f"düzeltme sonrası anlamlı: **{res['n_sig_adj']}**."))
    if res["n_sig_raw"] > res["n_sig_adj"]:
        lines.append(I.bullet(f"{res['n_sig_raw'] - res['n_sig_adj']} sonuç düzeltmeden sonra anlamlılığını yitirdi — "
                              "bunlar muhtemelen **çoklu test kaynaklı yanlış pozitiflerdi.**"))
    lines.append(I.bullet("Çok sayıda test yapıldığında I. tip hata şişer; Bonferroni katıdır, "
                          "FDR (BH) daha az muhafazakâr ve keşifsel çalışmalar için tercih edilir."))
    return I.joinlines(lines)
