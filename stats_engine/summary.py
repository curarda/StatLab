"""Yönetici özeti / anlatı üreteci — oturumdaki analizleri düz-dil hikayeye çevirir.

Yalnızca YORUM katmanıdır: ham sayıları ilgili sekmelerde bırakır; burada
'ne bulundu / ne anlama geliyor / ne yapmalı' anlatısını üretir.
"""
from __future__ import annotations

import pandas as pd

from . import interpret as I, insights as INS


def build_narrative(df: pd.DataFrame, session: dict, alpha: float = 0.05) -> dict:
    """session: st.session_state'ten toplanan analiz sonuçları sözlüğü."""
    bulgular, anlamlar, oneriler = [], [], []

    # Veri özeti
    prof = INS.profile(df)
    flags = INS.quality_flags(df)
    ins = INS.auto_insights(df, alpha, max_items=6)
    high_flags = [f for f in flags if f["seviye"] == "yüksek"]

    veri_ozet = (f"Veri seti {prof['n_rows']} gözlem ve {prof['n_cols']} değişken içeriyor "
                 f"(ortalama %{prof['total_missing_pct']} eksik).")
    if high_flags:
        oneriler.append(f"Önce {len(high_flags)} kritik veri kalitesi sorununu giderin "
                        f"({', '.join(f['sütun'] for f in high_flags[:3])}).")

    # Otomatik içgörüler
    for it in ins[:4]:
        bulgular.append(it["fact"])
        anlamlar.append(it["yorum"])
        oneriler.append(it["oneri"])

    # Çalıştırılmış analizler
    reg = session.get("reg_res")
    if reg and "error" not in reg:
        sig = reg["coef_df"][(reg["coef_df"]["Terim"].str.lower() != "intercept") & (reg["coef_df"]["p"] < alpha)]
        bulgular.append(f"Regresyon: {reg['y']} modelinde R²={reg['r2']:.2f}, {len(sig)} anlamlı öngörücü.")
        if not sig.empty:
            drivers = ", ".join(sig["Terim"].head(3))
            anlamlar.append(f"{reg['y']} değişkenini en çok {drivers} açıklıyor; model değişkenliğin "
                            f"%{reg['r2']*100:.0f}'ini yakalıyor.")
            oneriler.append(f"{reg['y']} üzerinde etki için öncelikle {sig['Terim'].iloc[0]} üzerine odaklanın.")

    logit = session.get("logit_res")
    if logit and "error" not in logit:
        auc = logit.get("roc", {}).get("auc")
        bulgular.append(f"Lojistik model ({logit['y']}): AUC={auc:.2f}, doğruluk={logit['metrics']['accuracy']:.2f}.")
        anlamlar.append("Sınıflandırma gücü " + ("iyi" if auc and auc > 0.8 else "orta" if auc and auc > 0.7 else "sınırlı") + ".")

    cmp = session.get("cmp_res")
    if cmp and cmp.get("best"):
        anlamlar.append(f"Model karşılaştırmasında AIC'e göre en iyi: {cmp['best'].get('aic')}.")

    ts = session.get("ts_bundle")
    if ts:
        ov = ts.get("ov", {})
        trend = "yükselen" if ov.get("slope", 0) > 0 else "düşen"
        bulgular.append(f"Zaman serisi ({ts['val']}): {trend} trend (eğim {ov.get('slope', 0):.3f}).")
        oneriler.append("Tahmin ufkunu ve mevsimselliği Zaman Serisi sekmesinden izleyin.")

    fin = session.get("fin_opt")
    if fin:
        ms = fin["max_sharpe"]
        bulgular.append(f"Portföy: maks-Sharpe {ms['sharpe']:.2f} (getiri %{ms['ret']*100:.0f}, risk %{ms['vol']*100:.0f}).")
        oneriler.append("Önerilen ağırlıkları Optimizasyon sekmesinden uygulayın.")

    return {"veri_ozet": veri_ozet, "bulgular": bulgular, "anlamlar": anlamlar,
            "oneriler": list(dict.fromkeys(oneriler)), "n_analiz": _count(session)}


def _count(session: dict) -> int:
    keys = ["reg_res", "logit_res", "cmp_res", "ts_bundle", "fin_opt", "clu_res", "cnt_res",
            "anova", "km_res", "cox_res", "fa_res", "mix_res", "pnl_res"]
    return sum(1 for k in keys if session.get(k) and "error" not in (session.get(k) if isinstance(session.get(k), dict) else {}))


def render_narrative(nar: dict) -> str:
    lines = ["## 📝 Yönetici Özeti", "", f"_{nar['veri_ozet']}_", ""]
    lines.append("### 🔍 Ne bulundu?")
    if nar["bulgular"]:
        for b in nar["bulgular"]:
            lines.append(I.bullet(b))
    else:
        lines.append(I.bullet("Henüz belirgin bir bulgu yok — analiz çalıştırın veya otomatik içgörüye bakın."))
    lines.append("")
    lines.append("### 💡 Ne anlama geliyor?")
    if nar["anlamlar"]:
        for a in nar["anlamlar"]:
            lines.append(I.bullet(a))
    else:
        lines.append(I.bullet("Yorum için önce bir analiz çalıştırın."))
    lines.append("")
    lines.append("### ✅ Ne yapmalı? (öneriler)")
    if nar["oneriler"]:
        for o in nar["oneriler"]:
            lines.append(I.bullet(o))
    else:
        lines.append(I.bullet("Belirgin bir aksiyon önerisi çıkmadı."))
    return I.joinlines(lines)
