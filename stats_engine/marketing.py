"""Pazarlama analitiği: dönüşüm hunisi (funnel), kohort/elde tutma (retention), RFM."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import interpret as I


# ---------------- Funnel ----------------

def funnel_from_counts(stages: list[str], counts: list[float]) -> dict:
    counts = [float(c) for c in counts]
    rows = []
    top = counts[0] if counts else 0
    for i, (s, c) in enumerate(zip(stages, counts)):
        step_conv = (c / counts[i - 1]) if i > 0 and counts[i - 1] else np.nan
        overall = (c / top) if top else np.nan
        drop = (1 - step_conv) if i > 0 and step_conv == step_conv else np.nan
        rows.append({"Aşama": s, "Adet": c, "Adım dönüşümü": step_conv,
                     "Kümülatif %": overall, "Kayıp %": drop})
    return {"table": pd.DataFrame(rows), "stages": stages, "counts": counts,
            "overall_conv": (counts[-1] / top) if top else np.nan}


def funnel_from_columns(df: pd.DataFrame, stage_cols: list[str]) -> dict:
    """Her sütun ikili (0/1) bir aşamayı temsil eder; o aşamaya ulaşan sayısı = toplam."""
    counts = [int(pd.to_numeric(df[c], errors="coerce").fillna(0).astype(bool).sum()) for c in stage_cols]
    return funnel_from_counts(stage_cols, counts)


def interpret_funnel(res: dict) -> str:
    t = res["table"]
    lines = ["### Dönüşüm hunisi — yorum katmanı"]
    lines.append(I.bullet(f"Baştan sona genel dönüşüm: **%{res['overall_conv']*100:.1f}** "
                          f"({res['counts'][0]:.0f} → {res['counts'][-1]:.0f})."))
    steps = t.dropna(subset=["Kayıp %"])
    if not steps.empty:
        worst = steps.loc[steps["Kayıp %"].idxmax()]
        lines.append(I.bullet(f"**En büyük kayıp:** '{worst['Aşama']}' adımında %{worst['Kayıp %']*100:.1f} "
                              "kullanıcı düşüyor — optimizasyon için en yüksek öncelik burası."))
    lines.append(I.bullet("Küçük bir adım iyileştirmesi, hunide aşağıdaki tüm aşamaları çarpan etkisiyle büyütür."))
    return I.joinlines(lines)


# ---------------- Kohort / Retention ----------------

def cohort_retention(df: pd.DataFrame, id_col: str, event_date_col: str,
                     cohort_date_col: str | None = None, period: str = "M", max_periods: int = 12) -> dict:
    """Kohort × dönem elde tutma matrisi.
    cohort_date_col verilmezse her kullanıcının ilk olay tarihi kohort kabul edilir.
    """
    data = df[[id_col, event_date_col] + ([cohort_date_col] if cohort_date_col else [])].copy()
    data[event_date_col] = pd.to_datetime(data[event_date_col], errors="coerce")
    data = data.dropna(subset=[event_date_col])
    if cohort_date_col:
        data[cohort_date_col] = pd.to_datetime(data[cohort_date_col], errors="coerce")
        data["_cohort"] = data[cohort_date_col].dt.to_period(period)
    else:
        first = data.groupby(id_col)[event_date_col].transform("min")
        data["_cohort"] = first.dt.to_period(period)
    data["_ev"] = data[event_date_col].dt.to_period(period)
    data["_idx"] = (data["_ev"] - data["_cohort"]).apply(lambda x: x.n if hasattr(x, "n") else int(x))
    data = data[(data["_idx"] >= 0) & (data["_idx"] < max_periods)]
    # kohort büyüklüğü (dönem 0'daki tekil kullanıcı)
    sizes = data[data["_idx"] == 0].groupby("_cohort")[id_col].nunique()
    counts = data.groupby(["_cohort", "_idx"])[id_col].nunique().unstack(fill_value=0)
    ret = counts.div(sizes, axis=0)
    ret.index = ret.index.astype(str)
    ret.columns = [f"D{c}" for c in ret.columns]
    return {"retention": ret, "sizes": sizes, "period": period}


def interpret_retention(res: dict) -> str:
    ret = res["retention"]
    lines = ["### Kohort / elde tutma — yorum katmanı"]
    if ret.shape[1] < 2:
        lines.append(I.bullet("Elde tutma için en az 2 dönem gerekir."))
        return I.joinlines(lines)
    # ortalama D1 ve son dönem
    d1 = ret.iloc[:, 1].mean()
    last = ret.iloc[:, -1].mean()
    lines.append(I.bullet(f"Ortalama 1. dönem elde tutma: **%{d1*100:.1f}** → son dönem: **%{last*100:.1f}**."))
    if d1 < 0.4:
        lines.append(I.bullet("İlk dönem elde tutma düşük (<%40): kullanıcılar erken kayboluyor — "
                              "onboarding ve ilk değer deneyimine odaklanın."))
    # düzleşme
    tail = ret.iloc[:, -3:].mean().mean() if ret.shape[1] >= 3 else last
    lines.append(I.bullet(f"Uzun dönem elde tutma ~%{tail*100:.1f}'de "
                          + ("düzleşiyor → sadık çekirdek kitle var." if tail > 0.15 else
                             "düşük → tekrar kullanımı teşvik eden döngüler gerekli.")))
    lines.append(I.bullet("Kohortları karşılaştırın: sonraki kohortlar daha iyiyse ürün/pazarlama gelişiyor demektir."))
    return I.joinlines(lines)


# ---------------- RFM segmentasyonu ----------------

def rfm(df: pd.DataFrame, id_col: str, date_col: str, value_col: str, ref_date=None) -> dict:
    data = df[[id_col, date_col, value_col]].copy()
    data[date_col] = pd.to_datetime(data[date_col], errors="coerce")
    data[value_col] = pd.to_numeric(data[value_col], errors="coerce")
    data = data.dropna()
    ref = pd.to_datetime(ref_date) if ref_date else data[date_col].max()
    g = data.groupby(id_col).agg(Recency=(date_col, lambda x: (ref - x.max()).days),
                                 Frequency=(date_col, "count"),
                                 Monetary=(value_col, "sum")).reset_index()
    for col, asc in [("Recency", False), ("Frequency", True), ("Monetary", True)]:
        try:
            g[col[0]] = pd.qcut(g[col].rank(method="first"), 5, labels=[1, 2, 3, 4, 5]).astype(int)
        except Exception:
            g[col[0]] = 3
    g["RFM"] = g["R"].astype(str) + g["F"].astype(str) + g["M"].astype(str)
    g["Segment"] = g.apply(_rfm_segment, axis=1)
    seg_counts = g["Segment"].value_counts()
    return {"table": g, "segments": seg_counts, "ref_date": str(ref.date())}


def _rfm_segment(row) -> str:
    r, f, m = row["R"], row["F"], row["M"]
    if r >= 4 and f >= 4:
        return "Şampiyonlar"
    if r >= 3 and f >= 3:
        return "Sadık"
    if r >= 4 and f <= 2:
        return "Yeni/Umut vadeden"
    if r <= 2 and f >= 3:
        return "Riskli (kaybedilebilir)"
    if r <= 2 and f <= 2:
        return "Uykuda/Kayıp"
    return "Ortalama"


def interpret_rfm(res: dict) -> str:
    sc = res["segments"]
    lines = ["### RFM segmentasyonu — yorum katmanı"]
    lines.append(I.bullet(f"Referans tarih: {res['ref_date']}. En büyük segment: **{sc.index[0]}** ({sc.iloc[0]} müşteri)."))
    champ = sc.get("Şampiyonlar", 0)
    risk = sc.get("Riskli (kaybedilebilir)", 0)
    lines.append(I.bullet(f"Şampiyonlar: {champ} (ödüllendir/elde tut) · Riskli: {risk} (geri kazanım kampanyası)."))
    lines.append(I.bullet("Recency=son alışveriş yakınlığı, Frequency=sıklık, Monetary=toplam harcama. "
                          "Her segmente farklı aksiyon uygulayın."))
    return I.joinlines(lines)
