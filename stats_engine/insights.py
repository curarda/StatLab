"""Otomatik veri profili, kalite kontrolü ve içgörü motoru.

TASARIM İLKESİ — yorum katmanı ayrımı:
Her bulgu, HAM GERÇEK ile YORUM/ÖNERİyi ayrı alanlarda tutar:
    {"fact": <ham sayı/olgu>, "yorum": <ne anlama geliyor>,
     "oneri": <ne yapmalı>, "onem": <0-1 önem>}
Hesaplama (facts) ile yorum (yorum/oneri) hiçbir zaman karışmaz.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from . import interpret as I


# ---------------- Ham profil (yalnızca gerçekler) ----------------

def profile(df: pd.DataFrame) -> dict:
    rows = []
    for c in df.columns:
        s = df[c]
        num = pd.api.types.is_numeric_dtype(s)
        rec = {"Sütun": c, "Tip": "sayısal" if num else "kategorik",
               "Dolu": int(s.notna().sum()), "Eksik %": round(s.isna().mean() * 100, 1),
               "Benzersiz": int(s.nunique())}
        if num:
            sv = pd.to_numeric(s, errors="coerce").dropna()
            rec.update({"Ortalama": sv.mean() if len(sv) else np.nan,
                        "Std": sv.std(ddof=1) if len(sv) > 1 else np.nan,
                        "Çarpıklık": stats.skew(sv, bias=False) if len(sv) > 2 else np.nan})
        rows.append(rec)
    return {"columns": pd.DataFrame(rows), "n_rows": len(df), "n_cols": df.shape[1],
            "duplicates": int(df.duplicated().sum()),
            "total_missing_pct": round(df.isna().mean().mean() * 100, 1)}


# ---------------- Veri kalitesi bayrakları (gerçek + öneri ayrı) ----------------

def quality_flags(df: pd.DataFrame) -> list[dict]:
    flags = []
    n = len(df)
    for c in df.columns:
        s = df[c]
        miss = s.isna().mean()
        if miss > 0.5:
            flags.append({"seviye": "yüksek", "sütun": c,
                          "fact": f"%{miss*100:.0f} eksik",
                          "yorum": "Sütunun yarısından fazlası boş; analizde güvenilmez.",
                          "oneri": "Sütunu çıkarmayı ya da neden eksik olduğunu araştırmayı düşün."})
        elif miss > 0.2:
            flags.append({"seviye": "orta", "sütun": c, "fact": f"%{miss*100:.0f} eksik",
                          "yorum": "Kayda değer eksik veri.",
                          "oneri": "Temizle sekmesinden doldur (medyan/mod) veya eksik satırları değerlendir."})
        if s.nunique(dropna=True) <= 1:
            flags.append({"seviye": "yüksek", "sütun": c, "fact": "sabit/tek değer",
                          "yorum": "Değişkenlik yok; hiçbir analize bilgi katmaz.",
                          "oneri": "Bu sütunu çıkar."})
        elif not pd.api.types.is_numeric_dtype(s) and s.nunique(dropna=True) > 0.9 * n and n > 20:
            flags.append({"seviye": "orta", "sütun": c, "fact": f"{s.nunique()} benzersiz (~kimlik)",
                          "yorum": "Neredeyse her satır farklı; kimlik/ID sütunu olabilir.",
                          "oneri": "Modelleme/gruplama dışında tut."})
        if pd.api.types.is_numeric_dtype(s):
            sv = pd.to_numeric(s, errors="coerce").dropna()
            if len(sv) >= 4:
                q1, q3 = sv.quantile(0.25), sv.quantile(0.75)
                iqr = q3 - q1
                nout = int(((sv < q1 - 1.5 * iqr) | (sv > q3 + 1.5 * iqr)).sum())
                if nout > 0 and nout / len(sv) > 0.02:
                    flags.append({"seviye": "orta", "sütun": c,
                                  "fact": f"{nout} aykırı değer (%{nout/len(sv)*100:.1f})",
                                  "yorum": "Uç değerler ortalama/regresyonu çarpıtabilir.",
                                  "oneri": "Temizle sekmesinden incele; hata mı gerçek mi kontrol et."})
    if df.duplicated().sum() > 0:
        flags.append({"seviye": "orta", "sütun": "(tablo)", "fact": f"{int(df.duplicated().sum())} yinelenen satır",
                      "yorum": "Tekrarlı kayıtlar sonuçları yanlı yapabilir.",
                      "oneri": "Yinelenenleri kaldırmayı değerlendir."})
    # birebir aynı (kopya) sütunlar
    num = df.select_dtypes("number")
    cols = list(num.columns)
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            a, b = num[cols[i]], num[cols[j]]
            if a.equals(b):
                flags.append({"seviye": "orta", "sütun": f"{cols[i]} = {cols[j]}",
                              "fact": "birebir aynı sütunlar",
                              "yorum": "İki sütun tamamen aynı (olası veri sızıntısı/kopya).",
                              "oneri": "Birini çıkar."})
    return flags


# ---------------- Otomatik içgörüler (gerçek + yorum + öneri ayrı) ----------------

def auto_insights(df: pd.DataFrame, alpha: float = 0.05, max_items: int = 12) -> list[dict]:
    ins = []
    # İkili (0/1) sayısal sütunlar kategorik gibi de değerlendirilir (ör. dönüşüm, tıklama)
    num_all = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    num_cols = [c for c in num_all if df[c].nunique(dropna=True) > 2]          # sürekli
    binary_cols = [c for c in num_all if df[c].nunique(dropna=True) == 2]      # ikili sayısal
    cat_cols = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]
    cat_like = cat_cols + binary_cols  # ilişki/gruplama için "kategorik gibi"

    # 1) Güçlü korelasyonlar (sürekli–sürekli)
    if len(num_cols) >= 2:
        for i in range(len(num_cols)):
            for j in range(i + 1, len(num_cols)):
                pair = df[[num_cols[i], num_cols[j]]].apply(pd.to_numeric, errors="coerce").dropna()
                if len(pair) < 5:
                    continue
                r, p = stats.pearsonr(pair.iloc[:, 0], pair.iloc[:, 1])
                if abs(r) >= 0.5 and p < alpha:
                    yon = "pozitif" if r > 0 else "negatif"
                    ins.append({"tur": "Korelasyon", "onem": abs(r),
                                "fact": f"{num_cols[i]} ↔ {num_cols[j]}: r = {r:.2f} (p={p:.3g}, n={len(pair)})",
                                "yorum": f"{I.effect_size_r(r).capitalize()} {yon} doğrusal ilişki. "
                                         f"Biri artarken diğeri {'artıyor' if r>0 else 'azalıyor'}.",
                                "oneri": "Regresyon sekmesinde modelle; ama korelasyon ≠ nedensellik — "
                                         "Nedensellik sekmesiyle doğrula."})

    # 2) Grup farkları (kategorik/ikili × sürekli sayısal) — anlamlıysa dahil et, önemi η²'ye göre
    for cc in cat_like:
        levels = df[cc].dropna().nunique()
        if not (2 <= levels <= 10):
            continue
        for nc in num_cols:
            sub = df[[cc, nc]].copy()
            sub[nc] = pd.to_numeric(sub[nc], errors="coerce")
            sub = sub.dropna()
            groups = [g[nc].values for _, g in sub.groupby(cc) if len(g) >= 3]
            if len(groups) < 2:
                continue
            try:
                f, p = stats.f_oneway(*groups)
            except Exception:
                continue
            if p < alpha:
                allv = np.concatenate(groups)
                grand = allv.mean()
                ss_b = sum(len(g) * (g.mean() - grand) ** 2 for g in groups)
                ss_t = ((allv - grand) ** 2).sum()
                eta2 = ss_b / ss_t if ss_t else 0
                if eta2 >= 0.02:  # anlamlı + en az küçük etki
                    mag = "belirgin" if eta2 > 0.14 else "orta" if eta2 > 0.06 else "küçük ama anlamlı"
                    ins.append({"tur": "Grup farkı", "onem": min(1, 0.35 + eta2 * 3),
                                "fact": f"'{nc}', '{cc}' gruplarına göre farklı (F={f:.2f}, p={p:.3g}, η²={eta2:.2f})",
                                "yorum": f"{cc} düzeyleri arasında {nc} ortalaması {mag} biçimde ayrışıyor.",
                                "oneri": "ANOVA/Kruskal sekmesinde post-hoc ile hangi grupların farklı olduğuna bak."})

    # 3) Kategorik ↔ kategorik ilişki (ki-kare + Cramér's V) — ör. persona × dönüşüm/kanal
    for i in range(len(cat_like)):
        for j in range(i + 1, len(cat_like)):
            a, b = cat_like[i], cat_like[j]
            tab = pd.crosstab(df[a], df[b])
            if tab.shape[0] < 2 or tab.shape[1] < 2 or tab.values.sum() < 20:
                continue
            try:
                chi2, p, dof, exp = stats.chi2_contingency(tab)
            except Exception:
                continue
            n = tab.values.sum()
            v = np.sqrt(chi2 / (n * (min(tab.shape) - 1))) if min(tab.shape) > 1 else 0
            if p < alpha and v >= 0.10:
                strength = "güçlü" if v > 0.3 else "orta" if v > 0.2 else "zayıf ama anlamlı"
                ins.append({"tur": "Kategorik ilişki", "onem": min(1, 0.4 + v),
                            "fact": f"'{a}' ↔ '{b}': anlamlı ilişki (χ²={chi2:.1f}, p={p:.3g}, Cramér's V={v:.2f})",
                            "yorum": f"{a} ile {b} bağımsız değil; {strength} bir bağ var "
                                     f"(bir profildeki dağılım diğerine göre değişiyor).",
                            "oneri": "Hipotez sekmesinde ki-kare/çapraz tablo ile hangi hücrelerin öne çıktığına bak."})

    # 3) Çarpıklık
    for nc in num_cols:
        sv = pd.to_numeric(df[nc], errors="coerce").dropna()
        if len(sv) > 8:
            sk = stats.skew(sv, bias=False)
            if abs(sk) > 1:
                ins.append({"tur": "Dağılım", "onem": min(1, abs(sk) / 3),
                            "fact": f"'{nc}' çarpıklık = {sk:.2f}",
                            "yorum": f"Dağılım belirgin biçimde {'sağa' if sk>0 else 'sola'} çarpık; "
                                     "parametrik testlerin normallik varsayımını zorlayabilir.",
                            "oneri": "Temizle sekmesinde log/Box-Cox dönüşümü ya da parametrik olmayan test kullan."})

    ins.sort(key=lambda d: -d["onem"])
    return ins[:max_items]


def summarize_findings(profile_res: dict, flags: list[dict], insights: list[dict]) -> str:
    """Yorum katmanı — bulguların düz-dil özeti (yalnızca yorum, gerçekler ayrı gösterilir)."""
    lines = ["### Veriden çıkan anlam (yorum katmanı)"]
    lines.append(I.bullet(f"Veri {profile_res['n_rows']} satır × {profile_res['n_cols']} sütun; "
                          f"ortalama %{profile_res['total_missing_pct']} eksiklik, "
                          f"{profile_res['duplicates']} yinelenen satır."))
    high = [f for f in flags if f["seviye"] == "yüksek"]
    if high:
        lines.append(I.bullet(f"⚠️ {len(high)} kritik kalite sorunu var (yukarıdaki tabloya bakın) — "
                              "analizden önce ele alınmalı."))
    elif flags:
        lines.append(I.bullet(f"{len(flags)} küçük/orta kalite uyarısı; çoğu analiz güvenle yapılabilir."))
    else:
        lines.append(I.bullet("Belirgin veri kalitesi sorunu görülmedi. ✅"))
    if insights:
        top = insights[0]
        lines.append(I.bullet(f"En dikkat çekici bulgu: **{top['fact']}** — {top['yorum']}"))
        lines.append(I.bullet(f"Toplam {len(insights)} kayda değer örüntü bulundu (aşağıda önem sırasına göre)."))
    else:
        lines.append(I.bullet("Otomatik taramada güçlü bir örüntü (korelasyon/grup farkı) öne çıkmadı."))
    return I.joinlines(lines)
