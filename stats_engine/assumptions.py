"""Varsayım & geçerlilik bekçisi.

Ayrım: KONTROLLER (ham gerçekler: test adı + durum + detay) ile
GÜVEN SKORU / VERDİKT (yorum katmanı) ayrı döner.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from . import interpret as I


def _normality(sv: pd.Series):
    sv = pd.to_numeric(sv, errors="coerce").dropna()
    if len(sv) < 3:
        return None
    p = stats.shapiro(sv)[1] if len(sv) <= 5000 else stats.normaltest(sv)[1]
    return p


def _check(name, ok, detail, weight=1.0):
    return {"kontrol": name, "durum": "geçti" if ok else "uyarı", "ok": bool(ok),
            "detay": detail, "agirlik": weight}


def for_ttest(a: pd.Series, b: pd.Series | None = None, alpha: float = 0.05) -> dict:
    checks = []
    sa = pd.to_numeric(a, errors="coerce").dropna()
    checks.append(_check("Örneklem büyüklüğü (A)", len(sa) >= 30 or (_normality(sa) or 0) >= alpha,
                         f"n={len(sa)}" + ("" if len(sa) >= 30 else "; küçük örneklem, normallik önemli")))
    pa = _normality(sa)
    if pa is not None:
        checks.append(_check("Normallik (A)", pa >= alpha, f"Shapiro p={pa:.3g}"))
    if b is not None:
        sb = pd.to_numeric(b, errors="coerce").dropna()
        pb = _normality(sb)
        if pb is not None:
            checks.append(_check("Normallik (B)", pb >= alpha, f"Shapiro p={pb:.3g}"))
        lev = stats.levene(sa, sb, center="median")[1]
        checks.append(_check("Varyans homojenliği", lev >= alpha,
                             f"Levene p={lev:.3g}" + ("" if lev >= alpha else "; Welch t kullanın")))
    return _score(checks, "t-testi")


def for_anova(groups: list[pd.Series], alpha: float = 0.05) -> dict:
    checks = []
    arrs = [pd.to_numeric(g, errors="coerce").dropna() for g in groups]
    arrs = [a for a in arrs if len(a) > 2]
    small = [i for i, a in enumerate(arrs) if len(a) < 20]
    checks.append(_check("Grup büyüklükleri", len(small) == 0,
                         "tüm gruplar ≥20" if not small else f"{len(small)} grup <20 gözlem"))
    ps = [_normality(a) for a in arrs]
    ps = [p for p in ps if p is not None]
    if ps:
        checks.append(_check("Normallik (gruplar)", min(ps) >= alpha,
                             f"en küçük Shapiro p={min(ps):.3g}"))
    if len(arrs) >= 2:
        lev = stats.levene(*arrs, center="median")[1]
        checks.append(_check("Varyans homojenliği", lev >= alpha,
                             f"Levene p={lev:.3g}" + ("" if lev >= alpha else "; Welch ANOVA + Games-Howell")))
    return _score(checks, "ANOVA")


def for_regression(resid: pd.Series, diag: dict, alpha: float = 0.05) -> dict:
    checks = []
    pr = _normality(pd.Series(resid))
    if pr is not None:
        checks.append(_check("Artık normalliği", pr >= alpha, f"Shapiro p={pr:.3g}"))
    if "breusch_pagan_p" in diag and diag["breusch_pagan_p"] == diag["breusch_pagan_p"]:
        bp = diag["breusch_pagan_p"]
        checks.append(_check("Sabit varyans (homoskedasti)", bp >= alpha,
                             f"Breusch-Pagan p={bp:.3g}" + ("" if bp >= alpha else "; robust std. hata")))
    if "durbin_watson" in diag:
        dw = diag["durbin_watson"]
        checks.append(_check("Bağımsızlık (otokorelasyon)", 1.5 <= dw <= 2.5, f"Durbin-Watson={dw:.2f}"))
    return _score(checks, "Regresyon")


def _score(checks: list[dict], analysis: str) -> dict:
    if not checks:
        return {"analysis": analysis, "checks": pd.DataFrame(), "score": np.nan, "verdict": "Kontrol yapılamadı."}
    w = sum(c["agirlik"] for c in checks)
    passed = sum(c["agirlik"] for c in checks if c["ok"])
    score = int(round(passed / w * 100)) if w else 0
    df = pd.DataFrame([{"Kontrol": c["kontrol"], "Durum": "✅ geçti" if c["ok"] else "⚠️ uyarı",
                        "Detay": c["detay"]} for c in checks])
    return {"analysis": analysis, "checks": df, "score": score,
            "n_warn": sum(1 for c in checks if not c["ok"])}


def interpret_score(res: dict) -> str:
    """Yorum katmanı: güven skoru + öneri (kontrollerden ayrı)."""
    s = res["score"]
    if s != s:
        return res.get("verdict", "Kontrol yapılamadı.")
    band = ("yüksek — sonuçlara güvenilebilir" if s >= 80 else
            "orta — bazı varsayımlar zorlanıyor, dikkatli yorumla" if s >= 50 else
            "düşük — varsayımlar ihlal ediliyor, sonuç yanıltıcı olabilir")
    lines = [f"### {res['analysis']} — geçerlilik değerlendirmesi (yorum katmanı)"]
    lines.append(I.bullet(f"**Güven skoru: {s}/100** → {band}."))
    if res.get("n_warn", 0) > 0:
        lines.append(I.bullet(f"{res['n_warn']} varsayım uyarısı var (yukarıdaki kontrol tablosuna bakın). "
                              "İlgili öneriyi (Welch/robust/parametrik olmayan) uygulamayı düşün."))
    else:
        lines.append(I.bullet("Tüm temel varsayımlar karşılanıyor. ✅"))
    lines.append(I.bullet("Bu skor yalnızca test varsayımlarını özetler; pratik/iş anlamlılığını ayrıca değerlendir."))
    return I.joinlines(lines)
