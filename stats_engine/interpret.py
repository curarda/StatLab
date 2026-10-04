"""Kural tabanlı Türkçe yorumlama yardımcıları.

Her analiz modülü sonuç sözlüğü üretir; buradaki fonksiyonlar bu sözlükleri
insan diline çevirir. Amaç: p-değeri, R², F, korelasyon katsayısı gibi
büyüklükleri tutarlı, anlaşılır Türkçe cümlelere dönüştürmek.
"""
from __future__ import annotations

ALPHA_DEFAULT = 0.05


def stars(p: float) -> str:
    """Anlamlılık yıldızları."""
    if p is None or p != p:  # NaN
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    if p < 0.10:
        return "."
    return ""


def p_label(p: float, alpha: float = ALPHA_DEFAULT) -> str:
    if p is None or p != p:
        return "hesaplanamadı"
    if p < alpha:
        return "anlamlı"
    return "anlamlı değil"


def p_sentence(p: float, alpha: float = ALPHA_DEFAULT) -> str:
    """p-değeri için standart karar cümlesi."""
    if p is None or p != p:
        return "p-değeri hesaplanamadı."
    p_txt = "< 0.001" if p < 0.001 else f"= {p:.4f}"
    if p < alpha:
        return (
            f"p {p_txt} < α = {alpha:g} olduğundan sonuç **istatistiksel olarak "
            f"anlamlıdır**; boş hipotez (H₀) reddedilir."
        )
    return (
        f"p {p_txt} ≥ α = {alpha:g} olduğundan sonuç **istatistiksel olarak "
        f"anlamlı değildir**; boş hipotez (H₀) reddedilemez."
    )


def effect_size_r(r: float) -> str:
    """Korelasyon katsayısı büyüklüğü (Cohen)."""
    a = abs(r)
    if a < 0.10:
        return "ihmal edilebilir"
    if a < 0.30:
        return "zayıf"
    if a < 0.50:
        return "orta"
    if a < 0.70:
        return "güçlü"
    return "çok güçlü"


def direction(r: float) -> str:
    if r > 0:
        return "pozitif (aynı yönde)"
    if r < 0:
        return "negatif (ters yönde)"
    return "yönsüz"


def r2_quality(r2: float) -> str:
    if r2 < 0.10:
        return "çok düşük — model değişkenliğin çok küçük bir kısmını açıklıyor"
    if r2 < 0.30:
        return "düşük"
    if r2 < 0.50:
        return "orta düzey"
    if r2 < 0.70:
        return "iyi"
    if r2 < 0.90:
        return "yüksek"
    return "çok yüksek (aşırı uyum/overfitting açısından da kontrol edilmeli)"


def cohens_d_label(d: float) -> str:
    a = abs(d)
    if a < 0.20:
        return "ihmal edilebilir"
    if a < 0.50:
        return "küçük"
    if a < 0.80:
        return "orta"
    return "büyük"


def eta_sq_label(eta2: float) -> str:
    if eta2 < 0.01:
        return "ihmal edilebilir"
    if eta2 < 0.06:
        return "küçük"
    if eta2 < 0.14:
        return "orta"
    return "büyük"


def bullet(text: str) -> str:
    return f"- {text}"


def joinlines(lines: list[str]) -> str:
    return "\n".join(l for l in lines if l)
