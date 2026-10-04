"""'Hangi testi kullanmalıyım?' — kural tabanlı test seçim rehberi."""
from __future__ import annotations

from . import interpret as I

# Sekme adları (app.py ile uyumlu)
TAB = {
    "hipotez": "🧫 Hipotez Testi", "anova": "🧪 ANOVA", "korelasyon": "🔗 Korelasyon",
    "regresyon": "📉 Regresyon", "lojistik": "🔀 Lojistik", "sayim": "🔢 Sayım (GLM)",
    "zaman": "⏳ Zaman Serisi", "panel": "🧊 Panel", "sagkalim": "🩺 Sağkalım",
    "kumeleme": "🧭 Kümeleme/PCA", "guc": "⚡ Güç & Güvenilirlik", "gelismis": "İleri modeller",
}


def recommend(goal: str, dv: str = None, n_groups: str = None, paired: bool = False,
              normal: bool = True, equal_var: bool = True, n_pred: str = "1",
              iv_type: str = "Sürekli") -> dict:
    """Kullanıcının cevaplarına göre uygun test(ler)i öner."""
    rec = {"test": None, "tab": None, "why": [], "alt": []}

    def out(test, tab, why, alt=None):
        rec.update(test=test, tab=TAB.get(tab, tab), why=why, alt=alt or [])
        return rec

    # 1) Grup/ortalama karşılaştırma
    if goal == "Grupları/ortalamaları karşılaştır":
        if n_groups == "1 (tek örneklem)":
            if normal:
                return out("Tek örneklem t-testi", "hipotez",
                           ["Bir grubun ortalamasını bilinen bir değerle karşılaştırıyorsun.",
                            "Veri yaklaşık normal → parametrik t-testi uygun."])
            return out("Wilcoxon işaretli sıra (tek örneklem)", "hipotez",
                       ["Normal olmayan tek grup; parametrik olmayan alternatif."])
        if n_groups == "2 grup":
            if paired:
                return out("Eşleştirilmiş t-testi" if normal else "Wilcoxon işaretli sıra", "hipotez",
                           ["Aynı denekten iki ölçüm (öncesi/sonrası) → eşleştirilmiş tasarım.",
                            ("Normal fark → parametrik." if normal else "Normal değil → Wilcoxon.")],
                           alt=["Etki büyüklüğü için Cohen's d otomatik verilir."])
            if normal:
                test = "Bağımsız örneklem t-testi" + ("" if equal_var else " (Welch)")
                return out(test, "hipotez",
                           ["İki bağımsız grubun ortalaması karşılaştırılıyor.",
                            ("Varyanslar homojen → klasik t." if equal_var else "Varyanslar eşit değil → Welch düzeltmesi.")])
            return out("Mann-Whitney U", "hipotez",
                       ["İki bağımsız grup, normal değil → parametrik olmayan Mann-Whitney."])
        # 3+ grup
        if paired:
            if normal:
                return out("Tekrarlı ölçüm ANOVA", "anova",
                           ["Aynı denekler 3+ koşulda ölçülmüş → within-subjects.",
                            "Anlamlıysa Nemenyi/eşleştirilmiş post-hoc."])
            return out("Friedman testi", "hipotez",
                       ["3+ tekrarlı ölçüm, normal değil → Friedman (+ Nemenyi post-hoc)."])
        if normal:
            if equal_var:
                return out("Tek yönlü ANOVA (+ Tukey HSD)", "anova",
                           ["3+ bağımsız grup, normal, eşit varyans → klasik ANOVA.",
                            "Anlamlıysa Tukey HSD ile ikilileri incele."])
            return out("Welch ANOVA (+ Games-Howell)", "anova",
                       ["3+ grup ama varyanslar eşit değil → Welch ANOVA.",
                        "Post-hoc için Games-Howell kullan."])
        return out("Kruskal-Wallis (+ Dunn)", "hipotez",
                   ["3+ bağımsız grup, normal değil → Kruskal-Wallis.",
                    "Anlamlıysa Dunn post-hoc ile ikilileri incele."])

    # 2) İlişki / korelasyon
    if goal == "İki değişken arasındaki ilişki":
        if dv == "İkili (0/1)":
            return out("Nokta-çift serili korelasyon", "hipotez",
                       ["Bir değişken ikili, diğeri sürekli → nokta-çift serili."])
        if not normal or iv_type == "Sıralı":
            return out("Spearman korelasyonu", "korelasyon",
                       ["Sıralı veri veya normal değil / doğrusal olmayan tekdüze ilişki → Spearman."])
        return out("Pearson korelasyonu", "korelasyon",
                   ["İki sürekli, yaklaşık normal değişken → Pearson.",
                    "Üçüncü bir değişkeni kontrol etmek istersen **Kısmi korelasyon**."],
                   alt=["Kısmi korelasyon (Korelasyon sekmesi) üçüncü değişkenin etkisini dışlar."])

    # 3) Tahmin / modelleme
    if goal == "Bir sonucu tahmin et / modelle":
        if dv == "Sürekli":
            base = ["Sürekli bir sonucu bir/çok öngörücüyle modelliyorsun → OLS regresyon."]
            if n_pred == "Çok (çoklu)":
                base.append("Birden çok modeli AIC/BIC ile kıyaslamak için Karşılaştır sekmesi.")
            return out("Doğrusal regresyon (OLS)", "regresyon", base,
                       alt=["Aykırı değer varsa robust regresyon; dağılımın farklı bölgeleri için kantil regresyon."])
        if dv == "İkili (0/1)":
            return out("İkili lojistik regresyon", "lojistik",
                       ["Sonuç iki kategorili (evet/hayır) → lojistik regresyon (odds oranları, ROC)."])
        if dv == "Sıralı":
            return out("Sıralı (ordinal) lojistik", "lojistik",
                       ["Sonuç sıralı kategorik (düşük<orta<yüksek) → ordinal lojistik."])
        if dv == "Kategorik (3+)":
            return out("Multinomial lojistik", "lojistik",
                       ["Sonuç 3+ sırasız kategori → multinomial lojistik."])
        if dv == "Sayım":
            return out("Poisson / Negatif Binom (GLM)", "sayim",
                       ["Sonuç sayım (0,1,2,…) → Poisson; aşırı yayılım varsa Negatif Binom.",
                        "Çok fazla sıfır varsa sıfır-şişirilmiş (ZIP/ZINB) modeli (İleri modeller)."])
        if dv == "Süre / olay":
            return out("Cox regresyon", "sagkalim",
                       ["Bir olaya kadar geçen süre + sansür → Cox orantılı tehlike modeli."])

    # 4) Kategorik ilişki
    if goal == "İki kategorik değişken ilişkisi":
        if paired:
            return out("McNemar testi", "hipotez",
                       ["Eşleştirilmiş ikili kategorik (öncesi/sonrası) → McNemar."])
        return out("Ki-kare bağımsızlık (küçük örneklemde Fisher exact)", "hipotez",
                   ["İki kategorik değişkenin ilişkisi → ki-kare.",
                    "2×2 ve beklenen frekans < 5 ise Fisher kesin testi."])

    # 5) Gruplama / boyut indirgeme
    if goal == "Gözlemleri grupla / boyut indir":
        return out("K-Means / Hiyerarşik kümeleme + PCA", "kumeleme",
                   ["Etiketsiz gözlemleri doğal gruplara ayır (kümeleme) ya da çok değişkeni "
                    "az bileşene indir (PCA).",
                    "Anket maddelerinin altında yatan boyutlar için Faktör Analizi (İleri modeller)."])

    # 6) Zaman içinde değişim / tahmin
    if goal == "Zaman içinde değişim / gelecek tahmini":
        return out("Zaman serisi (auto-ARIMA)", "zaman",
                   ["Zamana bağlı tek seri → trend, mevsimsellik, durağanlık ve tahmin.",
                    "Aynı birimlerin zaman içinde tekrarı (panel) ise Panel sekmesi."])

    # 7) Güvenilirlik
    if goal == "Ölçek/anket güvenilirliği":
        return out("Cronbach α (+ Faktör Analizi)", "guc",
                   ["Bir ölçeğin iç tutarlılığı → Cronbach α.",
                    "Maddelerin boyut yapısı için Faktör Analizi (İleri modeller)."])

    return out("Belirlenemedi", None, ["Seçimlerden bir öneri çıkarılamadı; soruları gözden geçir."])


def format_recommendation(rec: dict) -> str:
    if not rec.get("test") or rec["test"] == "Belirlenemedi":
        return I.joinlines(["### Öneri bulunamadı", *[I.bullet(w) for w in rec.get("why", [])]])
    lines = [f"### ✅ Önerilen: **{rec['test']}**"]
    if rec.get("tab"):
        lines.append(I.bullet(f"Nereden: **{rec['tab']}** sekmesi."))
    for w in rec["why"]:
        lines.append(I.bullet(w))
    if rec.get("alt"):
        lines.append("\n**İpuçları:**")
        for aopt in rec["alt"]:
            lines.append(I.bullet(aopt))
    return I.joinlines(lines)
