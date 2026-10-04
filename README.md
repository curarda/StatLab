# StatLab — İstatistiksel Analiz ve Veri Analizi

Kendi verinizi (CSV / Excel / JSON) ya da regresyon fonksiyonunuzu verin; uygulama
regresyon, korelasyon, ANOVA ve hipotez testlerini çalıştırıp **grafik + otomatik
Türkçe yorum** üretsin. İsterseniz Claude API ile derin doğal dil yorumu ekleyin.

## Kurulum ve çalıştırma

En kolayı:

```bash
cd "/Users/mac/StatLab"
./run.sh
```

`run.sh` gerekirse sanal ortamı kurar ve uygulamayı tarayıcıda açar. Elle:

```bash
./.venv/bin/streamlit run app.py
```

Uygulama `http://localhost:8501` adresinde açılır.

## Özellikler

| Sekme | Ne yapar |
|-------|----------|
| 📁 Veri | Önizleme, tipler, eksik değer raporu |
| 📈 Betimsel | Ortalama/sapma/çarpıklık + histogram, Q-Q, Shapiro & D'Agostino normallik |
| 🔗 Korelasyon | Pearson/Spearman/Kendall ısı haritası, anlamlılık, saçılım + regresyon çizgisi |
| 📉 Regresyon | OLS: katsayı t-testleri, R²/F, VIF, artık tanıları, tahmin aracı, grafikler |
| 🔀 Lojistik | İkili, **çok sınıflı (multinomial)** ve **sıralı (ordinal)** lojistik: odds oranları, ROC/AUC, karışıklık matrisi |
| 🧪 ANOVA | Tek/çift yönlü, **Welch**, **tekrarlı ölçüm**, η²/ω², Levene, **Tukey/Games-Howell/Nemenyi** post-hoc |
| 🧫 Hipotez | t-testleri, Mann-Whitney, Wilcoxon, Kruskal-Wallis (+Dunn), Friedman (+Nemenyi), ki-kare, **Fisher, McNemar, uyum iyiliği, nokta-çift serili, binom, oran z, Bartlett** |
| ⏳ Zaman Serisi | Trend, ADF/KPSS durağanlık, mevsimsel ayrıştırma, ACF/PACF, Holt-Winters / ARIMA / **auto-ARIMA** tahmin |
| 🧊 Panel | Panel veri: Pooled OLS, Sabit Etkiler (FE), Rassal Etkiler (RE) + **Hausman testi** |
| 🧭 Kümeleme/PCA | K-Means, hiyerarşik kümeleme (silhouette, elbow) ve **PCA** boyut indirgeme |
| 🔢 Sayım (GLM) | Poisson / Negatif Binom regresyon, IRR, aşırı yayılım kontrolü |
| ⚡ Güç & Güvenilirlik | Güç analizi / örneklem büyüklüğü, **Cronbach α**, çoklu karşılaştırma (Bonferroni/FDR) |
| 🅰️ A/B Test | Deney tasarımı (örneklem/süre), dönüşüm & sürekli metrik analizi, uplift + Bayesçi bakış |
| 📿 Bayesçi | Bayes faktörü (t-testi, korelasyon) — H₀ lehine de kanıt sunar; Bayesçi A/B (P(B>A)) |
| 🩺 Sağkalım | Kaplan-Meier + log-rank, **Cox** orantılı tehlike modeli (HR) |
| 🧬 Faktör Analizi | Açımlayıcı faktör analizi (EFA) + KMO, Bartlett küresellik, scree, yükler |
| 🔗 Karışık Model | Çok düzeyli / karışık etki modeli (MixedLM), ICC, rassal kesişim/eğim |
| 🧙 Test Seçici | "Hangi testi kullanmalıyım?" — verine göre doğru yöntemi öneren sihirbaz |
| 💰 Finans — Getiri & Risk | yfinance ile canlı fiyat, Sharpe/Sortino/Calmar, maks. düşüş, VaR/CVaR, drawdown & korelasyon |
| 💰 Finans — Optimizasyon | Markowitz etkin sınır, maks-Sharpe / min-varyans / risk paritesi portföyleri |
| 💰 Finans — CAPM/Faktör | Alpha/beta, yuvarlanan beta, çok faktörlü regresyon |
| 💰 Finans — Monte Carlo/GARCH | Portföy simülasyonu (hedef olasılığı) ve GARCH oynaklık modeli |
| 💰 Finans — Makro & Faiz | Canlı ABD Hazine getirileri (3A/5Y/10Y/30Y), getiri eğrisi, eğim (resesyon sinyali), VIX |
| 💰 Optimizasyon — risk profili | Muhafazakâr/Dengeli/Büyüme/Agresif profillere göre portföy (sermaye tahsis doğrusu) |
| 📣 Pazarlama — Funnel | Dönüşüm hunisi, adım kayıpları, en zayıf halka |
| 📣 Pazarlama — Kohort/Retention | Kohort × dönem elde tutma matrisi + ısı haritası |
| 📣 Pazarlama — RFM | Recency/Frequency/Monetary müşteri segmentasyonu |

Karışık-tip ilişki matrisi (Korelasyon), ANCOVA (ANOVA), CUPED (A/B) da eklendi.

### Rigor & derinlik katmanı

| Özellik | Nerede |
|---------|--------|
| ✔️ **Çapraz doğrulama / holdout** (örneklem-dışı R²/AUC, aşırı uyum tespiti) | Regresyon & Modelleme → Doğrulama |
| 🔀 **Aracılık & düzenleyicilik** (X→M→Y, etkileşim/basit eğimler) | Regresyon & Modelleme → Aracılık/Düzenleyici |
| ⚖️ **Eşdeğerlik testi (TOST)** ("fark yok"u kanıtlar) | Hipotez Testi |
| 🔁 **Backtest** (portföy geçmiş simülasyonu, yeniden dengeleme, bilgi oranı) | Finans → Backtest |
| 💵 **Tahvil matematiği** (fiyat, YTM, süre/duration, konveksite) | Finans → Tahvil |

Ek yetenekler: **kısmi korelasyon** (Korelasyon), **robust/kantil/adımsal regresyon + Cook-kaldıraç tanıları** (Regresyon), **sıfır-şişirilmiş ZIP/ZINB** (Sayım).

### 🔎 İçgörü & Yorum (yorum katmanı ayrı tutulur)

Ham parametreler ve **yorum/öneri katmanı görsel ve mantıksal olarak ayrıdır**; sol menüden "💡 Yorum & öneri katmanı" düğmesiyle katman tümüyle gizlenebilir.

| Sekme | Ne yapar |
|-------|----------|
| 💡 Otomatik İçgörü | Tek tıkla veriyi tarar; güçlü korelasyon, grup farkı, çarpıklık gibi örüntüleri önem sırasıyla bulur |
| 📋 Veri Profili & Kalite | Sütun profili + eksik/aykırı/sabit/sızıntı bayrakları |
| ✅ Varsayım Bekçisi | Test varsayımlarını kontrol eder, 0–100 **güven skoru** verir |
| 🔗 Nedensellik | Karıştırıcı kontrolü, fark-içinde-fark (DiD), eğilim skoru eşleştirme (correlation→causation) |
| 📝 Yönetici Özeti | Oturumdaki analizleri "ne bulundu / ne anlama geliyor / ne yapmalı" hikayesine dönüştürür |
| ƒ Fonksiyon | `y = 2*x + 3` fonksiyonu çiz + türev/kök/tip analizi **veya veriye eğri uydur** (polinom/üstel/log/güç) |
| 📄 Rapor | Analizleri tek PDF'te topla + **Excel/CSV dışa aktar** |
| 🧹 Temizle | Eksik değer doldurma, aykırı değer tespiti (IQR/z-skoru), dönüşüm (log/z/Box-Cox…) |
| ⚖️ Karşılaştır | Birden çok regresyon modelini AIC/BIC/düzeltilmiş R² ile yan yana kıyasla |

Ayrıca: **oturum kaydet/yükle** (`.statlab` dosyası, sol menü) ve **çift-tıkla açılan Mac uygulaması** (`StatLab.app`).

Her analizde: **kural tabanlı Türkçe yorum** (çevrimdışı, ücretsiz) + isteğe bağlı
**"Claude ile derinleştir"** butonu.

## Claude yorumu (opsiyonel)

Sol menüye `ANTHROPIC_API_KEY` girin veya ortam değişkeni olarak tanımlayın:

```bash
export ANTHROPIC_API_KEY="sk-ant-..."
```

Anahtar yoksa uygulama kural tabanlı yorumlarla tam çalışır.

## Çift tıkla açılan Mac uygulaması (.app)

Hafif başlatıcı `.app` oluştur (sunucuyu başlatıp tarayıcıda açar):

```bash
cd "/Users/mac/StatLab" && ./build_app.sh
```

`StatLab.app` oluşur; çift tıklayabilir veya Uygulamalar'a taşıyabilirsiniz. İlk açılışta
macOS bir kez otomasyon izni isteyebilir (normaldir). Tam bağımsız bundle denemek isteyenler
için `packaging/setup.py` (py2app) mevcuttur — ama bilimsel yığın nedeniyle deneyseldir.

## Proje yapısı

```
StatLab/
├── app.py                    # Streamlit arayüzü (UI/UX)
├── requirements.txt
├── run.sh
└── stats_engine/
    ├── loaders.py            # CSV/Excel/JSON yükleme
    ├── descriptive.py        # betimsel + normallik
    ├── correlation.py        # korelasyon
    ├── regression.py         # OLS + tanılar
    ├── anova.py              # ANOVA + post-hoc
    ├── hypothesis.py         # hipotez testleri
    ├── logistic.py           # lojistik regresyon + ROC/sınıflandırma
    ├── timeseries.py         # trend/durağanlık/ayrıştırma/tahmin
    ├── report.py             # PDF rapor (matplotlib)
    ├── function_parser.py    # fonksiyon girişi (sympy)
    ├── interpret.py          # kural tabanlı Türkçe yorum
    └── claude_interpret.py   # Claude API (opsiyonel)
```
