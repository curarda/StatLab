"""StatLab — İstatistiksel Analiz ve Veri Analizi Uygulaması.

Çalıştırma:  streamlit run app.py
"""
from __future__ import annotations

import io
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from stats_engine import (
    loaders, descriptive, correlation, regression, anova, hypothesis,
    function_parser, interpret as I, claude_interpret,
    logistic, timeseries, report, cleaning, fitting, compare, export, panel,
    count_models, power, reliability, clustering, survival,
    posthoc, advanced, regression_extras, wizard, finance, ab_test, bayesian,
    insights, assumptions, causal, summary as summary_engine,
    association, experiments, marketing, validation, mediation,
)

st.set_page_config(page_title="StatLab — İstatistiksel Analiz", page_icon="📊", layout="wide")

# ---------------- Stil ----------------
st.markdown(
    """
    <style>
      .main .block-container {padding-top: 1.5rem; max-width: 1300px;}
      h1, h2, h3 {letter-spacing: -0.02em;}
      .stTabs [data-baseweb="tab-list"] {gap: 4px;}
      .stTabs [data-baseweb="tab"] {padding: 8px 14px; border-radius: 8px 8px 0 0;}
      div[data-testid="stMetricValue"] {font-size: 1.4rem;}
      .interp-box {background: rgba(99,102,241,.07); border-left: 4px solid #6366f1;
                   padding: 14px 18px; border-radius: 8px; margin: 6px 0 14px;}
      .interp-head {font-size:.72rem; font-weight:700; letter-spacing:.06em; text-transform:uppercase;
                    color:#6366f1; margin-bottom:8px; opacity:.9;}
      .badge {display:inline-block;padding:2px 10px;border-radius:12px;font-size:.75rem;
              font-weight:600;margin-left:6px;}
      .b-sig {background:#dcfce7;color:#166534;}
      .b-nsig {background:#fee2e2;color:#991b1b;}
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------- Durum ----------------
if "df" not in st.session_state:
    st.session_state.df = None
if "df_name" not in st.session_state:
    st.session_state.df_name = None


def interp_box(md: str, header: str = "💡 Yorum & Öneri"):
    """Yorum katmanı — ham parametrelerden görsel ve mantıksal olarak ayrı.
    Sol menüdeki düğme ile tamamen gizlenebilir (yalnızca parametreler görünür)."""
    if not st.session_state.get("show_interp", True):
        return
    st.markdown(
        f'<div class="interp-box"><div class="interp-head">{header}</div>{_md_to_html(md)}</div>',
        unsafe_allow_html=True)


def _render_funnel(res):
    t = res["table"]
    ff = go.Figure(go.Funnel(y=t["Aşama"], x=t["Adet"], textinfo="value+percent initial"))
    ff.update_layout(height=340, margin=dict(t=20))
    st.plotly_chart(ff, use_container_width=True)
    st.dataframe(t.style.format({"Adım dönüşümü": "{:.1%}", "Kümülatif %": "{:.1%}", "Kayıp %": "{:.1%}"},
                                na_rep="—"), use_container_width=True, hide_index=True)
    interp_box(marketing.interpret_funnel(res))


def _md_to_html(md: str) -> str:
    # Basit markdown -> HTML (kalın + madde + satır)
    import html
    out_lines = []
    for line in md.split("\n"):
        raw = line
        line = html.escape(line, quote=False)
        # **bold**
        while "**" in line:
            line = line.replace("**", "<b>", 1)
            line = line.replace("**", "</b>", 1)
        if raw.strip().startswith("- "):
            out_lines.append("• " + line.strip()[2:])
        elif raw.strip().startswith("### "):
            out_lines.append("<h4 style='margin:.3rem 0'>" + line.strip()[4:] + "</h4>")
        else:
            out_lines.append(line)
    return "<br>".join(out_lines)


def claude_expander(context: str, key: str):
    ok, msg = claude_interpret.available()
    api_key = st.session_state.get("api_key") or None
    with st.expander("🤖 Claude ile derinleştir (doğal dilde yorum)"):
        if not (ok or api_key):
            st.info(f"Claude yorumu için sol menüden API anahtarı girin. Durum: {msg}")
            return
        q = st.text_input("İsteğe bağlı ek soru", key=f"q_{key}",
                          placeholder="Örn: Bu sonucu bir yöneticiye nasıl anlatırım?")
        if st.button("Claude yorumu al", key=f"btn_{key}"):
            with st.spinner("Claude düşünüyor..."):
                model = st.session_state.get("model", claude_interpret.DEFAULT_MODEL)
                ans = claude_interpret.deep_interpret(context, q or None, model=model, api_key=api_key)
            st.markdown(ans)


def _make_sample(name: str) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    if name == "Reklam → Satış":
        tv = rng.uniform(10, 300, 120)
        radyo = rng.uniform(0, 50, 120)
        gazete = rng.uniform(0, 100, 120)
        satis = 3 + 0.045 * tv + 0.18 * radyo + 0.01 * gazete + rng.normal(0, 1.5, 120)
        return pd.DataFrame({"TV": tv, "Radyo": radyo, "Gazete": gazete, "Satis": satis})
    if name == "Öğrenci Notları":
        calisma = rng.uniform(0, 20, 100)
        uyku = rng.uniform(4, 10, 100)
        devam = rng.uniform(50, 100, 100)
        cinsiyet = rng.choice(["Kadın", "Erkek"], 100)
        not_ = 20 + 2.5 * calisma + 1.2 * uyku + 0.2 * devam + rng.normal(0, 5, 100)
        return pd.DataFrame({"CalismaSaati": calisma, "Uyku": uyku, "DevamYuzde": devam,
                             "Cinsiyet": cinsiyet, "Not": np.clip(not_, 0, 100)})
    if name == "Sağkalım (klinik)":
        n = 200
        yas = rng.normal(60, 10, n)
        tedavi = rng.choice(["İlaç A", "İlaç B"], n)
        risk = 0.03 * (yas - 60) + np.where(tedavi == "İlaç B", -0.5, 0.5)
        sure = rng.exponential(np.exp(2.5 - risk))
        olay = (rng.uniform(size=n) < 0.7).astype(int)  # 1=olay, 0=sansür
        return pd.DataFrame({"Sure": sure.round(1), "Olay": olay, "Tedavi": tedavi, "Yas": yas.round(0)})
    if name == "Anket (Likert 1-5)":
        n = 250
        gizli = rng.normal(0, 1, n)  # gizli tutum
        items = {f"Madde{i+1}": np.clip(np.round(3 + 1.2 * gizli + rng.normal(0, 0.7, n)), 1, 5).astype(int)
                 for i in range(6)}
        memnuniyet = pd.cut(3 + 1.5 * gizli + rng.normal(0, 0.8, n),
                            bins=[-99, 2, 4, 99], labels=["Düşük", "Orta", "Yüksek"])
        sikayet = rng.poisson(np.exp(0.5 - 0.4 * gizli))  # sayım
        return pd.DataFrame({**items, "Memnuniyet": memnuniyet, "SikayetSayisi": sikayet})
    if name == "Firma Paneli (panel veri)":
        N, T = 25, 6
        ent = np.repeat([f"Firma{i+1}" for i in range(N)], T)
        yil = np.tile(np.arange(2019, 2019 + T), N)
        firm_eff = np.repeat(rng.normal(0, 8, N), T)
        arge = rng.uniform(0, 20, N * T)
        calisan = rng.uniform(10, 200, N * T)
        karlilik = 5 + 0.8 * arge + 0.03 * calisan + firm_eff + rng.normal(0, 3, N * T)
        return pd.DataFrame({"Firma": ent, "Yil": yil, "ArGe": arge,
                             "Calisan": calisan, "Karlilik": karlilik})
    if name == "Aylık Satış (zaman serisi)":
        n = 96
        t = np.arange(n)
        satis = 200 + 2.5 * t + 40 * np.sin(2 * np.pi * t / 12) + rng.normal(0, 12, n)
        tarih = pd.date_range("2018-01-01", periods=n, freq="MS")
        return pd.DataFrame({"Tarih": tarih, "Satis": satis})
    # Bitki Boyu
    groups = np.repeat(["Kontrol", "Gübre A", "Gübre B"], 40)
    base = {"Kontrol": 20, "Gübre A": 25, "Gübre B": 32}
    boy = np.array([base[g] + rng.normal(0, 3) for g in groups])
    isik = rng.choice(["Az", "Çok"], len(groups))
    return pd.DataFrame({"Grup": groups, "Isik": isik, "Boy": boy})


# ---------------- Kenar çubuğu ----------------
with st.sidebar:
    st.title("📊 StatLab")
    st.caption("İstatistiksel analiz ve veri analizi")

    st.subheader("1) Veri kaynağı")
    src = st.radio("Kaynak", ["Dosya yükle", "Örnek veri"], label_visibility="collapsed")

    if src == "Dosya yükle":
        up = st.file_uploader("CSV / Excel / JSON", type=["csv", "xlsx", "xls", "xlsm", "json"])
        if up is not None:
            try:
                df = loaders.load_any(up, up.name)
                st.session_state.df = df
                st.session_state.df_name = up.name
                st.success(f"Yüklendi: {df.shape[0]} satır × {df.shape[1]} sütun")
            except Exception as e:
                st.error(f"Yükleme hatası: {e}")
    else:
        sample = st.selectbox("Örnek veri seti", ["Reklam → Satış", "Öğrenci Notları",
                                                  "Aylık Satış (zaman serisi)", "Firma Paneli (panel veri)",
                                                  "Sağkalım (klinik)", "Anket (Likert 1-5)",
                                                  "Bitki Boyu (gruplu)"])
        if st.button("Örnek veriyi yükle"):
            st.session_state.df = _make_sample(sample)
            st.session_state.df_name = sample
            st.success("Örnek veri yüklendi.")

    st.divider()
    st.subheader("2) Ayarlar")
    alpha = st.select_slider("Anlamlılık düzeyi (α)", options=[0.01, 0.05, 0.10], value=0.05)
    st.session_state.alpha = alpha
    st.session_state.show_interp = st.toggle("💡 Yorum & öneri katmanı", value=True,
                                             help="Kapatınca yalnızca ham parametreler/tablolar görünür; "
                                                  "açınca her sonucun altında ayrı bir yorum-öneri katmanı belirir.")

    st.divider()
    st.subheader("3) Claude (opsiyonel)")
    ok, msg = claude_interpret.available()
    st.session_state.api_key = st.text_input("ANTHROPIC_API_KEY", type="password",
                                              help="Boş bırakılırsa ortam değişkeni kullanılır.")
    st.session_state.model = st.selectbox("Model", ["claude-sonnet-5", "claude-opus-5", "claude-haiku-4-5-20251001"])
    st.caption(("✅ " if ok else "ℹ️ ") + msg)

    st.divider()
    st.subheader("4) Oturum")
    if st.session_state.df is not None:
        sess_bytes = export.save_session(st.session_state.df,
                                         {"df_name": st.session_state.df_name, "alpha": st.session_state.get("alpha", 0.05)})
        st.download_button("💾 Oturumu kaydet (.statlab)", sess_bytes,
                           f"{(st.session_state.df_name or 'statlab')}.statlab", "application/zip")
    sess_up = st.file_uploader("Oturum yükle (.statlab)", type=["statlab", "zip"], key="sess_up")
    if sess_up is not None and st.button("Oturumu geri yükle"):
        try:
            dfr, meta = export.load_session(sess_up.read())
            st.session_state.df = dfr
            st.session_state.df_original = dfr.copy()
            st.session_state.df_name = meta.get("df_name", "oturum")
            st.session_state.df_original_name = st.session_state.df_name
            st.success("Oturum geri yüklendi.")
        except Exception as e:
            st.error(f"Oturum yüklenemedi: {e}")


# ---------------- Ana içerik ----------------
st.title("İstatistiksel Analiz Merkezi")

df = st.session_state.df
if df is not None and st.session_state.get("df_original_name") != st.session_state.df_name:
    st.session_state.df_original = df.copy()
    st.session_state.df_original_name = st.session_state.df_name
if df is None:
    st.info("👈 Çoğu analiz için sol menüden veri yükleyin (CSV/Excel/JSON) veya örnek veri seçin. "
            "**💰 Finans** sekmeleri sembol girerek (yfinance) veri olmadan da çalışır.")
    df = pd.DataFrame()  # boş çerçeve — sekmeler yine de render olsun (özellikle Finans)

num_cols = loaders.numeric_columns(df)
cat_cols = loaders.categorical_columns(df)
alpha = st.session_state.alpha

# Sekmeler kategorilere gruplanır: üstte kategori, altında ilgili sekmeler.
# 17 sekme, gövde bloklarına dokunmadan tek bir tabs[] listesine eşlenir.
tabs = [None] * 39
_cat = st.tabs(["🔎 İçgörü & Yorum", "📁 Veri & Betimleme", "🔗 Keşif", "📉 Regresyon & Modelleme",
                "🧪 Testler", "⏳ Zaman & Panel & Sağkalım", "💰 Finans", "📣 Pazarlama", "🧮 Araçlar"])
with _cat[0]:
    _s = st.tabs(["💡 Otomatik İçgörü", "📋 Veri Profili & Kalite", "✅ Varsayım Bekçisi",
                  "🔗 Nedensellik", "📝 Yönetici Özeti"])
    tabs[26], tabs[27], tabs[28], tabs[29], tabs[30] = _s
_cat = _cat[1:]  # kalan kategoriler eski indeksleriyle eşleşsin
with _cat[0]:
    _s = st.tabs(["📁 Veri", "🧹 Temizle", "📈 Betimsel"])
    tabs[0], tabs[10], tabs[1] = _s
with _cat[1]:
    _s = st.tabs(["🔗 Korelasyon", "🧭 Kümeleme/PCA", "🧬 Faktör Analizi"])
    tabs[2], tabs[13], tabs[17] = _s
with _cat[2]:
    _s = st.tabs(["📉 Regresyon", "🔀 Lojistik", "🔢 Sayım (GLM)", "⚖️ Karşılaştır",
                  "✔️ Doğrulama (CV)", "🔀 Aracılık/Düzenleyici"])
    tabs[3], tabs[4], tabs[14], tabs[11], tabs[35], tabs[36] = _s
with _cat[3]:
    _s = st.tabs(["🧪 ANOVA", "🧫 Hipotez Testi", "⚡ Güç & Güvenilirlik", "🅰️ A/B Test", "📿 Bayesçi"])
    tabs[5], tabs[6], tabs[15], tabs[24], tabs[25] = _s
with _cat[4]:
    _s = st.tabs(["⏳ Zaman Serisi", "🧊 Panel", "🔗 Karışık Model", "🩺 Sağkalım"])
    tabs[7], tabs[12], tabs[18], tabs[16] = _s
with _cat[5]:
    _s = st.tabs(["📈 Getiri & Risk", "⚖️ Optimizasyon", "📊 CAPM/Faktör", "🎲 Monte Carlo/GARCH",
                  "🏦 Makro & Faiz", "🔁 Backtest", "💵 Tahvil"])
    tabs[20], tabs[21], tabs[22], tabs[23], tabs[34], tabs[37], tabs[38] = _s
with _cat[6]:
    _s = st.tabs(["🚰 Funnel", "📅 Kohort/Retention", "🎯 RFM Segment"])
    tabs[31], tabs[32], tabs[33] = _s
with _cat[7]:
    _s = st.tabs(["🧙 Test Seçici", "ƒ Fonksiyon", "📄 Rapor"])
    tabs[19], tabs[8], tabs[9] = _s

# ===== 1. VERİ =====
with tabs[0]:
    st.subheader(f"Veri önizleme — {st.session_state.df_name}")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Satır", df.shape[0])
    c2.metric("Sütun", df.shape[1])
    c3.metric("Sayısal", len(num_cols))
    c4.metric("Kategorik", len(cat_cols))
    st.dataframe(df, use_container_width=True, height=340)

    with st.expander("Sütun tipleri ve eksik değerler"):
        info = pd.DataFrame({
            "Tip": df.dtypes.astype(str),
            "Eksik": df.isna().sum(),
            "Eksik %": (df.isna().mean() * 100).round(1),
            "Benzersiz": df.nunique(),
        })
        st.dataframe(info, use_container_width=True)

# ===== 2. BETİMSEL =====
with tabs[1]:
    st.subheader("Betimsel istatistik")
    if not num_cols:
        st.warning("Sayısal sütun bulunamadı.")
    else:
        desc = descriptive.describe(df, num_cols)
        st.dataframe(desc.style.format(precision=3), use_container_width=True)

        st.subheader("Dağılım ve normallik")
        col = st.selectbox("Değişken seç", num_cols, key="desc_col")
        cc1, cc2 = st.columns([1.3, 1])
        with cc1:
            fig = px.histogram(df, x=col, marginal="box", nbins=30, opacity=0.85)
            fig.update_layout(height=360, margin=dict(t=30, b=10))
            st.plotly_chart(fig, use_container_width=True)
        with cc2:
            import scipy.stats as _sstats
            s = pd.to_numeric(df[col], errors="coerce").dropna()
            qq = _sstats.probplot(s, dist="norm")
            theo, obs = qq[0]
            qfig = go.Figure()
            qfig.add_scatter(x=theo, y=obs, mode="markers", name="Veri")
            qfig.add_scatter(x=theo, y=qq[1][1] + qq[1][0] * theo, mode="lines", name="Normal")
            qfig.update_layout(title="Q-Q grafiği", height=360, margin=dict(t=40, b=10),
                               xaxis_title="Teorik", yaxis_title="Gözlenen")
            st.plotly_chart(qfig, use_container_width=True)

        norm = descriptive.normality(df[col], alpha)
        interp_box(descriptive.interpret_normality(col, norm, alpha))
        claude_expander(f"Değişken '{col}' normallik sonuçları: {norm}", key=f"norm_{col}")

# ===== 3. KORELASYON =====
with tabs[2]:
    corr_mode = st.radio("Analiz", ["Sayısal korelasyon", "Karışık-tip ilişki matrisi"], horizontal=True)
    if corr_mode == "Karışık-tip ilişki matrisi":
        st.subheader("Karışık-tip ilişki matrisi (sayısal + kategorik)")
        st.caption("Tek ölçekte (0–1): sayısal–sayısal |r|, kategorik–kategorik Cramér's V, sayısal–kategorik η. "
                   "Metrikler ile insan/segment profillerinin ilişkisini birlikte gör.")
        acols = st.multiselect("Değişkenler (sayısal + kategorik karışık olabilir)", list(df.columns),
                               default=list(df.columns)[:min(8, df.shape[1])], key="assoc_cols")
        if len(acols) >= 2 and st.button("İlişki matrisini hesapla", type="primary"):
            M, T = association.association_matrix(df, acols)
            hf = px.imshow(M, text_auto=".2f", color_continuous_scale="Viridis", zmin=0, zmax=1,
                           aspect="auto", title="İlişki gücü (0–1)")
            hf.update_layout(height=440, margin=dict(t=40))
            st.plotly_chart(hf, use_container_width=True)
            top = association.top_associations(M, T)
            st.markdown("#### 📊 En güçlü ilişkiler (ham)")
            st.dataframe(top.style.format({"İlişki (0-1)": "{:.3f}"}), use_container_width=True, hide_index=True)
            interp_box(association.interpret_association(top))
            claude_expander(f"Karışık-tip ilişki matrisi en güçlüler:\n{top.to_string(index=False)}", key="assoc_claude")
    elif len(num_cols) < 2:
        st.warning("Korelasyon için en az 2 sayısal sütun gerekir.")
    else:
        st.subheader("Korelasyon analizi")
        method = st.radio("Yöntem", ["pearson", "spearman", "kendall"], horizontal=True,
                          format_func=lambda m: {"pearson": "Pearson (doğrusal)",
                                                 "spearman": "Spearman (sıralı)",
                                                 "kendall": "Kendall τ"}[m])
        chosen = st.multiselect("Değişkenler", num_cols, default=num_cols[: min(6, len(num_cols))])
        if len(chosen) >= 2:
            corr, pmat = correlation.correlation_matrix(df, chosen, method)
            hfig = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r",
                             zmin=-1, zmax=1, aspect="auto")
            hfig.update_layout(height=420, margin=dict(t=30))
            st.plotly_chart(hfig, use_container_width=True)

            st.markdown("**En güçlü ilişkiler**")
            top = correlation.top_correlations(corr, pmat, alpha)
            st.dataframe(top.style.format({"r": "{:.3f}", "p": "{:.4g}"}), use_container_width=True)
            st.caption("Anlamlılık: *** p<.001, ** p<.01, * p<.05, . p<.10")

            st.divider()
            st.markdown("**İkili inceleme (saçılım + regresyon çizgisi)**")
            pc1, pc2 = st.columns(2)
            xx = pc1.selectbox("X", chosen, key="corr_x")
            yy = pc2.selectbox("Y", [c for c in chosen if c != xx], key="corr_y")
            res = correlation.pairwise(df, xx, yy, method, alpha)
            sfig = px.scatter(df, x=xx, y=yy, trendline="ols", opacity=0.7)
            sfig.update_layout(height=380, margin=dict(t=20))
            st.plotly_chart(sfig, use_container_width=True)
            interp_box(correlation.interpret_pairwise(res))
            claude_expander(f"Korelasyon sonucu: {res}", key=f"corr_{xx}_{yy}")

            if len(chosen) >= 3:
                st.divider()
                st.markdown("**Kısmi korelasyon** — üçüncü değişken(ler)in etkisini dışlayarak")
                pcov = st.multiselect("Kontrol edilecek değişken(ler)",
                                      [c for c in chosen if c not in (xx, yy)],
                                      default=[c for c in chosen if c not in (xx, yy)][:1], key="pcorr_cov")
                if pcov and st.button("Kısmi korelasyon hesapla"):
                    pres = correlation.partial(df, xx, yy, pcov, method, alpha)
                    interp_box(correlation.interpret_partial(pres))
                    claude_expander(str(pres), key="pcorr_claude")

# ===== 4. REGRESYON =====
with tabs[3]:
    st.subheader("Çoklu doğrusal regresyon (OLS)")
    if not num_cols:
        st.warning("Regresyon için sayısal sütun gerekir.")
    else:
        rc1, rc2 = st.columns(2)
        y = rc1.selectbox("Bağımlı değişken (Y)", num_cols, key="reg_y")
        x_candidates = [c for c in df.columns if c != y]
        x_vars = rc2.multiselect("Bağımsız değişken(ler) (X)",
                                 x_candidates, default=[c for c in num_cols if c != y][:3])
        cat_selected = [v for v in x_vars if v in cat_cols]
        if cat_selected:
            st.caption(f"Kategorik olarak ele alınacak: {', '.join(cat_selected)}")

        if x_vars and st.button("Regresyonu çalıştır", type="primary"):
            res = regression.fit_ols(df, y, x_vars, categorical=cat_selected, alpha=alpha)
            st.session_state["reg_res"] = res

        res = st.session_state.get("reg_res")
        if res and res.get("y") == y:
            if "error" in res:
                st.error(res["error"])
            else:
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("R²", f"{res['r2']:.3f}")
                m2.metric("Düzeltilmiş R²", f"{res['adj_r2']:.3f}")
                m3.metric("F p-değeri", f"{res['f_pvalue']:.3g}")
                m4.metric("RMSE", f"{res['rmse']:.3f}")

                st.markdown("**Katsayılar**")
                st.dataframe(res["coef_df"].style.format({
                    "Katsayı (β)": "{:.4f}", "Std. Hata": "{:.4f}", "t": "{:.3f}",
                    "p": "{:.4g}", "CI alt": "{:.4f}", "CI üst": "{:.4f}"}),
                    use_container_width=True)

                g1, g2 = st.columns(2)
                with g1:
                    # Gerçek vs tahmin
                    pf = go.Figure()
                    actual = res["fitted"] + res["resid"]
                    pf.add_scatter(x=res["fitted"], y=actual, mode="markers", name="Gözlem", opacity=0.7)
                    lo, hi = float(actual.min()), float(actual.max())
                    pf.add_scatter(x=[lo, hi], y=[lo, hi], mode="lines", name="Mükemmel uyum")
                    pf.update_layout(title="Gerçek vs Tahmin", height=340,
                                     xaxis_title="Tahmin", yaxis_title="Gözlenen", margin=dict(t=40))
                    st.plotly_chart(pf, use_container_width=True)
                with g2:
                    # Artık grafiği
                    rf = px.scatter(x=res["fitted"], y=res["resid"], opacity=0.7,
                                    labels={"x": "Tahmin", "y": "Artık"})
                    rf.add_hline(y=0, line_dash="dash")
                    rf.update_layout(title="Artık grafiği", height=340, margin=dict(t=40))
                    st.plotly_chart(rf, use_container_width=True)

                if not res["vif_df"].empty:
                    with st.expander("Çoklu bağıntı (VIF)"):
                        st.dataframe(res["vif_df"].style.format({"VIF": "{:.2f}"}), use_container_width=True)

                # Basit tahmin aracı (tek sayısal öngörücüler)
                num_predictors = [v for v in x_vars if v not in cat_selected]
                if num_predictors:
                    with st.expander("🎯 Tahmin aracı"):
                        vals = {}
                        pcols = st.columns(min(4, len(num_predictors)))
                        for i, v in enumerate(num_predictors):
                            default = float(pd.to_numeric(df[v], errors="coerce").mean())
                            vals[v] = pcols[i % len(pcols)].number_input(v, value=round(default, 3), key=f"pred_{v}")
                        yhat = regression.predict(res, vals)
                        st.metric(f"Tahmini {y}", f"{yhat:.4f}")

                interp_box(regression.interpret_regression(res))

                num_x = [v for v in x_vars if v in num_cols]
                with st.expander("🔬 İleri tanılar & alternatif regresyonlar"):
                    if num_x:
                        st.markdown("**Etkili gözlemler (Cook uzaklığı / kaldıraç)**")
                        infl = regression_extras.influence(df, y, num_x, alpha)
                        if "error" not in infl:
                            cfig = px.scatter(infl["table"], x="Kaldıraç", y="Cook D",
                                              hover_data=["Gözlem"], title="Cook D vs Kaldıraç")
                            cfig.add_hline(y=infl["cook_thr"], line_dash="dash")
                            cfig.update_layout(height=300, margin=dict(t=40))
                            st.plotly_chart(cfig, use_container_width=True)
                            interp_box(regression_extras.interpret_influence(infl))
                            if len(infl["flagged"]):
                                st.dataframe(infl["flagged"].head(10).style.format(precision=4), use_container_width=True)
                        st.divider()
                        ralt = st.radio("Alternatif regresyon",
                                        ["Robust std. hata (HC3)", "Robust (Huber)", "Kantil", "Adımsal seçim"],
                                        horizontal=True, key="reg_alt")
                        if ralt == "Kantil":
                            qv = st.slider("Kantil (q)", 0.05, 0.95, 0.50, 0.05, key="reg_q")
                        if st.button("Alternatifi çalıştır", key="reg_alt_btn"):
                            if ralt == "Robust std. hata (HC3)":
                                ares = regression_extras.robust_se(df, y, num_x, "HC3", alpha)
                                if "error" not in ares:
                                    st.dataframe(ares["table"].style.format({"Katsayı (β)": "{:.4f}",
                                                 "OLS std. hata": "{:.4f}", "OLS p": "{:.4g}",
                                                 "HC3 std. hata": "{:.4f}", "HC3 p": "{:.4g}"}),
                                                 use_container_width=True, hide_index=True)
                                    interp_box(regression_extras.interpret_robust_se(ares))
                            elif ralt == "Robust (Huber)":
                                ares = regression_extras.robust_regression(df, y, num_x, alpha)
                                if "error" not in ares:
                                    st.dataframe(ares["coef_df"].style.format({"Katsayı (β)": "{:.4f}",
                                                 "Std. Hata": "{:.4f}", "t/z": "{:.3f}", "p": "{:.4g}"}),
                                                 use_container_width=True)
                                    interp_box(regression_extras.interpret_robust(ares))
                            elif ralt == "Kantil":
                                ares = regression_extras.quantile_regression(df, y, num_x, qv, alpha)
                                if "error" not in ares:
                                    st.dataframe(ares["coef_df"].style.format({"Katsayı (β)": "{:.4f}",
                                                 "Std. Hata": "{:.4f}", "t/z": "{:.3f}", "p": "{:.4g}"}),
                                                 use_container_width=True)
                                    interp_box(regression_extras.interpret_quantile(ares))
                            else:
                                ares = regression_extras.stepwise(df, y, num_x, alpha)
                                interp_box(regression_extras.interpret_stepwise(ares))
                    else:
                        st.caption("İleri tanılar için en az bir sayısal öngörücü gerekir.")

                with st.expander("Ayrıntılı statsmodels çıktısı"):
                    st.code(res["summary_text"])
                claude_ctx = (f"Regresyon: {res['y']} ~ {res['x_vars']}. R²={res['r2']:.3f}, "
                              f"düzeltilmiş R²={res['adj_r2']:.3f}, F p={res['f_pvalue']:.4g}. "
                              f"Katsayılar:\n{res['coef_df'].to_string(index=False)}\n"
                              f"Tanılar: {res['diag']}")
                claude_expander(claude_ctx, key="reg_claude")

# ===== 5. LOJİSTİK REGRESYON =====
with tabs[4]:
    lmode = st.radio("Mod", ["İkili (2 sınıf)", "Çok sınıflı (3+ sınıf)", "Sıralı (ordinal)"],
                     horizontal=True, key="logit_mode")
    cand_targets = [c for c in df.columns if df[c].nunique(dropna=True) <= 12]

    if lmode == "Sıralı (ordinal)":
        st.subheader("Sıralı (ordinal) lojistik regresyon")
        st.caption("Hedef sıralı kategorik olduğunda (ör. düşük < orta < yüksek, memnuniyet 1–5).")
        ord_targets = [c for c in df.columns if 3 <= df[c].nunique(dropna=True) <= 12]
        if not ord_targets:
            st.warning("3–12 düzeyli uygun sıralı hedef bulunamadı.")
        else:
            oy = st.selectbox("Sıralı hedef değişken", ord_targets, key="ord_y")
            lv = [str(x) for x in df[oy].dropna().unique().tolist()]
            order = st.multiselect("Sıralamayı seç (düşükten yükseğe, tüm düzeyleri ekleyin)",
                                   lv, default=sorted(lv), key="ord_order")
            ox = st.multiselect("Bağımsız değişken(ler) (sayısal)", [c for c in num_cols if c != oy],
                                default=[c for c in num_cols if c != oy][:3], key="ord_x")
            if ox and len(order) == len(lv) and st.button("Sıralı modeli çalıştır", type="primary"):
                st.session_state["ord_res"] = logistic.fit_ordinal(df, oy, ox, order, alpha)
            elif len(order) != len(lv):
                st.info("Lütfen tüm düzeyleri sıralamaya ekleyin.")
            ores = st.session_state.get("ord_res")
            if ores and ores.get("y") == oy:
                if "error" in ores:
                    st.error(ores["error"])
                else:
                    q1, q2 = st.columns(2)
                    q1.metric("Sözde-R² (McFadden)", f"{ores['pseudo_r2']:.3f}")
                    q2.metric("AIC", f"{ores['aic']:.1f}")
                    st.dataframe(ores["coef_df"].style.format({"Katsayı (β)": "{:.4f}", "Odds Oranı": "{:.3f}",
                                 "OR CI alt": "{:.3f}", "OR CI üst": "{:.3f}", "z": "{:.3f}", "p": "{:.4g}"}),
                                 use_container_width=True)
                    interp_box(logistic.interpret_ordinal(ores))
                    claude_expander(f"Sıralı lojistik {oy} ({' < '.join(ores['order'])}) ~ {ox}, "
                                    f"sözde-R²={ores['pseudo_r2']:.3f}.", key="ord_claude")

    elif lmode == "Çok sınıflı (3+ sınıf)":
        st.subheader("Çok sınıflı (multinomial) lojistik regresyon")
        multi_targets = [c for c in df.columns if 3 <= df[c].nunique(dropna=True) <= 12]
        if not multi_targets:
            st.warning("3–12 sınıflı uygun hedef sütun bulunamadı.")
        else:
            my = st.selectbox("Hedef (3+ sınıflı)", multi_targets, key="mn_y")
            mx = st.multiselect("Bağımsız değişken(ler)", [c for c in df.columns if c != my],
                                default=[c for c in num_cols if c != my][:3], key="mn_x")
            mcat = [v for v in mx if v in cat_cols]
            if mx and st.button("Multinomial modeli çalıştır", type="primary"):
                st.session_state["mn_res"] = logistic.fit_multinomial(df, my, mx, categorical=mcat, alpha=alpha)
            mres = st.session_state.get("mn_res")
            if mres and mres.get("y") == my:
                if "error" in mres:
                    st.error(mres["error"])
                else:
                    q1, q2, q3 = st.columns(3)
                    q1.metric("Sözde-R² (McFadden)", f"{mres['pseudo_r2']:.3f}")
                    q2.metric("Doğruluk", f"{mres['accuracy']:.3f}")
                    q3.metric("Sınıf sayısı", len(mres["classes"]))
                    cf = px.imshow(mres["confusion"].values, text_auto=True, color_continuous_scale="Blues",
                                   x=list(mres["confusion"].columns), y=list(mres["confusion"].index))
                    cf.update_layout(title="Karışıklık matrisi", height=360, margin=dict(t=40))
                    st.plotly_chart(cf, use_container_width=True)
                    for cls, tbl in mres["per_class"].items():
                        st.markdown(f"**{mres['baseline']} → {cls}** katsayıları")
                        st.dataframe(tbl.style.format({"Katsayı (β)": "{:.4f}", "Odds Oranı": "{:.3f}", "p": "{:.4g}"}),
                                     use_container_width=True)
                    interp_box(logistic.interpret_multinomial(mres))
                    with st.expander("Ayrıntılı çıktı"):
                        st.code(mres["summary_text"])
                    claude_expander(f"Multinomial lojistik {my} ({mres['classes']}), baseline={mres['baseline']}, "
                                    f"sözde-R²={mres['pseudo_r2']:.3f}, doğruluk={mres['accuracy']:.3f}.", key="mn_claude")

    elif not cand_targets:
        st.warning("Uygun ikili/kategorik hedef sütun bulunamadı.")
    else:
        st.subheader("İkili lojistik regresyon")
        st.caption("Sonucu iki kategorili (evet/hayır, 0/1, başarılı/başarısız) hedef değişkenler için.")
        lc1, lc2 = st.columns(2)
        ly = lc1.selectbox("Hedef (bağımlı) değişken", cand_targets, key="logit_y")
        levels = df[ly].dropna().unique().tolist()
        pos = lc2.selectbox("Pozitif sınıf (1 olarak kodlanacak)", levels, key="logit_pos")
        lx = st.multiselect("Bağımsız değişken(ler)", [c for c in df.columns if c != ly],
                            default=[c for c in num_cols if c != ly][:3], key="logit_x")
        lcat = [v for v in lx if v in cat_cols]
        thr = st.slider("Sınıflandırma eşiği", 0.05, 0.95, 0.50, 0.05, key="logit_thr")
        if lx and st.button("Lojistik regresyonu çalıştır", type="primary"):
            res = logistic.fit_logit(df, ly, lx, pos, categorical=lcat, alpha=alpha, threshold=thr)
            st.session_state["logit_res"] = res

        res = st.session_state.get("logit_res")
        if res and res.get("y") == ly:
            if "error" in res:
                st.error(res["error"])
            else:
                m = res["metrics"]
                k1, k2, k3, k4 = st.columns(4)
                k1.metric("Sözde-R² (McFadden)", f"{res['pseudo_r2']:.3f}")
                k2.metric("AUC", f"{res['roc']['auc']:.3f}")
                k3.metric("Doğruluk", f"{m['accuracy']:.3f}")
                k4.metric("F1", f"{m['f1']:.3f}")

                st.markdown("**Katsayılar ve odds oranları**")
                st.dataframe(res["coef_df"].style.format({
                    "Katsayı (β)": "{:.4f}", "Odds Oranı": "{:.3f}", "OR CI alt": "{:.3f}",
                    "OR CI üst": "{:.3f}", "z": "{:.3f}", "p": "{:.4g}"}), use_container_width=True)

                gg1, gg2 = st.columns(2)
                with gg1:
                    roc = res["roc"]
                    rf = go.Figure()
                    rf.add_scatter(x=roc["fpr"], y=roc["tpr"], mode="lines",
                                   name=f"AUC={roc['auc']:.3f}", line=dict(width=3))
                    rf.add_scatter(x=[0, 1], y=[0, 1], mode="lines", name="Rastgele",
                                   line=dict(dash="dash", color="gray"))
                    rf.update_layout(title="ROC eğrisi", height=360, xaxis_title="Yanlış poz. oranı",
                                     yaxis_title="Doğru poz. oranı", margin=dict(t=40))
                    st.plotly_chart(rf, use_container_width=True)
                with gg2:
                    cm = res["confusion"]
                    mat = [[cm["TN"], cm["FP"]], [cm["FN"], cm["TP"]]]
                    cf = px.imshow(mat, text_auto=True, color_continuous_scale="Blues",
                                   x=["Tahmin 0", "Tahmin 1"], y=["Gerçek 0", "Gerçek 1"])
                    cf.update_layout(title="Karışıklık matrisi", height=360, margin=dict(t=40))
                    st.plotly_chart(cf, use_container_width=True)

                num_lp = [v for v in lx if v not in lcat]
                if num_lp:
                    with st.expander("🎯 Olasılık tahmini"):
                        vals = {}
                        pcols = st.columns(min(4, len(num_lp)))
                        for i, v in enumerate(num_lp):
                            dv = float(pd.to_numeric(df[v], errors="coerce").mean())
                            vals[v] = pcols[i % len(pcols)].number_input(v, value=round(dv, 3), key=f"lpred_{v}")
                        proba = logistic.predict_proba(res, vals)
                        st.metric(f"P({ly} = {pos})", f"{proba:.1%}")

                interp_box(logistic.interpret_logit(res))
                with st.expander("Ayrıntılı çıktı"):
                    st.code(res["summary_text"])
                claude_expander(f"Lojistik regresyon {ly}={pos} ~ {lx}. "
                                f"Sözde-R²={res['pseudo_r2']:.3f}, AUC={res['roc']['auc']:.3f}. "
                                f"Katsayılar:\n{res['coef_df'].to_string(index=False)}\nMetrikler: {m}",
                                key="logit_claude")

# ===== 6. ANOVA =====
with tabs[5]:
    st.subheader("Varyans analizi (ANOVA)")
    if not cat_cols or not num_cols:
        st.warning("ANOVA için en az bir kategorik (grup) ve bir sayısal sütun gerekir.")
    else:
        mode = st.radio("Tür", ["Tek yönlü", "Welch (eşit olmayan varyans)", "Çift yönlü", "Tekrarlı ölçüm",
                                "ANCOVA (kovaryas kontrollü)", "MANOVA (çok değişkenli)"], horizontal=True)

        if mode == "MANOVA (çok değişkenli)":
            st.caption("Birden çok bağımlı değişkeni BİRLİKTE test eder — ayrı ANOVA'ların şişirdiği yanlış-pozitif "
                       "riskini kontrol eder.")
            mdvs = st.multiselect("Bağımlı değişkenler (≥2 sayısal)", num_cols,
                                  default=num_cols[:min(3, len(num_cols))], key="man_dvs")
            mg = st.selectbox("Grup değişkeni", cat_cols, key="man_g")
            if len(mdvs) >= 2 and st.button("MANOVA çalıştır", type="primary", key="manova_btn"):
                res = posthoc.manova(df, mdvs, mg, alpha)
                if "error" in res:
                    st.error(res["error"])
                else:
                    k1, k2 = st.columns(2)
                    k1.metric("Wilks' λ p", f"{res['wilks']['p']:.3g}")
                    k2.metric("Pillai p", f"{res['pillai']['p']:.3g}")
                    interp_box(posthoc.interpret_manova(res))
                    claude_expander(str(res), key="manova_claude")

        elif mode == "ANCOVA (kovaryas kontrollü)":
            st.caption("Grup etkisini, ilgili bir ortak değişkeni (kovaryas) sabitleyerek ölçer → daha yüksek kesinlik/güç.")
            av = st.selectbox("Bağımlı değişken", num_cols, key="anc_v")
            ag = st.selectbox("Grup değişkeni", cat_cols, key="anc_g")
            acov = st.multiselect("Kovaryas(lar) (sayısal)", [c for c in num_cols if c != av],
                                  default=[c for c in num_cols if c != av][:1], key="anc_cov")
            if acov and st.button("ANCOVA çalıştır", type="primary", key="ancova_btn"):
                res = experiments.ancova(df, av, ag, acov, alpha)
                st.markdown("#### 📊 Ortalamalar: ham → düzeltilmiş")
                mt = pd.DataFrame({"Grup": list(res["adj_means"].keys()),
                                   "Ham": [res["raw_means"].get(k) for k in res["adj_means"]],
                                   "Düzeltilmiş": list(res["adj_means"].values())})
                st.dataframe(mt.style.format({"Ham": "{:.3f}", "Düzeltilmiş": "{:.3f}"}),
                             use_container_width=True, hide_index=True)
                interp_box(experiments.interpret_ancova(res))
                claude_expander(str(res), key="ancova_claude")

        elif mode == "Tekrarlı ölçüm":
            st.caption("Aynı deneklerin 3+ koşulda ölçümü (within-subjects). Her koşul bir sayısal sütundur.")
            rm_cols = st.multiselect("Koşul sütunları (≥3)", num_cols,
                                     default=num_cols[:3] if len(num_cols) >= 3 else num_cols, key="rm_cols")
            if len(rm_cols) >= 3 and st.button("RM-ANOVA çalıştır", type="primary", key="rmanova"):
                res = posthoc.rm_anova_from_wide(df, rm_cols, alpha)
                if "error" in res:
                    st.error(res["error"])
                else:
                    bf = px.box(df[rm_cols], points="all"); bf.update_layout(height=340, margin=dict(t=20))
                    st.plotly_chart(bf, use_container_width=True)
                    interp_box(posthoc.interpret_rm(res))
                    if res["p"] < alpha:
                        nem = posthoc.nemenyi_friedman(df[rm_cols].apply(pd.to_numeric, errors="coerce").dropna())
                        st.markdown("**Nemenyi post-hoc — koşul ikilileri**")
                        st.dataframe(nem["pairs"].style.format({"p": "{:.4g}"}), use_container_width=True)
                    claude_expander(f"RM-ANOVA {rm_cols}: F={res['F']:.3f}, p={res['p']:.4g}.", key="rm_claude")

        elif mode == "Welch (eşit olmayan varyans)":
            val = st.selectbox("Bağımlı (sayısal) değişken", num_cols, key="anova_val_w")
            grp = st.selectbox("Grup değişkeni", cat_cols, key="anova_grp_w")
            if st.button("Welch ANOVA çalıştır", type="primary", key="welch"):
                res = posthoc.welch_anova(df, val, grp, alpha)
                bf = px.box(df, x=grp, y=val, points="all"); bf.update_layout(height=340, margin=dict(t=20))
                st.plotly_chart(bf, use_container_width=True)
                interp_box(posthoc.interpret_welch(res))
                if res["p"] < alpha:
                    gh = posthoc.games_howell(df, val, grp, alpha)
                    st.markdown("**Games-Howell post-hoc**")
                    st.dataframe(gh["table"].style.format({"Ort. fark": "{:.3f}", "t": "{:.3f}", "p": "{:.4g}"}),
                                 use_container_width=True)
                    interp_box(posthoc.interpret_posthoc(gh))
                claude_expander(f"Welch ANOVA {val}~{grp}: F={res['F']:.3f}, p={res['p']:.4g}.", key="welch_claude")

        elif mode == "Tek yönlü":
            val = st.selectbox("Bağımlı (sayısal) değişken", num_cols, key="anova_val")
            grp = st.selectbox("Grup değişkeni", cat_cols, key="anova_grp")
            if st.button("ANOVA çalıştır", type="primary", key="anova1"):
                res = anova.one_way(df, val, grp, alpha)
                if "error" in res:
                    st.error(res["error"])
                else:
                    bf = px.box(res["data"], x=grp, y=val, points="all")
                    bf.update_layout(height=360, margin=dict(t=20))
                    st.plotly_chart(bf, use_container_width=True)
                    st.dataframe(res["group_stats"].style.format({"Ortalama": "{:.3f}", "Std": "{:.3f}"}),
                                 use_container_width=True)
                    interp_box(anova.interpret_one_way(res))
                    if "tukey" in res:
                        st.markdown("**Tukey HSD — ikili karşılaştırmalar (eşit varyans)**")
                        st.dataframe(res["tukey"], use_container_width=True)
                    if res["p"] < alpha and res["k"] > 2:
                        with st.expander("Games-Howell post-hoc (varyanslar eşit değilse)"):
                            gh = posthoc.games_howell(df, val, grp, alpha)
                            st.dataframe(gh["table"].style.format({"Ort. fark": "{:.3f}", "t": "{:.3f}", "p": "{:.4g}"}),
                                         use_container_width=True)
                    claude_expander(f"Tek yönlü ANOVA {val}~{grp}: F={res['f']:.3f}, p={res['p']:.4g}, "
                                    f"eta2={res['eta2']:.3f}. Grup ortalamaları:\n{res['group_stats'].to_string()}",
                                    key="anova1_claude")
        else:
            val = st.selectbox("Bağımlı (sayısal) değişken", num_cols, key="anova_val")
            fa = st.selectbox("Faktör A", cat_cols, key="anova_fa")
            fb = st.selectbox("Faktör B", [c for c in cat_cols if c != fa], key="anova_fb")
            if st.button("ANOVA çalıştır", type="primary", key="anova2"):
                res = anova.two_way(df, val, fa, fb, alpha)
                if "error" in res:
                    st.error(res["error"])
                else:
                    means = df.groupby([fa, fb])[val].mean().reset_index()
                    lf = px.line(means, x=fa, y=val, color=fb, markers=True)
                    lf.update_layout(title="Etkileşim grafiği", height=340, margin=dict(t=40))
                    st.plotly_chart(lf, use_container_width=True)
                    st.dataframe(res["anova_table"].style.format(precision=4), use_container_width=True)
                    interp_box(anova.interpret_two_way(res))
                    claude_expander(f"Çift yönlü ANOVA tablosu:\n{res['anova_table'].to_string()}", key="anova2_claude")

# ===== 7. HİPOTEZ TESTİ =====
with tabs[6]:
    st.subheader("Hipotez testleri")
    test = st.selectbox("Test seç", [
        "Tek örneklem t-testi", "Bağımsız örneklem t-testi", "Eşleştirilmiş t-testi",
        "Mann-Whitney U", "Wilcoxon işaretli sıra", "Kruskal-Wallis", "Friedman",
        "Ki-kare bağımsızlık", "Fisher exact (2×2)", "McNemar (eşleştirilmiş kategorik)",
        "Ki-kare uyum iyiliği", "Nokta-çift serili korelasyon", "Binom testi (tek oran)",
        "Oran z-testi (iki grup)", "Bartlett (varyans homojenliği)", "Eşdeğerlik (TOST)",
    ])
    alt_map = {"İki yönlü": "two-sided", "Küçük (<)": "less", "Büyük (>)": "greater"}

    if test == "Tek örneklem t-testi":
        col = st.selectbox("Değişken", num_cols)
        mu = st.number_input("Test edilecek ortalama (µ₀)", value=0.0)
        alt = st.radio("Alternatif hipotez", list(alt_map), horizontal=True)
        if st.button("Test et", type="primary"):
            res = hypothesis.one_sample_t(df[col], mu, alpha, alt_map[alt])
            interp_box(hypothesis.interpret(res))
            claude_expander(str(res), key="h1")

    elif test in ("Bağımsız örneklem t-testi", "Mann-Whitney U"):
        st.caption("İki grubu bir kategorik değişkene göre karşılaştırın.")
        val = st.selectbox("Sayısal değişken", num_cols, key="ind_val")
        grp = st.selectbox("Grup değişkeni (2 düzeyli)", cat_cols, key="ind_grp") if cat_cols else None
        if grp:
            levels = df[grp].dropna().unique().tolist()
            if len(levels) < 2:
                st.warning("Grup değişkeninde en az 2 düzey gerekir.")
            else:
                g1, g2 = st.columns(2)
                l1 = g1.selectbox("1. grup", levels, key="ind_l1")
                l2 = g2.selectbox("2. grup", [x for x in levels if x != l1], key="ind_l2")
                alt = st.radio("Alternatif", list(alt_map), horizontal=True, key="ind_alt")
                if st.button("Test et", type="primary"):
                    a = df.loc[df[grp] == l1, val]
                    b = df.loc[df[grp] == l2, val]
                    if test == "Bağımsız örneklem t-testi":
                        res = hypothesis.independent_t(a, b, alpha, None, alt_map[alt], (str(l1), str(l2)))
                    else:
                        res = hypothesis.mann_whitney(a, b, alpha, alt_map[alt])
                    bf = px.box(df[df[grp].isin([l1, l2])], x=grp, y=val, points="all")
                    bf.update_layout(height=320, margin=dict(t=20))
                    st.plotly_chart(bf, use_container_width=True)
                    interp_box(hypothesis.interpret(res))
                    claude_expander(str(res), key="h2")
        else:
            st.warning("Kategorik grup sütunu yok.")

    elif test == "Eşleştirilmiş t-testi":
        st.caption("Aynı denekten iki ölçüm (öncesi/sonrası).")
        c1, c2 = st.columns(2)
        a = c1.selectbox("1. ölçüm", num_cols, key="pair_a")
        b = c2.selectbox("2. ölçüm", [c for c in num_cols if c != a], key="pair_b")
        alt = st.radio("Alternatif", list(alt_map), horizontal=True, key="pair_alt")
        if st.button("Test et", type="primary"):
            res = hypothesis.paired_t(df[a], df[b], alpha, alt_map[alt], (a, b))
            interp_box(hypothesis.interpret(res))
            claude_expander(str(res), key="h3")

    elif test == "Wilcoxon işaretli sıra":
        st.caption("Eşleştirilmiş t-testinin parametrik olmayan alternatifi (öncesi/sonrası).")
        c1, c2 = st.columns(2)
        a = c1.selectbox("1. ölçüm", num_cols, key="wil_a")
        b = c2.selectbox("2. ölçüm", [c for c in num_cols if c != a], key="wil_b")
        alt = st.radio("Alternatif", list(alt_map), horizontal=True, key="wil_alt")
        if st.button("Test et", type="primary"):
            res = hypothesis.wilcoxon_signed(df[a], df[b], alpha, alt_map[alt], (a, b))
            interp_box(hypothesis.interpret(res))
            claude_expander(str(res), key="h_wil")

    elif test == "Friedman":
        st.caption("3+ tekrarlı ölçümün parametrik olmayan karşılaştırması (aynı denekler).")
        cols_sel = st.multiselect("Ölçüm sütunları (≥3)", num_cols,
                                  default=num_cols[:3] if len(num_cols) >= 3 else num_cols, key="fr_cols")
        if len(cols_sel) >= 3 and st.button("Test et", type="primary"):
            res = hypothesis.friedman([df[c] for c in cols_sel], cols_sel, alpha)
            if "error" in res:
                st.error(res["error"])
            else:
                mf = px.box(df[cols_sel], points="all"); mf.update_layout(height=320, margin=dict(t=20))
                st.plotly_chart(mf, use_container_width=True)
                interp_box(hypothesis.interpret(res))
                claude_expander(str(res), key="h_fr")

    elif test == "Kruskal-Wallis":
        val = st.selectbox("Sayısal değişken", num_cols, key="kw_val")
        grp = st.selectbox("Grup değişkeni", cat_cols, key="kw_grp") if cat_cols else None
        if grp and st.button("Test et", type="primary"):
            groups, labels = [], []
            for k, g in df.groupby(grp):
                groups.append(g[val]); labels.append(str(k))
            res = hypothesis.kruskal(groups, labels, alpha)
            bf = px.box(df, x=grp, y=val, points="all"); bf.update_layout(height=320, margin=dict(t=20))
            st.plotly_chart(bf, use_container_width=True)
            interp_box(hypothesis.interpret(res))
            if res["p"] < alpha and res["k"] > 2:
                st.markdown("**Dunn post-hoc — ikili karşılaştırmalar (Holm düzeltmeli)**")
                dn = posthoc.dunn(df, val, grp, "holm", alpha)
                st.dataframe(dn["pairs"].style.format({"p": "{:.4g}"}), use_container_width=True)
            claude_expander(str(res), key="h4")

    elif test == "Fisher exact (2×2)":
        if len(cat_cols) < 2:
            st.warning("En az 2 kategorik sütun gerekir.")
        else:
            c1, c2 = st.columns(2)
            a = c1.selectbox("1. kategorik (2 düzeyli)", cat_cols, key="fish_a")
            b = c2.selectbox("2. kategorik (2 düzeyli)", [c for c in cat_cols if c != a], key="fish_b")
            if st.button("Test et", type="primary"):
                res = hypothesis.fisher_exact(df, a, b, alpha)
                if "error" in res:
                    st.error(res["error"])
                else:
                    st.dataframe(res["table"], use_container_width=True)
                    interp_box(hypothesis.interpret(res))
                    claude_expander(str({k: v for k, v in res.items() if k != "table"}), key="h_fish")

    elif test == "McNemar (eşleştirilmiş kategorik)":
        st.caption("Aynı deneklerde öncesi/sonrası ikili sonuç (ör. tedavi öncesi/sonrası olumlu-olumsuz).")
        if len(cat_cols) < 2:
            st.warning("En az 2 kategorik (2 düzeyli) sütun gerekir.")
        else:
            c1, c2 = st.columns(2)
            a = c1.selectbox("Öncesi", cat_cols, key="mc_a")
            b = c2.selectbox("Sonrası", [c for c in cat_cols if c != a], key="mc_b")
            if st.button("Test et", type="primary"):
                res = hypothesis.mcnemar_test(df, a, b, alpha)
                if "error" in res:
                    st.error(res["error"])
                else:
                    st.dataframe(res["table"], use_container_width=True)
                    interp_box(hypothesis.interpret(res))
                    claude_expander(str({k: v for k, v in res.items() if k != "table"}), key="h_mc")

    elif test == "Ki-kare uyum iyiliği":
        st.caption("Bir kategorik değişkenin gözlenen dağılımını beklenen (varsayılan eşit) dağılımla karşılaştırır.")
        if not cat_cols:
            st.warning("Kategorik sütun gerekir.")
        else:
            gcol = st.selectbox("Kategorik değişken", cat_cols, key="gof_col")
            counts = df[gcol].value_counts().sort_index()
            st.write("Gözlenen frekanslar:", counts.to_dict())
            exp_mode = st.radio("Beklenen dağılım", ["Eşit", "Elle gir"], horizontal=True)
            expected = None
            if exp_mode == "Elle gir":
                etext = st.text_input("Beklenen oranlar (virgülle, sırayla)",
                                      value=",".join(["1"] * len(counts)))
                try:
                    expected = [float(x) for x in etext.split(",")]
                except Exception:
                    expected = None
            if st.button("Test et", type="primary"):
                res = hypothesis.chi2_goodness(counts.values.tolist(), alpha, expected, list(counts.index.astype(str)))
                interp_box(hypothesis.interpret(res))
                claude_expander(str({k: v for k, v in res.items() if k not in ("observed", "expected")}), key="h_gof")

    elif test == "Nokta-çift serili korelasyon":
        st.caption("İkili (0/1) değişken ile sürekli değişken arasındaki ilişki.")
        c1, c2 = st.columns(2)
        bcol = c1.selectbox("İkili değişken", num_cols, key="pb_b")
        ccol = c2.selectbox("Sürekli değişken", [c for c in num_cols if c != bcol], key="pb_c")
        if st.button("Test et", type="primary"):
            res = hypothesis.point_biserial(df[bcol], df[ccol], alpha)
            if "error" in res:
                st.error(res["error"])
            else:
                interp_box(hypothesis.interpret(res))
                claude_expander(str(res), key="h_pb")

    elif test == "Binom testi (tek oran)":
        st.caption("Bir oranı beklenen bir değerle karşılaştırır (ör. yazı-tura adil mi?).")
        c1, c2, c3 = st.columns(3)
        succ = c1.number_input("Başarı sayısı", 0, 10_000_000, 60)
        ntot = c2.number_input("Toplam deneme", 1, 10_000_000, 100)
        p0 = c3.number_input("Beklenen oran (p₀)", 0.0, 1.0, 0.5, 0.05)
        alt = st.radio("Alternatif", list(alt_map), horizontal=True, key="bin_alt")
        if st.button("Test et", type="primary"):
            res = hypothesis.binomial_test(int(succ), int(ntot), p0, alpha, alt_map[alt])
            interp_box(hypothesis.interpret(res))
            claude_expander(str({k: v for k, v in res.items() if k != "ci"}), key="h_bin")

    elif test == "Oran z-testi (iki grup)":
        st.caption("İki grubun başarı oranlarını karşılaştırır.")
        c1, c2, c3, c4 = st.columns(4)
        s1 = c1.number_input("Grup 1 başarı", 0, 10_000_000, 40)
        n1 = c2.number_input("Grup 1 n", 1, 10_000_000, 100)
        s2 = c3.number_input("Grup 2 başarı", 0, 10_000_000, 55)
        n2 = c4.number_input("Grup 2 n", 1, 10_000_000, 100)
        alt = st.radio("Alternatif", list(alt_map), horizontal=True, key="prop_alt")
        if st.button("Test et", type="primary"):
            res = hypothesis.proportions_z([int(s1), int(s2)], [int(n1), int(n2)], alpha, alt_map[alt])
            interp_box(hypothesis.interpret(res))
            claude_expander(str(res), key="h_prop")

    elif test == "Eşdeğerlik (TOST)":
        st.caption("İki grubun pratikte EŞDEĞER olduğunu kanıtlar (fark önemsiz bandın içinde). Normal test bunu yapamaz.")
        val = st.selectbox("Sayısal değişken", num_cols, key="tost_val")
        grp = st.selectbox("Grup (2 düzeyli)", cat_cols, key="tost_grp") if cat_cols else None
        bc1, bc2 = st.columns(2)
        low = bc1.number_input("Eşdeğerlik alt sınır", value=-1.0)
        upp = bc2.number_input("Eşdeğerlik üst sınır", value=1.0)
        if grp:
            lv = df[grp].dropna().unique().tolist()
            if len(lv) >= 2 and st.button("Test et", type="primary"):
                res = experiments.tost(df.loc[df[grp] == lv[0], val], df.loc[df[grp] == lv[1], val], low, upp, alpha)
                interp_box(experiments.interpret_tost(res))
                claude_expander(str(res), key="h_tost")
        else:
            st.warning("Grup sütunu yok.")

    elif test == "Bartlett (varyans homojenliği)":
        st.caption("Grupların varyanslarının eşit olup olmadığını sınar (Levene'ye normal-duyarlı alternatif).")
        val = st.selectbox("Sayısal değişken", num_cols, key="bart_val")
        grp = st.selectbox("Grup değişkeni", cat_cols, key="bart_grp") if cat_cols else None
        if grp and st.button("Test et", type="primary"):
            groups, labels = [], []
            for k, g in df.groupby(grp):
                groups.append(g[val]); labels.append(str(k))
            res = hypothesis.bartlett_test(groups, labels, alpha)
            interp_box(hypothesis.interpret(res))
            claude_expander(str(res), key="h_bart")

    elif test == "Ki-kare bağımsızlık":
        if len(cat_cols) < 2:
            st.warning("Ki-kare için en az 2 kategorik sütun gerekir.")
        else:
            c1, c2 = st.columns(2)
            a = c1.selectbox("1. kategorik", cat_cols, key="chi_a")
            b = c2.selectbox("2. kategorik", [c for c in cat_cols if c != a], key="chi_b")
            if st.button("Test et", type="primary"):
                res = hypothesis.chi_square(df, a, b, alpha)
                if "error" in res:
                    st.error(res["error"])
                else:
                    st.markdown("**Gözlenen çapraz tablo**")
                    st.dataframe(res["table"], use_container_width=True)
                    interp_box(hypothesis.interpret(res))
                    claude_expander(str({k: v for k, v in res.items() if k not in ("table", "expected")}), key="h5")

# ===== 8. ZAMAN SERİSİ =====
with tabs[7]:
    st.subheader("Zaman serisi analizi")
    if not num_cols:
        st.warning("Zaman serisi için sayısal bir değer sütunu gerekir.")
    else:
        tc1, tc2 = st.columns(2)
        val = tc1.selectbox("Değer (sayısal) sütunu", num_cols, key="ts_val")
        time_opts = ["(sıra numarası)"] + list(df.columns)
        tsel = tc2.selectbox("Zaman/tarih sütunu (opsiyonel)", time_opts, key="ts_time")
        tcol = None if tsel == "(sıra numarası)" else tsel

        oc1, oc2, oc3 = st.columns(3)
        period = oc1.number_input("Mevsim periyodu", min_value=2, max_value=365, value=12, step=1,
                                  help="Aylık veri için 12, çeyreklik için 4, haftalık için 7.")
        steps = oc2.number_input("Tahmin ufku (dönem)", min_value=1, max_value=120, value=12, step=1)
        method = oc3.selectbox("Tahmin yöntemi", ["Holt-Winters", "ARIMA", "Otomatik (auto-ARIMA)"])
        model_type = st.radio("Ayrıştırma modeli", ["additive", "multiplicative"], horizontal=True,
                              format_func=lambda x: "Toplamsal" if x == "additive" else "Çarpımsal")

        if st.button("Zaman serisini analiz et", type="primary"):
            s = timeseries.build_series(df, val, tcol)
            if len(s) < 4:
                st.error("Yeterli gözlem yok.")
            else:
                ov = timeseries.overview(s)
                stat = timeseries.stationarity(s, alpha)
                dec = timeseries.decompose(s, int(period), model_type)
                ac = timeseries.autocorr(s)
                if method == "Holt-Winters":
                    seas = "add" if model_type == "additive" else "mul"
                    fc = timeseries.forecast_holt(s, int(steps), int(period), seasonal=seas)
                elif method == "Otomatik (auto-ARIMA)":
                    with st.spinner("En iyi ARIMA dereceleri aranıyor..."):
                        fc = timeseries.auto_arima(s, max_p=3, max_q=3, seasonal=(int(period) > 1 and len(s) >= 2 * int(period)),
                                                   m=int(period), steps=int(steps), alpha=alpha)
                else:
                    fc = timeseries.forecast_arima(s, (1, 1, 1), int(steps), alpha)
                st.session_state["ts_bundle"] = {"s": s, "ov": ov, "stat": stat, "dec": dec,
                                                 "ac": ac, "fc": fc, "val": val}

        bundle = st.session_state.get("ts_bundle")
        if bundle and bundle.get("val") == val:
            s, ov, stat, dec, ac, fc = (bundle[k] for k in ("s", "ov", "stat", "dec", "ac", "fc"))
            xax = list(range(len(s))) if tcol is None else list(s.index)

            lf = go.Figure()
            lf.add_scatter(x=xax, y=s.values, mode="lines", name="Gözlem")
            lf.add_scatter(x=xax, y=ov["trend_line"], mode="lines", name="Trend", line=dict(dash="dash"))
            if "forecast" in fc:
                fx = list(range(len(s), len(s) + len(fc["forecast"]))) if tcol is None else \
                    list(pd.date_range(s.index[-1], periods=len(fc["forecast"]) + 1, freq="D")[1:])
                lf.add_scatter(x=fx, y=fc["forecast"], mode="lines", name="Tahmin", line=dict(color="#dc2626", width=3))
                if "ci_low" in fc:
                    lf.add_scatter(x=fx, y=fc["ci_high"], mode="lines", line=dict(width=0), showlegend=False)
                    lf.add_scatter(x=fx, y=fc["ci_low"], mode="lines", fill="tonexty", line=dict(width=0),
                                   fillcolor="rgba(220,38,38,.15)", name="%95 GA")
            lf.update_layout(title=f"{val} — seri, trend ve tahmin", height=380, margin=dict(t=40))
            st.plotly_chart(lf, use_container_width=True)

            interp_box(timeseries.interpret_overview(ov, stat, dec, alpha))
            if "error" in fc:
                st.warning(fc["error"])
            else:
                interp_box(timeseries.interpret_forecast(fc, float(s.iloc[-1])))
                if "candidates" in fc:
                    st.success(f"Otomatik seçilen model: **{fc['method']}** "
                               f"(fark alma d={fc['d_selected']}, {fc['n_tried']} model denendi, AIC={fc['aic']:.1f}).")
                    with st.expander("En iyi 5 aday (AIC'e göre)"):
                        st.dataframe(fc["candidates"], use_container_width=True)

            if "error" not in dec:
                with st.expander("Mevsimsel ayrıştırma (trend / mevsimsel / artık)"):
                    df_dec = pd.DataFrame({"Trend": dec["trend"], "Mevsimsel": dec["seasonal"],
                                           "Artık": dec["resid"]})
                    st.line_chart(df_dec)
            with st.expander("Otokorelasyon (ACF / PACF)"):
                acf_df = pd.DataFrame({"Gecikme": range(len(ac["acf"])), "ACF": ac["acf"], "PACF": ac["pacf"]})
                st.bar_chart(acf_df.set_index("Gecikme"))
                st.caption("ACF/PACF, ARIMA'nın p ve q derecelerini seçmede yardımcı olur.")

            claude_expander(f"Zaman serisi '{val}': eğim={ov['slope']:.4f}, ADF p={stat.get('adf_p')}, "
                            f"mevsimsel güç={dec.get('seasonal_strength')}, tahmin yöntemi={fc.get('method')}.",
                            key="ts_claude")

# ===== 9. FONKSİYON =====
with tabs[8]:
    fmode = st.radio("Mod", ["Fonksiyon çiz", "Veriye eğri uydur"], horizontal=True)

    if fmode == "Veriye eğri uydur":
        st.subheader("Veriye eğri / model uydur")
        if len(num_cols) < 2:
            st.warning("Eğri uydurmak için en az 2 sayısal sütun gerekir.")
        else:
            cc1, cc2 = st.columns(2)
            fx = cc1.selectbox("X (bağımsız)", num_cols, key="fit_x")
            fy = cc2.selectbox("Y (bağımlı)", [c for c in num_cols if c != fx], key="fit_y")
            mc1, mc2 = st.columns([2, 1])
            fmodel = mc1.selectbox("Model türü", ["poly", "exp", "log", "power"],
                                   format_func=lambda m: {"poly": "Polinom", "exp": "Üstel (a·e^bx+c)",
                                                          "log": "Logaritmik (a·ln x+b)", "power": "Güç yasası (a·x^b)"}[m])
            fdeg = mc2.number_input("Polinom derecesi", 1, 6, 2, disabled=(fmodel != "poly"))
            if st.button("Eğriyi uydur", type="primary"):
                res = fitting.fit_curve(df[fx], df[fy], fmodel, int(fdeg))
                if "error" in res:
                    st.error(res["error"])
                else:
                    sf = go.Figure()
                    sf.add_scatter(x=res["x"], y=res["y"], mode="markers", name="Veri", opacity=0.6)
                    sf.add_scatter(x=res["x_line"], y=res["y_line"], mode="lines", name="Uydurulan eğri",
                                   line=dict(color="#dc2626", width=3))
                    sf.update_layout(title=res["equation"], height=380, xaxis_title=fx,
                                     yaxis_title=fy, margin=dict(t=50))
                    st.plotly_chart(sf, use_container_width=True)
                    mm1, mm2, mm3 = st.columns(3)
                    mm1.metric("R²", f"{res['r2']:.4f}")
                    mm2.metric("Düz. R²", f"{res['adj_r2']:.4f}")
                    mm3.metric("RMSE", f"{res['rmse']:.4f}")
                    interp_box(fitting.interpret_fit(res))
                    st.markdown("**Model türlerini karşılaştır**")
                    cmp = fitting.compare_fits(df[fx], df[fy],
                                               [("poly", 1), ("poly", 2), ("poly", 3), ("exp", 0), ("log", 0), ("power", 0)])
                    st.dataframe(cmp.style.format({"R²": "{:.4f}", "Düz. R²": "{:.4f}", "RMSE": "{:.4f}"}),
                                 use_container_width=True)
                    claude_expander(f"Eğri uydurma {fy}~{fx}: {res['equation']}, R²={res['r2']:.3f}", key="fit_claude")

    else:
        st.subheader("Kendi regresyon / matematik fonksiyonunu gir")
        st.caption("Örnekler:  `y = 2*x + 3`   ·   `y = 1.5*x**2 - 4*x + 10`   ·   `y = 10*exp(-0.3*x)`")
        expr_text = st.text_input("Fonksiyon", value="y = 2*x + 3")
        fc1, fc2, fc3 = st.columns(3)
        xmin = fc1.number_input("Alt sınır", value=-10.0)
        xmax = fc2.number_input("Üst sınır", value=10.0)
        npts = fc3.number_input("Nokta sayısı", value=200, min_value=10, max_value=2000, step=10)

        if st.button("Analiz et & çiz", type="primary"):
            parsed = function_parser.parse_function(expr_text)
            if "error" in parsed:
                st.error(parsed["error"])
            else:
                variables = parsed["variables"]
                if len(variables) == 1:
                    v = variables[0]
                    xr = np.linspace(xmin, xmax, int(npts))
                    table = function_parser.evaluate(parsed, {v: xr})
                    lf = px.line(table, x=v, y=parsed["lhs"])
                    lf.update_layout(height=380, margin=dict(t=20))
                    st.plotly_chart(lf, use_container_width=True)
                    interp_box(function_parser.analyze_function(parsed))
                    with st.expander("Değer tablosu"):
                        st.dataframe(table, use_container_width=True, height=260)
                    st.download_button("Tabloyu CSV indir", table.to_csv(index=False).encode("utf-8"),
                                       "fonksiyon.csv", "text/csv")
                else:
                    st.info(f"Çok değişkenli fonksiyon ({', '.join(variables)}). Matematiksel yorum aşağıda; "
                            "grafik için tek değişkenli girin veya veriyle Regresyon sekmesini kullanın.")
                    interp_box(function_parser.analyze_function(parsed))
                claude_expander(f"Kullanıcı fonksiyonu: {expr_text} → {parsed.get('expr_str')}", key="func_claude")

# ===== 10. RAPOR =====
with tabs[9]:
    st.subheader("PDF rapor oluştur")
    st.caption("Analizlerini tek bir profesyonel PDF dosyasında topla — grafikler ve Türkçe yorumlarla.")

    reg_res = st.session_state.get("reg_res")
    logit_res = st.session_state.get("logit_res")
    ts_bundle = st.session_state.get("ts_bundle")

    st.markdown("**Rapora eklenecek bölümler**")
    inc_desc = st.checkbox("Betimsel istatistik + dağılım", value=True)
    inc_corr = st.checkbox("Korelasyon matrisi", value=len(num_cols) >= 2, disabled=len(num_cols) < 2)
    inc_reg = st.checkbox(f"Regresyon {'(' + reg_res['y'] + ')' if reg_res and 'y' in reg_res else '(önce çalıştırın)'}",
                          value=bool(reg_res and "error" not in (reg_res or {})), disabled=not (reg_res and "error" not in reg_res))
    inc_logit = st.checkbox(f"Lojistik regresyon {'(' + logit_res['y'] + ')' if logit_res and 'y' in logit_res else '(önce çalıştırın)'}",
                            value=bool(logit_res and "error" not in (logit_res or {})), disabled=not (logit_res and "error" not in logit_res))
    inc_ts = st.checkbox(f"Zaman serisi {'(' + ts_bundle['val'] + ')' if ts_bundle else '(önce çalıştırın)'}",
                         value=bool(ts_bundle), disabled=not ts_bundle)

    if st.button("📄 Raporu oluştur", type="primary"):
        with st.spinner("Rapor hazırlanıyor..."):
            records = []
            # Betimsel
            if inc_desc and num_cols:
                desc = descriptive.describe(df, num_cols)
                figs = [report.fig_hist(pd.to_numeric(df[num_cols[0]], errors="coerce").dropna(),
                                        f"{num_cols[0]} dağılımı")]
                txt_lines = ["### Betimsel istatistik", I.bullet(f"{len(num_cols)} sayısal değişken, {df.shape[0]} gözlem.")]
                for c in num_cols[:6]:
                    nres = descriptive.normality(df[c], alpha)
                    if nres.get("normal") is not None:
                        txt_lines.append(I.bullet(f"{c}: {'normal dağılım' if nres['normal'] else 'normal dağılımdan sapıyor'} "
                                                  f"(p={nres.get('p', float('nan')):.3g})"))
                records.append({"title": "Betimsel istatistik", "text": I.joinlines(txt_lines),
                                "tables": [desc], "figures": figs})
            # Korelasyon
            if inc_corr and len(num_cols) >= 2:
                cols = num_cols[:8]
                corr, pmat = correlation.correlation_matrix(df, cols, "pearson")
                top = correlation.top_correlations(corr, pmat, alpha)
                txt = "### Korelasyon analizi\n" + I.joinlines(
                    [I.bullet(f"{r['Değişken 1']} ↔ {r['Değişken 2']}: r={r['r']:.3f} ({r['Güç']}) {r['Anlamlılık']}")
                     for _, r in top.head(6).iterrows()])
                records.append({"title": "Korelasyon", "text": txt, "tables": [top],
                                "figures": [report.fig_corr_heatmap(corr)]})
            # Regresyon
            if inc_reg and reg_res and "error" not in reg_res:
                actual = reg_res["fitted"] + reg_res["resid"]
                figs = [report.fig_scatter_fit(reg_res["fitted"], actual),
                        report.fig_residuals(reg_res["fitted"], reg_res["resid"])]
                records.append({"title": f"Regresyon: {reg_res['y']}", "text": regression.interpret_regression(reg_res),
                                "tables": [reg_res["coef_df"]], "figures": figs})
            # Lojistik
            if inc_logit and logit_res and "error" not in logit_res:
                roc = logit_res["roc"]
                figs = [report.fig_roc(roc["fpr"], roc["tpr"], roc["auc"]),
                        report.fig_confusion(logit_res["confusion"])]
                records.append({"title": f"Lojistik: {logit_res['y']}", "text": logistic.interpret_logit(logit_res),
                                "tables": [logit_res["coef_df"]], "figures": figs})
            # Zaman serisi
            if inc_ts and ts_bundle:
                s, ov, stat, dec, fc = (ts_bundle[k] for k in ("s", "ov", "stat", "dec", "fc"))
                figs = [report.fig_series_forecast(s.values, fc.get("fitted"), fc.get("forecast"))]
                if "error" not in dec:
                    figs.append(report.fig_decompose(dec["trend"], dec["seasonal"], dec["resid"]))
                txt = timeseries.interpret_overview(ov, stat, dec, alpha)
                if "error" not in fc:
                    txt += "\n\n" + timeseries.interpret_forecast(fc, float(s.iloc[-1]))
                records.append({"title": f"Zaman serisi: {ts_bundle['val']}", "text": txt, "figures": figs})

            if not records:
                st.warning("En az bir bölüm seçin (veya ilgili analizi önce çalıştırın).")
            else:
                meta = {"dataset": st.session_state.df_name, "rows": df.shape[0],
                        "cols": df.shape[1], "alpha": alpha}
                pdf_bytes = report.build_report(meta, records)
                st.session_state["pdf_bytes"] = pdf_bytes
                st.success(f"Rapor hazır — {len(records)} bölüm, {len(pdf_bytes)//1024} KB.")

    if st.session_state.get("pdf_bytes"):
        from datetime import datetime
        fname = f"StatLab_Rapor_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf"
        st.download_button("⬇️ PDF'i indir", st.session_state["pdf_bytes"], fname, "application/pdf",
                           type="primary")

    st.divider()
    st.markdown("**Excel / CSV dışa aktar**")
    if df.empty:
        st.caption("Dışa aktarma için önce veri yükleyin.")
    else:
        ec1, ec2 = st.columns(2)
        ec1.download_button("⬇️ Veriyi CSV indir", export.df_to_csv_bytes(df),
                            f"{(st.session_state.df_name or 'veri')}.csv", "text/csv")
        # Analiz sonuçlarını çok sayfalı Excel olarak
        sheets = {"Veri": df, "Betimsel": descriptive.describe(df, num_cols) if num_cols else pd.DataFrame(),
                  "Eksik_Rapor": cleaning.missing_report(df)}
        if len(num_cols) >= 2:
            cmat, _ = correlation.correlation_matrix(df, num_cols[:8], "pearson")
            sheets["Korelasyon"] = cmat.reset_index().rename(columns={"index": "Değişken"})
        reg_r = st.session_state.get("reg_res")
        if reg_r and "error" not in reg_r:
            sheets["Regresyon_Katsayi"] = reg_r["coef_df"]
        lg_r = st.session_state.get("logit_res")
        if lg_r and "error" not in lg_r:
            sheets["Lojistik_Katsayi"] = lg_r["coef_df"]
        ec2.download_button("⬇️ Sonuçları Excel indir", export.sheets_to_excel_bytes(sheets),
                            f"{(st.session_state.df_name or 'statlab')}_sonuclar.xlsx",
                            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

# ===== 11. TEMİZLE =====
with tabs[10]:
    st.subheader("Veri temizleme ve dönüşüm")
    orig = st.session_state.get("df_original")
    tc1, tc2 = st.columns([3, 1])
    tc1.caption("Yapılan değişiklikler tüm sekmelere yansır. İstediğin an orijinale dönebilirsin.")
    if orig is not None and tc2.button("↺ Orijinale dön"):
        st.session_state.df = orig.copy()
        st.rerun()

    st.markdown("**Eksik değer raporu**")
    st.dataframe(cleaning.missing_report(df), use_container_width=True)

    st.markdown("**1) Eksik değer doldurma**")
    mc1, mc2, mc3 = st.columns([2, 1.4, 1])
    fill_cols = mc1.multiselect("Sütun(lar)", list(df.columns),
                                default=[c for c in df.columns if df[c].isna().any()])
    fill_method = mc2.selectbox("Yöntem", ["mean", "median", "mode", "ffill", "bfill", "interpolate", "constant"],
                                format_func=lambda m: {"mean": "Ortalama", "median": "Medyan", "mode": "Mod",
                                                       "ffill": "İleri doldur", "bfill": "Geri doldur",
                                                       "interpolate": "İnterpolasyon", "constant": "Sabit değer"}[m])
    const_val = mc3.text_input("Sabit", value="0") if fill_method == "constant" else None
    b1, b2 = st.columns(2)
    if fill_cols and b1.button("Eksikleri doldur"):
        cval = float(const_val) if (const_val and const_val.replace(".", "").replace("-", "").isdigit()) else const_val
        st.session_state.df = cleaning.fill_missing(df, fill_cols, fill_method, cval)
        st.rerun()
    if b2.button("Eksik satırları sil"):
        st.session_state.df = cleaning.drop_missing(df, subset=fill_cols or None)
        st.rerun()

    st.divider()
    st.markdown("**2) Aykırı değer tespiti**")
    if num_cols:
        oc1, oc2, oc3 = st.columns([2, 1.4, 1.2])
        ocol = oc1.selectbox("Sayısal sütun", num_cols, key="out_col")
        omethod = oc2.selectbox("Yöntem", ["iqr", "zscore"],
                                format_func=lambda m: "IQR (çeyrekler)" if m == "iqr" else "Z-skoru")
        ofactor = oc3.number_input("Eşik/katsayı", 1.0, 5.0, 1.5 if omethod == "iqr" else 3.0, 0.5)
        od = cleaning.detect_outliers(df[ocol], omethod, ofactor)
        if "error" in od:
            st.info(od["error"])
        else:
            bf = px.box(df, y=ocol, points="outliers")
            bf.update_layout(height=280, margin=dict(t=20))
            st.plotly_chart(bf, use_container_width=True)
            st.caption(f"Tespit edilen aykırı değer: **{od['n_outliers']}** (%{od['pct']}) "
                       + (f"— sınırlar [{od['lower']:.3f}, {od['upper']:.3f}]" if od.get("lower") is not None else ""))
            act = st.radio("İşlem", ["cap", "remove", "nan"], horizontal=True,
                           format_func=lambda a: {"cap": "Sınırla (winsorize)", "remove": "Satırı sil",
                                                  "nan": "Eksik yap"}[a])
            if od["n_outliers"] > 0 and st.button("Aykırı değerleri işle"):
                st.session_state.df = cleaning.handle_outliers(df, ocol, od, act)
                st.rerun()

    st.divider()
    st.markdown("**3) Dönüşüm**")
    if num_cols:
        trc1, trc2 = st.columns([2, 2])
        tcol = trc1.selectbox("Sütun", num_cols, key="trans_col")
        tkind = trc2.selectbox("Dönüşüm", list(cleaning.TRANSFORMS),
                               format_func=lambda k: k, key="trans_kind")
        st.caption(cleaning.TRANSFORMS[tkind])
        if st.button("Dönüşümü uygula (yeni sütun)"):
            newdf, note = cleaning.transform(df, tcol, tkind)
            st.session_state.df = newdf
            if note:
                st.info(note)
            st.success(f"Yeni sütun eklendi: {tcol}_{tkind}")
            newname = f"{tcol}_{tkind}"
            if newname in newdf.columns:
                st.markdown(cleaning.transform_skew_note(df[tcol], newdf[newname]).replace("- ", ""))
            st.rerun()

# ===== 12. KARŞILAŞTIR =====
with tabs[11]:
    st.subheader("Regresyon modeli karşılaştırma")
    st.caption("Aynı bağımlı değişken için farklı öngörücü kümelerini AIC / BIC / düzeltilmiş R² ile kıyaslayın.")
    if not num_cols:
        st.warning("Karşılaştırma için sayısal sütun gerekir.")
    else:
        cmp_y = st.selectbox("Bağımlı değişken (Y)", num_cols, key="cmp_y")
        x_pool = [c for c in df.columns if c != cmp_y]
        n_models = st.slider("Model sayısı", 2, 4, 2)
        specs = []
        mcols = st.columns(n_models)
        defaults = [c for c in num_cols if c != cmp_y]
        for i in range(n_models):
            with mcols[i]:
                st.markdown(f"**Model {i+1}**")
                default_x = defaults[: i + 1] if i + 1 <= len(defaults) else defaults
                xv = st.multiselect(f"X (Model {i+1})", x_pool, default=default_x, key=f"cmp_x_{i}")
                if xv:
                    specs.append({"name": f"M{i+1}: " + "+".join(xv), "x_vars": xv,
                                  "categorical": [v for v in xv if v in cat_cols]})
        if len(specs) >= 2 and st.button("Modelleri karşılaştır", type="primary"):
            res = compare.compare_models(df, cmp_y, specs, alpha)
            st.session_state["cmp_res"] = res
        res = st.session_state.get("cmp_res")
        if res and res.get("y") == cmp_y:
            tbl = res["table"]
            st.dataframe(tbl.style.format({"R²": "{:.4f}", "Düz. R²": "{:.4f}", "AIC": "{:.1f}",
                                           "BIC": "{:.1f}", "RMSE": "{:.4f}", "F p-değeri": "{:.3g}"}),
                         use_container_width=True)
            if "AIC" in tbl.columns and tbl["AIC"].notna().any():
                gc1, gc2 = st.columns(2)
                with gc1:
                    fig = px.bar(tbl, x="Model", y="AIC", title="AIC (düşük = iyi)")
                    fig.update_layout(height=320, margin=dict(t=40))
                    st.plotly_chart(fig, use_container_width=True)
                with gc2:
                    fig2 = px.bar(tbl, x="Model", y="Düz. R²", title="Düzeltilmiş R² (yüksek = iyi)")
                    fig2.update_layout(height=320, margin=dict(t=40))
                    st.plotly_chart(fig2, use_container_width=True)
            interp_box(compare.interpret_comparison(res))
            claude_expander(f"Model karşılaştırma Y={cmp_y}:\n{tbl.to_string(index=False)}", key="cmp_claude")

# ===== 13. PANEL VERİ =====
with tabs[12]:
    st.subheader("Panel veri analizi (Pooled / Sabit Etkiler / Rassal Etkiler)")
    st.caption("Aynı birimlerin (kişi, firma, ülke…) zaman içinde tekrar gözlendiği veriler için. "
               "Birim ve zaman kimliği sütunları gerekir.")
    if len(num_cols) < 1 or df.shape[1] < 3:
        st.warning("Panel analizi için birim, zaman ve en az bir sayısal değişken gerekir.")
    else:
        pc1, pc2, pc3 = st.columns(3)
        p_entity = pc1.selectbox("Birim kimliği (entity)", list(df.columns), key="pnl_entity")
        p_time = pc2.selectbox("Zaman kimliği", [c for c in df.columns if c != p_entity], key="pnl_time")
        p_y = pc3.selectbox("Bağımlı değişken (Y)", [c for c in num_cols if c not in (p_entity, p_time)], key="pnl_y")
        p_x = st.multiselect("Bağımsız değişken(ler) (X)",
                             [c for c in num_cols if c not in (p_entity, p_time, p_y)], key="pnl_x")
        if p_x and st.button("Panel modellerini çalıştır", type="primary"):
            st.session_state["pnl_res"] = panel.fit_panel(df, p_y, p_x, p_entity, p_time, alpha)
        pres = st.session_state.get("pnl_res")
        if pres and pres.get("y") == p_y:
            if "error" in pres:
                st.error(pres["error"])
            else:
                e1, e2, e3 = st.columns(3)
                e1.metric("Birim sayısı", pres["n_entities"])
                e2.metric("Dönem sayısı", pres["n_periods"])
                e3.metric("Gözlem", pres["n"])
                st.markdown("**Model karşılaştırması**")
                st.dataframe(pres["summary_table"].style.format({"R² (overall)": "{:.4f}", "R² (within)": "{:.4f}"}),
                             use_container_width=True)
                pick = st.selectbox("Katsayı tablosu — model seç", list(pres["coef_tables"]))
                ct = pres["coef_tables"][pick]
                st.dataframe(ct.style.format({"Katsayı (β)": "{:.4f}", "Std. Hata": "{:.4f}",
                                              "t": "{:.3f}", "p": "{:.4g}"}), use_container_width=True)
                interp_box(panel.interpret_panel(pres))
                claude_expander(f"Panel {p_y}~{p_x}, {pres['n_entities']} birim × {pres['n_periods']} dönem. "
                                f"Hausman p={pres.get('hausman',{}).get('p')}.\n{pres['summary_table'].to_string(index=False)}",
                                key="pnl_claude")

# ===== 14. KÜMELEME / PCA =====
with tabs[13]:
    st.subheader("Kümeleme ve boyut indirgeme")
    if len(num_cols) < 2:
        st.warning("Kümeleme/PCA için en az 2 sayısal sütun gerekir.")
    else:
        kmode = st.radio("Analiz", ["K-Means", "Hiyerarşik", "PCA (boyut indirgeme)"], horizontal=True)
        kcols = st.multiselect("Değişkenler", num_cols, default=num_cols[:min(5, len(num_cols))], key="clu_cols")
        scale = st.checkbox("Standartlaştır (önerilir)", value=True, key="clu_scale")

        if kmode in ("K-Means", "Hiyerarşik") and len(kcols) >= 2:
            kk = st.slider("Küme sayısı (k)", 2, 8, 3, key="clu_k")
            if kmode == "Hiyerarşik":
                link = st.selectbox("Bağlantı", ["ward", "complete", "average", "single"], key="clu_link")
            if st.button("Kümele", type="primary"):
                if kmode == "K-Means":
                    res = clustering.kmeans(df, kcols, kk, scale)
                else:
                    res = clustering.hierarchical(df, kcols, kk, link, scale)
                st.session_state["clu_res"] = res
                if "error" not in res:
                    st.session_state["elbow_res"] = clustering.elbow(df, kcols, 8, scale)
            res = st.session_state.get("clu_res")
            if res and not res.get("error") and set(res.get("cols", [])) == set(kcols):
                sc1, sc2, sc3 = st.columns(3)
                sc1.metric("Küme sayısı", res["k"])
                sc2.metric("Silhouette", f"{res['silhouette']:.3f}")
                if "inertia" in res:
                    sc3.metric("Inertia", f"{res['inertia']:.1f}")
                pdf_plot = df.loc[res["index"], kcols].copy()
                pdf_plot["Küme"] = [f"Küme {l}" for l in res["labels"]]
                if len(kcols) >= 2:
                    sf = px.scatter(pdf_plot, x=kcols[0], y=kcols[1], color="Küme",
                                    title=f"{res['method']} — {kcols[0]} vs {kcols[1]}")
                    sf.update_layout(height=380, margin=dict(t=40))
                    st.plotly_chart(sf, use_container_width=True)
                st.markdown("**Küme merkezleri (değişken ortalamaları)**")
                st.dataframe(res["centers"].style.format(precision=3), use_container_width=True)
                interp_box(clustering.interpret_clustering(res))
                el = st.session_state.get("elbow_res")
                if el:
                    with st.expander("Optimal k — Elbow ve Silhouette"):
                        el_df = pd.DataFrame({"k": el["k"], "Inertia": el["inertia"], "Silhouette": el["silhouette"]})
                        ec1, ec2 = st.columns(2)
                        ec1.plotly_chart(px.line(el_df, x="k", y="Inertia", markers=True,
                                                 title="Elbow (dirsek)").update_layout(height=300), use_container_width=True)
                        ec2.plotly_chart(px.line(el_df, x="k", y="Silhouette", markers=True,
                                                 title=f"En iyi k = {el['best_k']}").update_layout(height=300), use_container_width=True)
                # kümeyi veriye ekle
                if st.button("Küme etiketini veriye ekle"):
                    newdf = df.copy()
                    newdf.loc[res["index"], "Kume"] = res["labels"]
                    st.session_state.df = newdf
                    st.rerun()
                claude_expander(f"{res['method']} {res['k']} küme, silhouette={res['silhouette']:.3f}. "
                                f"Merkezler:\n{res['centers'].to_string(index=False)}", key="clu_claude")

        elif kmode == "PCA (boyut indirgeme)" and len(kcols) >= 2:
            if st.button("PCA çalıştır", type="primary"):
                st.session_state["pca_res"] = clustering.pca(df, kcols, None, scale)
            res = st.session_state.get("pca_res")
            if res and set(res.get("cols", [])) == set(kcols):
                st.dataframe(res["var_df"].style.format({"Açıklanan Varyans %": "{:.1f}", "Kümülatif %": "{:.1f}"}),
                             use_container_width=True)
                vc1, vc2 = st.columns(2)
                vc1.plotly_chart(px.bar(res["var_df"], x="Bileşen", y="Açıklanan Varyans %",
                                        title="Açıklanan varyans").update_layout(height=320), use_container_width=True)
                if res["scores"].shape[1] >= 2:
                    sdf = pd.DataFrame(res["scores"][:, :2], columns=["PC1", "PC2"])
                    vc2.plotly_chart(px.scatter(sdf, x="PC1", y="PC2", title="Gözlemler (PC1–PC2)",
                                                opacity=0.6).update_layout(height=320), use_container_width=True)
                st.markdown("**Bileşen yükleri (loadings)**")
                st.dataframe(res["loadings"].style.format(precision=3).background_gradient(cmap="RdBu", axis=None),
                             use_container_width=True)
                interp_box(clustering.interpret_pca(res))
                claude_expander(f"PCA {kcols}: ilk PC={res['explained'][0]*100:.1f}% varyans.\n"
                                f"Yükler:\n{res['loadings'].to_string()}", key="pca_claude")

# ===== 15. SAYIM (GLM) =====
with tabs[14]:
    st.subheader("Sayım verisi regresyonu (Poisson / Negatif Binom)")
    st.caption("Bağımlı değişken sayım (0,1,2,… olay sayısı) olduğunda; OLS yerine log-bağlantılı GLM.")
    if not num_cols:
        st.warning("Sayım modeli için sayısal sütun gerekir.")
    else:
        gc1, gc2 = st.columns(2)
        gy = gc1.selectbox("Sayım değişkeni (Y ≥ 0)", num_cols, key="cnt_y")
        gfam = gc2.selectbox("Aile / model", ["poisson", "negbin", "zip", "zinb"],
                             format_func=lambda f: {"poisson": "Poisson", "negbin": "Negatif Binom",
                                                    "zip": "Sıfır-şişirilmiş Poisson (ZIP)",
                                                    "zinb": "Sıfır-şişirilmiş Neg. Binom (ZINB)"}[f])
        gx = st.multiselect("Bağımsız değişken(ler)", [c for c in df.columns if c != gy],
                            default=[c for c in num_cols if c != gy][:3], key="cnt_x")
        gcat = [v for v in gx if v in cat_cols]
        if gx and st.button("Model çalıştır", type="primary"):
            if gfam in ("zip", "zinb"):
                znum = [v for v in gx if v in num_cols]
                st.session_state["cnt_res"] = advanced.fit_zero_inflated(
                    df, gy, znum, "negbin" if gfam == "zinb" else "poisson", alpha)
            else:
                st.session_state["cnt_res"] = count_models.fit_count(df, gy, gx, gfam, gcat, alpha)
        res = st.session_state.get("cnt_res")
        if res and res.get("y") == gy:
            if "error" in res:
                st.error(res["error"])
            elif "zero_obs" in res:  # sıfır-şişirilmiş
                z1, z2 = st.columns(2)
                z1.metric("Sıfır oranı", f"{res['zero_obs']*100:.1f}%")
                z2.metric("AIC", f"{res['aic']:.1f}")
                st.dataframe(res["coef_df"].style.format({"Katsayı (β)": "{:.4f}", "IRR": "{:.3f}", "p": "{:.4g}"}),
                             use_container_width=True)
                interp_box(advanced.interpret_zero_inflated(res))
                claude_expander(f"Sıfır-şişirilmiş {res['family']} {gy}, sıfır={res['zero_obs']:.2f}.", key="zi_claude")
            else:
                c1, c2, c3 = st.columns(3)
                c1.metric("Sözde-R²", f"{res['pseudo_r2']:.3f}")
                c2.metric("Yayılım oranı", f"{res['dispersion']:.2f}")
                c3.metric("AIC", f"{res['aic']:.1f}")
                st.dataframe(res["coef_df"].style.format({"Katsayı (β)": "{:.4f}", "IRR (oran oranı)": "{:.3f}",
                             "IRR CI alt": "{:.3f}", "IRR CI üst": "{:.3f}", "z": "{:.3f}", "p": "{:.4g}"}),
                             use_container_width=True)
                interp_box(count_models.interpret_count(res))
                claude_expander(f"{res['family']} GLM {gy}~{gx}, yayılım={res['dispersion']:.2f}.\n"
                                f"{res['coef_df'].to_string(index=False)}", key="cnt_claude")

# ===== 16. GÜÇ & GÜVENİLİRLİK =====
with tabs[15]:
    st.subheader("Güç analizi, güvenilirlik ve çoklu karşılaştırma")
    sub = st.radio("Araç", ["Güç / örneklem", "Cronbach α", "Çoklu karşılaştırma düzeltmesi"], horizontal=True)

    if sub == "Güç / örneklem":
        pt = st.selectbox("Test türü", ["İki örneklem t-testi", "Tek yönlü ANOVA", "İki oran"])
        solve = st.radio("Ne hesaplansın?", ["Örneklem büyüklüğü (n)", "Güç"], horizontal=True)
        pcol = st.columns(3)
        if pt == "İki örneklem t-testi":
            eff = pcol[0].number_input("Etki (Cohen's d)", 0.1, 3.0, 0.5, 0.1)
            if solve == "Örneklem büyüklüğü (n)":
                pw = pcol[1].slider("Hedef güç", 0.5, 0.99, 0.80)
                res = power.solve_ttest(eff, None, pw, alpha)
            else:
                nn = pcol[1].number_input("Grup başına n", 5, 100000, 30)
                res = power.solve_ttest(eff, int(nn), None, alpha)
            curve = power.power_curve("ttest", eff, alpha, range(5, 200, 5))
        elif pt == "Tek yönlü ANOVA":
            eff = pcol[0].number_input("Etki (Cohen's f)", 0.05, 2.0, 0.25, 0.05)
            kg = pcol[2].number_input("Grup sayısı", 2, 20, 3)
            if solve == "Örneklem büyüklüğü (n)":
                pw = pcol[1].slider("Hedef güç", 0.5, 0.99, 0.80)
                res = power.solve_anova(eff, int(kg), None, pw, alpha)
            else:
                nn = pcol[1].number_input("Grup başına n", 5, 100000, 30)
                res = power.solve_anova(eff, int(kg), int(nn), None, alpha)
            curve = power.power_curve("anova", eff, alpha, range(5, 200, 5), int(kg))
        else:
            p1 = pcol[0].number_input("Oran 1", 0.01, 0.99, 0.30, 0.05)
            p2 = pcol[2].number_input("Oran 2", 0.01, 0.99, 0.50, 0.05)
            if solve == "Örneklem büyüklüğü (n)":
                pw = pcol[1].slider("Hedef güç", 0.5, 0.99, 0.80)
                res = power.solve_proportions(p1, p2, None, pw, alpha)
            else:
                nn = pcol[1].number_input("Grup başına n", 5, 100000, 30)
                res = power.solve_proportions(p1, p2, int(nn), None, alpha)
            curve = power.power_curve("prop", power.proportion_effectsize(p1, p2), alpha, range(5, 300, 5)) \
                if hasattr(power, "proportion_effectsize") else None
        interp_box(power.interpret_power(res))
        if curve:
            cf = px.line(pd.DataFrame(curve), x="n", y="power", title="Güç eğrisi (n'e göre)")
            cf.add_hline(y=0.80, line_dash="dash", annotation_text="0.80")
            cf.update_layout(height=340, margin=dict(t=40))
            st.plotly_chart(cf, use_container_width=True)

    elif sub == "Cronbach α":
        st.caption("Bir ölçeğin (ör. anket maddelerinin) iç tutarlılık güvenilirliği.")
        items = st.multiselect("Ölçek maddeleri (sayısal, ≥2)", num_cols,
                               default=num_cols[:min(5, len(num_cols))], key="cron_items")
        if len(items) >= 2 and st.button("Hesapla", type="primary"):
            res = reliability.cronbach_alpha(df, items)
            if "error" in res:
                st.error(res["error"])
            else:
                st.metric("Cronbach α", f"{res['alpha']:.3f}")
                st.dataframe(res["item_stats"].style.format({"Madde-Toplam r": "{:.3f}",
                             "α (madde silinirse)": "{:.3f}"}), use_container_width=True)
                interp_box(reliability.interpret_cronbach(res))
                claude_expander(f"Cronbach α={res['alpha']:.3f}, {res['k']} madde.", key="cron_claude")

    else:
        st.caption("Birden çok test yaptıysanız p-değerlerini yapıştırın; I. tip hata şişmesini düzeltir.")
        method = st.selectbox("Yöntem", ["fdr_bh", "bonferroni", "holm"],
                              format_func=lambda m: {"fdr_bh": "Benjamini-Hochberg (FDR)",
                                                     "bonferroni": "Bonferroni", "holm": "Holm"}[m])
        ptext = st.text_area("p-değerleri (virgül veya satırla ayrılmış)", value="0.001, 0.02, 0.04, 0.30, 0.60")
        if st.button("Düzelt", type="primary"):
            import re as _re
            pvals = [float(x) for x in _re.split(r"[,\s]+", ptext.strip()) if x]
            res = reliability.correct_pvalues(pvals, None, method, alpha)
            if "error" in res:
                st.error(res["error"])
            else:
                st.dataframe(res["table"].style.format({"p (ham)": "{:.4g}", "p (düzeltilmiş)": "{:.4g}"}),
                             use_container_width=True)
                interp_box(reliability.interpret_correction(res))

# ===== 17. SAĞKALIM =====
with tabs[16]:
    st.subheader("Sağkalım analizi (Kaplan-Meier / Cox)")
    st.caption("Bir olaya kadar geçen süre verileri için. Süre (numerik) ve olay (1=oldu, 0=sansür) sütunları gerekir.")
    if len(num_cols) < 2:
        st.warning("Sağkalım analizi için süre ve olay sütunları gerekir.")
    else:
        smode = st.radio("Analiz", ["Kaplan-Meier", "Cox regresyon"], horizontal=True)
        sc1, sc2 = st.columns(2)
        dur = sc1.selectbox("Süre değişkeni", num_cols, key="surv_dur")
        evt = sc2.selectbox("Olay değişkeni (1/0)", [c for c in num_cols if c != dur], key="surv_evt")
        if smode == "Kaplan-Meier":
            grp = st.selectbox("Grup (opsiyonel)", ["(yok)"] + cat_cols, key="surv_grp")
            gcol = None if grp == "(yok)" else grp
            if st.button("KM eğrisini çiz", type="primary"):
                st.session_state["km_res"] = survival.kaplan_meier(df, dur, evt, gcol)
            res = st.session_state.get("km_res")
            if res:
                if "error" in res:
                    st.error(res["error"])
                else:
                    kf = go.Figure()
                    for label, cur in res["curves"].items():
                        kf.add_scatter(x=cur.iloc[:, 0], y=cur.iloc[:, 1], mode="lines",
                                       name=label, line_shape="hv")
                    kf.update_layout(title="Kaplan-Meier sağkalım eğrisi", height=380,
                                     xaxis_title="Süre", yaxis_title="Sağkalım olasılığı", margin=dict(t=40))
                    st.plotly_chart(kf, use_container_width=True)
                    interp_box(survival.interpret_km(res, alpha))
                    claude_expander(f"KM medyanlar={res['medians']}, logrank={res.get('logrank')}.", key="km_claude")
        else:
            covs = st.multiselect("Ortak değişkenler (sayısal)", [c for c in num_cols if c not in (dur, evt)],
                                  default=[c for c in num_cols if c not in (dur, evt)][:3], key="cox_covs")
            if covs and st.button("Cox modeli çalıştır", type="primary"):
                st.session_state["cox_res"] = survival.cox_ph(df, dur, evt, covs, alpha)
            res = st.session_state.get("cox_res")
            if res:
                if "error" in res:
                    st.error(res["error"])
                else:
                    st.metric("C-indeksi (concordance)", f"{res['concordance']:.3f}")
                    st.dataframe(res["table"].style.format({"Katsayı (β)": "{:.4f}", "Tehlike Oranı (HR)": "{:.3f}",
                                 "HR CI alt": "{:.3f}", "HR CI üst": "{:.3f}", "p": "{:.4g}"}),
                                 use_container_width=True)
                    interp_box(survival.interpret_cox(res))
                    claude_expander(f"Cox C={res['concordance']:.3f}.\n{res['table'].to_string(index=False)}", key="cox_claude")

# ===== 18. FAKTÖR ANALİZİ =====
with tabs[17]:
    st.subheader("Açımlayıcı Faktör Analizi (EFA)")
    st.caption("Çok sayıda maddenin (ör. anket soruları) altında yatan gizli boyutları (faktörleri) ortaya çıkarır.")
    if len(num_cols) < 3:
        st.warning("Faktör analizi için en az 3 sayısal madde gerekir.")
    else:
        fitems = st.multiselect("Maddeler (sayısal)", num_cols,
                                default=num_cols[:min(6, len(num_cols))], key="fa_items")
        fc1, fc2 = st.columns(2)
        nfac = fc1.slider("Faktör sayısı", 1, min(6, max(2, len(fitems) - 1)), 2, key="fa_n")
        frot = fc2.selectbox("Döndürme", ["varimax", "quartimax", "yok"], key="fa_rot")
        if len(fitems) >= 3 and st.button("Faktör analizi çalıştır", type="primary"):
            st.session_state["fa_res"] = advanced.factor_analysis(df, fitems, nfac,
                                                                 None if frot == "yok" else frot)
        res = st.session_state.get("fa_res")
        if res:
            if "error" in res:
                st.error(res["error"])
            else:
                k1, k2 = st.columns(2)
                k1.metric("KMO örnekleme yeterliliği", f"{res['kmo']:.3f}")
                k2.metric("Bartlett küresellik p", f"{res['bartlett_p']:.3g}")
                st.dataframe(res["var_df"].style.format({"Açıklanan Varyans %": "{:.1f}", "Kümülatif %": "{:.1f}"}),
                             use_container_width=True)
                st.markdown("**Faktör yükleri (loadings)**")
                st.dataframe(res["loadings"].style.format(precision=3).background_gradient(cmap="RdBu", axis=None),
                             use_container_width=True)
                ev_df = pd.DataFrame({"Bileşen": range(1, len(res["eigenvalues"]) + 1), "Özdeğer": res["eigenvalues"]})
                sf = px.line(ev_df, x="Bileşen", y="Özdeğer", markers=True, title="Yamaç (scree) grafiği")
                sf.add_hline(y=1, line_dash="dash", annotation_text="Kaiser (=1)")
                sf.update_layout(height=320, margin=dict(t=40))
                st.plotly_chart(sf, use_container_width=True)
                interp_box(advanced.interpret_factor(res))
                claude_expander(f"EFA KMO={res['kmo']:.3f}, {res['n_factors']} faktör.\n"
                                f"Yükler:\n{res['loadings'].to_string()}", key="fa_claude")

# ===== 19. KARIŞIK MODEL =====
with tabs[18]:
    st.subheader("Karışık etki (çok düzeyli) modeli")
    st.caption("Verinin gruplar/kümeler içinde yuvalandığı durumlar (ör. öğrenciler-okullar, ölçümler-hastalar). "
               "Panel FE/RE'nin genellemesi; rassal kesişim/eğim.")
    if len(num_cols) < 1 or not (cat_cols or len(num_cols) >= 2):
        st.warning("Karışık model için bir gruplama sütunu ve sayısal değişkenler gerekir.")
    else:
        group_opts = cat_cols + [c for c in num_cols if df[c].nunique() <= 50]
        mc1, mc2 = st.columns(2)
        mgroup = mc1.selectbox("Gruplama (rassal etki) değişkeni", group_opts, key="mix_g")
        my = mc2.selectbox("Bağımlı değişken (Y)", [c for c in num_cols if c != mgroup], key="mix_y")
        mx = st.multiselect("Sabit etki öngörücüleri (X)", [c for c in num_cols if c not in (my, mgroup)],
                            default=[c for c in num_cols if c not in (my, mgroup)][:2], key="mix_x")
        rslope = st.selectbox("Rassal eğim (opsiyonel)", ["(yok)"] + mx, key="mix_rs")
        if mx and st.button("Karışık modeli çalıştır", type="primary"):
            st.session_state["mix_res"] = advanced.fit_mixedlm(df, my, mx, mgroup,
                                                              None if rslope == "(yok)" else rslope, alpha)
        res = st.session_state.get("mix_res")
        if res and res.get("y") == my:
            if "error" in res:
                st.error(res["error"])
            else:
                e1, e2, e3 = st.columns(3)
                e1.metric("ICC", f"{res['icc']:.3f}")
                e2.metric("Grup sayısı", res["n_groups"])
                e3.metric("Gözlem", res["n"])
                st.dataframe(res["fe"].style.format({"Katsayı (β)": "{:.4f}", "Std. Hata": "{:.4f}",
                             "z": "{:.3f}", "p": "{:.4g}"}), use_container_width=True)
                interp_box(advanced.interpret_mixedlm(res))
                claude_expander(f"MixedLM {my}~{mx}+(1|{mgroup}), ICC={res['icc']:.3f}.", key="mix_claude")

# ===== 20. TEST SEÇİCİ (SİHİRBAZ) =====
with tabs[19]:
    st.subheader("🧙 Hangi testi kullanmalıyım?")
    st.caption("Birkaç soruyu yanıtla; verine ve sorununa uygun yöntemi önereyim ve hangi sekmede olduğunu söyleyeyim.")
    goal = st.selectbox("Ne yapmak istiyorsun?", [
        "Grupları/ortalamaları karşılaştır", "İki değişken arasındaki ilişki",
        "Bir sonucu tahmin et / modelle", "İki kategorik değişken ilişkisi",
        "Gözlemleri grupla / boyut indir", "Zaman içinde değişim / gelecek tahmini",
        "Ölçek/anket güvenilirliği"])
    kw = {}
    if goal == "Grupları/ortalamaları karşılaştır":
        c1, c2, c3 = st.columns(3)
        kw["n_groups"] = c1.selectbox("Kaç grup / örneklem?", ["1 (tek örneklem)", "2 grup", "3+ grup"])
        kw["paired"] = c2.selectbox("Ölçümler eşleşmiş mi?", ["Hayır (bağımsız)", "Evet (aynı denek)"]) == "Evet (aynı denek)"
        kw["normal"] = c3.selectbox("Veri normal dağılıyor mu?", ["Evet / bilmiyorum", "Hayır"]) == "Evet / bilmiyorum"
        kw["equal_var"] = st.selectbox("Grupların varyansları benzer mi?", ["Evet / bilmiyorum", "Hayır"]) == "Evet / bilmiyorum"
    elif goal == "İki değişken arasındaki ilişki":
        c1, c2 = st.columns(2)
        kw["dv"] = c1.selectbox("Değişken tipleri", ["İkisi de sürekli", "Biri ikili (0/1)"])
        kw["dv"] = "İkili (0/1)" if kw["dv"] == "Biri ikili (0/1)" else "Sürekli"
        kw["normal"] = c2.selectbox("Sürekli değişken normal mi?", ["Evet", "Hayır"]) == "Evet"
    elif goal == "Bir sonucu tahmin et / modelle":
        c1, c2 = st.columns(2)
        kw["dv"] = c1.selectbox("Tahmin edilen (Y) tipi", ["Sürekli", "İkili (0/1)", "Sıralı", "Kategorik (3+)", "Sayım", "Süre / olay"])
        kw["n_pred"] = c2.selectbox("Kaç öngörücü?", ["Tek", "Çok (çoklu)"])
    elif goal == "İki kategorik değişken ilişkisi":
        kw["paired"] = st.selectbox("Ölçümler eşleşmiş mi (öncesi/sonrası)?", ["Hayır", "Evet"]) == "Evet"

    if st.button("Öneri getir", type="primary"):
        rec = wizard.recommend(goal, **kw)
        interp_box(wizard.format_recommendation(rec))

# ===== 21. FİNANS: GETİRİ & RİSK =====
with tabs[20]:
    st.subheader("📈 Getiri & risk analizi")
    st.caption("Sembolleri gir (ör. AAPL, MSFT, THYAO.IS), fiyatları çekelim; getiri ve risk metriklerini üretelim.")
    fc1, fc2, fc3 = st.columns([2.5, 1, 1])
    syms = fc1.text_input("Semboller (virgülle)", value="AAPL, MSFT, GOOGL", key="fin_syms")
    period = fc2.selectbox("Dönem", ["1y", "2y", "3y", "5y", "10y", "max"], index=2, key="fin_period")
    rf = fc3.number_input("Risksiz oran %", 0.0, 25.0, 2.0, 0.5, key="fin_rf") / 100
    rmethod = st.radio("Getiri türü", ["log", "simple"], horizontal=True,
                       format_func=lambda m: "Logaritmik" if m == "log" else "Basit", key="fin_rmethod")
    if st.button("Verileri çek ve analiz et", type="primary"):
        symbols = [s.strip().upper() for s in syms.split(",") if s.strip()]
        with st.spinner("Fiyatlar çekiliyor..."):
            try:
                prices = finance.fetch_prices(symbols, period)
            except Exception as e:
                prices = None
                st.error(f"Veri çekilemedi: {e}")
        if prices is not None and not prices.empty:
            rets = finance.to_returns(prices, rmethod)
            st.session_state["fin_prices"] = prices
            st.session_state["fin_returns"] = rets
            st.session_state["fin_rf_val"] = rf
            st.success(f"{prices.shape[1]} varlık, {prices.shape[0]} gün çekildi.")

    prices = st.session_state.get("fin_prices")
    rets = st.session_state.get("fin_returns")
    if prices is not None and rets is not None:
        norm = prices / prices.iloc[0] * 100
        pf = px.line(norm, title="Normalize fiyat (başlangıç = 100)")
        pf.update_layout(height=340, margin=dict(t=40), legend_title="")
        st.plotly_chart(pf, use_container_width=True)

        stats_df = finance.performance_stats(rets, st.session_state.get("fin_rf_val", rf))
        st.markdown("**Performans metrikleri (yıllıklandırılmış)**")
        st.dataframe(stats_df.style.format({
            "Yıllık Getiri": "{:.1%}", "Yıllık Volatilite": "{:.1%}", "Sharpe": "{:.2f}",
            "Sortino": "{:.2f}", "Calmar": "{:.2f}", "Maks. Düşüş": "{:.1%}",
            "VaR %95 (günlük)": "{:.2%}", "CVaR %95": "{:.2%}", "Toplam Getiri": "{:.1%}"}),
            use_container_width=True)

        gg1, gg2 = st.columns(2)
        with gg1:
            dd = pd.DataFrame({c: finance.drawdown_series(rets[c]) for c in rets.columns})
            ddf = px.area(dd, title="Düşüş (drawdown) grafiği")
            ddf.update_layout(height=320, margin=dict(t=40), legend_title="")
            st.plotly_chart(ddf, use_container_width=True)
        with gg2:
            cmat = rets.corr()
            hf = px.imshow(cmat, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                           title="Getiri korelasyonu")
            hf.update_layout(height=320, margin=dict(t=40))
            st.plotly_chart(hf, use_container_width=True)

        interp_box(finance.interpret_performance(stats_df, st.session_state.get("fin_rf_val", rf)))
        claude_expander(f"Portföy performansı:\n{stats_df.to_string(index=False)}", key="fin_perf_claude")
    else:
        st.info("Başlamak için sembolleri girip 'Verileri çek ve analiz et'e bas.")

# ===== 22. FİNANS: OPTİMİZASYON =====
with tabs[21]:
    st.subheader("⚖️ Portföy optimizasyonu (Markowitz)")
    rets = st.session_state.get("fin_returns")
    if rets is None:
        st.info("Önce '📈 Getiri & Risk' sekmesinden sembolleri çek.")
    elif rets.shape[1] < 2:
        st.warning("Optimizasyon için en az 2 varlık gerekir.")
    else:
        oc1, oc2 = st.columns(2)
        allow_short = oc1.checkbox("Kısa satışa izin ver", value=False)
        rf = st.session_state.get("fin_rf_val", 0.02)
        if st.button("Optimize et", type="primary"):
            with st.spinner("Etkin sınır hesaplanıyor..."):
                st.session_state["fin_opt"] = finance.optimize_portfolio(rets, rf, allow_short)
                st.session_state["fin_rp"] = finance.risk_parity(rets)
        opt = st.session_state.get("fin_opt")
        if opt:
            fr = opt["frontier"]
            ff = go.Figure()
            if not fr.empty:
                ff.add_scatter(x=fr["vol"], y=fr["ret"], mode="lines", name="Etkin sınır", line=dict(width=3))
            ap = pd.DataFrame(opt["asset_points"])
            ff.add_scatter(x=ap["vol"], y=ap["ret"], mode="markers+text", text=ap["Varlık"],
                           textposition="top center", name="Varlıklar", marker=dict(size=9))
            ff.add_scatter(x=[opt["max_sharpe"]["vol"]], y=[opt["max_sharpe"]["ret"]], mode="markers",
                           name="Maks. Sharpe", marker=dict(size=16, symbol="star", color="gold"))
            ff.add_scatter(x=[opt["min_var"]["vol"]], y=[opt["min_var"]["ret"]], mode="markers",
                           name="Min. Varyans", marker=dict(size=14, symbol="diamond", color="cyan"))
            ff.update_layout(title="Etkin sınır", xaxis_title="Yıllık volatilite", yaxis_title="Yıllık getiri",
                             height=420, margin=dict(t=40))
            st.plotly_chart(ff, use_container_width=True)

            wc1, wc2, wc3 = st.columns(3)
            def _wtab(d):
                return pd.DataFrame({"Varlık": list(d.keys()), "Ağırlık": list(d.values())})
            wc1.markdown("**Maks. Sharpe ağırlıkları**")
            wc1.dataframe(_wtab(opt["max_sharpe"]["weights"]).style.format({"Ağırlık": "{:.1%}"}), use_container_width=True)
            wc2.markdown("**Min. Varyans ağırlıkları**")
            wc2.dataframe(_wtab(opt["min_var"]["weights"]).style.format({"Ağırlık": "{:.1%}"}), use_container_width=True)
            rp = st.session_state.get("fin_rp")
            if rp:
                wc3.markdown("**Risk paritesi ağırlıkları**")
                wc3.dataframe(_wtab(rp["weights"]).style.format({"Ağırlık": "{:.1%}"}), use_container_width=True)
            # pasta
            pie = px.pie(_wtab({k: v for k, v in opt["max_sharpe"]["weights"].items() if v > 0.005}),
                         names="Varlık", values="Ağırlık", title="Maks. Sharpe portföy dağılımı")
            pie.update_layout(height=340, margin=dict(t=40))
            st.plotly_chart(pie, use_container_width=True)
            interp_box(finance.interpret_optimization(opt))

            st.divider()
            st.markdown("#### 👤 Yatırımcı risk profiline göre portföy")
            prof = st.radio("Risk profili", list(finance.RISK_PROFILES.keys()), horizontal=True,
                            format_func=lambda p: f"{p} (%{int(finance.RISK_PROFILES[p]['target_vol']*100)} vol)")
            pres = finance.optimize_for_profile(rets, prof, rf, allow_short)
            pc1, pc2, pc3 = st.columns(3)
            pc1.metric("Riskli varlık", f"%{pres['w_risky']*100:.0f}")
            pc2.metric("Beklenen getiri", f"%{pres['ret']*100:.1f}")
            pc3.metric("Beklenen oynaklık", f"%{pres['vol']*100:.1f}")
            wtab = pd.DataFrame({"Varlık": list(pres["weights"].keys()),
                                 "Ağırlık": list(pres["weights"].values())})
            wtab = wtab[wtab["Ağırlık"].abs() > 0.005]
            ppie = px.pie(wtab, names="Varlık", values="Ağırlık", title=f"{prof} portföy dağılımı")
            ppie.update_layout(height=320, margin=dict(t=40))
            st.plotly_chart(ppie, use_container_width=True)
            interp_box(finance.interpret_profile(pres))
            claude_expander(f"{prof} portföy: {pres['weights']}", key="fin_prof_claude")

# ===== 23. FİNANS: CAPM / FAKTÖR =====
with tabs[22]:
    st.subheader("📊 CAPM / faktör regresyonu")
    rets = st.session_state.get("fin_returns")
    if rets is None or rets.shape[1] < 2:
        st.info("Önce '📈 Getiri & Risk' sekmesinden en az 2 sembol çek (biri piyasa/endeks olsun, ör. SPY).")
    else:
        cc1, cc2 = st.columns(2)
        asset = cc1.selectbox("Varlık", list(rets.columns), key="capm_asset")
        market = cc2.selectbox("Piyasa/endeks (benchmark)", [c for c in rets.columns if c != asset], key="capm_mkt")
        rf = st.session_state.get("fin_rf_val", 0.02)
        if st.button("CAPM çalıştır", type="primary"):
            res = finance.capm(rets[asset], rets[market], rf)
            st.session_state["capm_res"] = (res, asset, market)
        cr = st.session_state.get("capm_res")
        if cr and cr[1] == asset and cr[2] == market:
            res = cr[0]
            k1, k2, k3 = st.columns(3)
            k1.metric("Beta", f"{res['beta']:.3f}")
            k2.metric("Alpha (yıllık)", f"{res['alpha_annual']:.2%}")
            k3.metric("R²", f"{res['r2']:.3f}")
            sf = px.scatter(x=rets[market], y=rets[asset], trendline="ols", opacity=0.5,
                            labels={"x": f"{market} getiri", "y": f"{asset} getiri"}, title="Karakteristik doğru")
            sf.update_layout(height=340, margin=dict(t=40))
            st.plotly_chart(sf, use_container_width=True)
            rb = finance.rolling_beta(rets[asset], rets[market], 60)
            if len(rb):
                rbf = px.line(rb, title="60 günlük yuvarlanan beta")
                rbf.add_hline(y=res["beta"], line_dash="dash", annotation_text="Ortalama beta")
                rbf.update_layout(height=300, margin=dict(t=40), showlegend=False)
                st.plotly_chart(rbf, use_container_width=True)
            interp_box(finance.interpret_capm(res))
            claude_expander(f"CAPM {asset}~{market}: {res}", key="capm_claude")

# ===== 24. FİNANS: MONTE CARLO / GARCH =====
with tabs[23]:
    st.subheader("🎲 Monte Carlo simülasyonu & GARCH oynaklık")
    rets = st.session_state.get("fin_returns")
    if rets is None:
        st.info("Önce '📈 Getiri & Risk' sekmesinden veri çek.")
    else:
        fmode = st.radio("Analiz", ["Monte Carlo", "GARCH oynaklık"], horizontal=True)
        asset = st.selectbox("Varlık", list(rets.columns), key="mc_asset")
        if fmode == "Monte Carlo":
            mc1, mc2, mc3, mc4 = st.columns(4)
            init = mc1.number_input("Başlangıç değeri", 100, 100_000_000, 10000, step=1000)
            horizon = mc2.number_input("Ufuk (gün)", 20, 2520, 252, step=20)
            nsims = mc3.number_input("Senaryo sayısı", 100, 20000, 1000, step=100)
            goal = mc4.number_input("Hedef değer (ops.)", 0, 100_000_000, 0, step=1000)
            if st.button("Simüle et", type="primary"):
                res = finance.monte_carlo(rets[asset], init, int(horizon), int(nsims))
                paths = res["paths"]
                idx = np.random.default_rng(0).choice(len(paths), min(100, len(paths)), replace=False)
                sample = pd.DataFrame(paths[idx].T)
                lf = go.Figure()
                for c in sample.columns[:100]:
                    lf.add_scatter(y=sample[c], mode="lines", line=dict(width=0.5, color="rgba(99,102,241,.25)"),
                                   showlegend=False, hoverinfo="skip")
                lf.update_layout(title=f"{asset} — {int(nsims)} senaryo (100 örnek yol)", height=360,
                                 xaxis_title="Gün", yaxis_title="Portföy değeri", margin=dict(t=40))
                st.plotly_chart(lf, use_container_width=True)
                hf = px.histogram(res["final"], nbins=50, title="Son değer dağılımı")
                hf.update_layout(height=300, margin=dict(t=40), showlegend=False)
                st.plotly_chart(hf, use_container_width=True)
                interp_box(finance.interpret_montecarlo(res, goal if goal > 0 else None))
                claude_expander(f"Monte Carlo {asset}: p50={res['p50']:.0f}, zarar olas.={res['prob_loss']:.2f}", key="mc_claude")
        else:
            gc1, gc2 = st.columns(2)
            hor = gc1.number_input("Oynaklık öngörü ufku (gün)", 1, 60, 10)
            if st.button("GARCH çalıştır", type="primary"):
                try:
                    res = finance.garch_volatility(rets[asset], 1, 1, int(hor))
                    cv = pd.Series(res["cond_vol"]) * np.sqrt(252)
                    vf = px.line(cv, title=f"{asset} — koşullu yıllık oynaklık (GARCH)")
                    vf.update_layout(height=340, margin=dict(t=40), showlegend=False)
                    st.plotly_chart(vf, use_container_width=True)
                    interp_box(finance.interpret_garch(res))
                    claude_expander(f"GARCH {asset}: {res['params']}", key="garch_claude")
                except Exception as e:
                    st.error(f"GARCH hatası: {e}")

# ===== 25. A/B TEST =====
with tabs[24]:
    st.subheader("🅰️ A/B test & deney tasarımı")
    ab_mode = st.radio("Mod", ["Deney tasarımı (örneklem)", "Dönüşüm analizi (oran)", "Metrik analizi (sürekli)"],
                       horizontal=True)

    if ab_mode == "Deney tasarımı (örneklem)":
        st.caption("Testi başlatmadan önce: kaç gözlem gerekir?")
        d1, d2, d3 = st.columns(3)
        base = d1.number_input("Taban dönüşüm oranı %", 0.1, 99.0, 10.0, 0.5, key="ab_base") / 100
        rel = d2.selectbox("Etki türü", ["Bağıl %", "Mutlak puan"], key="ab_rel") == "Bağıl %"
        mde = d3.number_input("Saptanacak min. etki (MDE) %", 0.1, 100.0, 20.0, 1.0, key="ab_mde") / 100
        d4, d5 = st.columns(2)
        pw = d4.slider("Hedef güç", 0.5, 0.99, 0.80, key="ab_pw")
        traffic = d5.number_input("Günlük trafik (ops.)", 0, 10_000_000, 2000, step=100, key="ab_traffic")
        if st.button("Örneklem hesapla", type="primary"):
            res = ab_test.design_conversion(base, mde, pw, alpha, rel, traffic if traffic > 0 else None)
            c1, c2, c3 = st.columns(3)
            c1.metric("Kol başına n", f"{res['n_per_arm']:,}")
            c2.metric("Toplam n", f"{res['n_total']:,}")
            if "days" in res:
                c3.metric("Tahmini süre", f"{res['days']} gün")
            interp_box(ab_test.interpret_design(res))
            claude_expander(str(res), key="ab_design_claude")

    elif ab_mode == "Dönüşüm analizi (oran)":
        st.caption("Test bittikten sonra: kontrol (A) ve varyant (B) dönüşümlerini karşılaştır.")
        c1, c2, c3, c4 = st.columns(4)
        sa = c1.number_input("A dönüşüm", 0, 10_000_000, 120, key="ab_sa")
        na = c2.number_input("A toplam", 1, 10_000_000, 2000, key="ab_na")
        sb = c3.number_input("B dönüşüm", 0, 10_000_000, 160, key="ab_sb")
        nb = c4.number_input("B toplam", 1, 10_000_000, 2000, key="ab_nb")
        if st.button("Analiz et", type="primary"):
            res = ab_test.analyze_conversion(int(sa), int(na), int(sb), int(nb), alpha)
            bar = px.bar(x=["A (kontrol)", "B (varyant)"], y=[res["p_a"], res["p_b"]],
                         labels={"x": "", "y": "Dönüşüm oranı"}, title="Dönüşüm oranları")
            bar.update_layout(height=300, margin=dict(t=40), yaxis_tickformat=".1%")
            st.plotly_chart(bar, use_container_width=True)
            interp_box(ab_test.interpret_analysis(res))
            with st.expander("📿 Bayesçi bakış (P(B>A))"):
                bres = bayesian.bayesian_ab(int(sa), int(na), int(sb), int(nb), alpha=alpha)
                interp_box(bayesian.interpret_bayes_ab(bres))
            claude_expander(str(res), key="ab_conv_claude")

    else:
        st.caption("Sürekli metrik (gelir, süre…) için gruplu karşılaştırma.")
        if not num_cols or not cat_cols:
            st.warning("Bir sayısal metrik ve bir grup (varyant) sütunu gerekir.")
        else:
            val = st.selectbox("Metrik (sayısal)", num_cols, key="ab_val")
            grp = st.selectbox("Varyant sütunu (2 düzeyli)", cat_cols, key="ab_grp")
            levels = df[grp].dropna().unique().tolist()
            if len(levels) >= 2:
                g1, g2 = st.columns(2)
                l1 = g1.selectbox("A (kontrol)", levels, key="ab_l1")
                l2 = g2.selectbox("B (varyant)", [x for x in levels if x != l1], key="ab_l2")
                use_cuped = st.checkbox("CUPED ile varyans azalt (deney öncesi metrik gerekir)", key="ab_cuped")
                pre_col = None
                if use_cuped:
                    pre_col = st.selectbox("Öncül (pre) metrik", [c for c in num_cols if c != val], key="ab_pre")
                if st.button("Analiz et", type="primary"):
                    res = ab_test.analyze_continuous(df.loc[df[grp] == l1, val], df.loc[df[grp] == l2, val],
                                                     alpha, labels=(str(l1), str(l2)))
                    bf = px.box(df[df[grp].isin([l1, l2])], x=grp, y=val, points="all")
                    bf.update_layout(height=300, margin=dict(t=20))
                    st.plotly_chart(bf, use_container_width=True)
                    interp_box(ab_test.interpret_analysis(res))
                    if use_cuped and pre_col:
                        cup = experiments.cuped_ab(df[df[grp].isin([l1, l2])], val, pre_col, grp, l1, l2, alpha)
                        st.markdown("#### 🎯 CUPED varyans azaltma")
                        cc1, cc2, cc3 = st.columns(3)
                        cc1.metric("Varyans azalması", f"%{cup['var_reduction']*100:.1f}")
                        cc2.metric("p (ham)", f"{cup['p_raw']:.4g}")
                        cc3.metric("p (CUPED)", f"{cup['p_cuped']:.4g}")
                        interp_box(experiments.interpret_cuped(cup))
                    claude_expander(str(res), key="ab_cont_claude")

# ===== 26. BAYESÇİ =====
with tabs[25]:
    st.subheader("📿 Bayesçi analiz (Bayes faktörü)")
    st.caption("Klasik p-değerinin ötesinde: kanıtın gücünü doğrudan ölçer — ve 'etki YOK' (H₀) lehine de kanıt sunabilir.")
    by_mode = st.radio("Analiz", ["Bayesçi t-testi", "Bayesçi korelasyon"], horizontal=True)

    if by_mode == "Bayesçi t-testi":
        sub = st.selectbox("Tür", ["İki bağımsız grup", "Eşleştirilmiş", "Tek örneklem"])
        if sub == "Tek örneklem":
            col = st.selectbox("Değişken", num_cols, key="by_col1")
            mu = st.number_input("Test edilecek ortalama (µ₀)", value=0.0, key="by_mu")
            if st.button("Hesapla", type="primary"):
                res = bayesian.bayesian_ttest(df[col], mu=mu, alpha=alpha)
                st.metric("Bayes faktörü (BF₁₀)", f"{res['bf10']:.3g}")
                interp_box(bayesian.interpret_bayes_ttest(res))
                claude_expander(str(res), key="by_t1_claude")
        elif sub == "Eşleştirilmiş":
            c1, c2 = st.columns(2)
            a = c1.selectbox("1. ölçüm", num_cols, key="by_a")
            b = c2.selectbox("2. ölçüm", [c for c in num_cols if c != a], key="by_b")
            if st.button("Hesapla", type="primary"):
                res = bayesian.bayesian_ttest(df[a], df[b], paired=True, alpha=alpha)
                st.metric("Bayes faktörü (BF₁₀)", f"{res['bf10']:.3g}")
                interp_box(bayesian.interpret_bayes_ttest(res))
                claude_expander(str(res), key="by_t2_claude")
        else:
            val = st.selectbox("Sayısal değişken", num_cols, key="by_val")
            grp = st.selectbox("Grup değişkeni (2 düzeyli)", cat_cols, key="by_grp") if cat_cols else None
            if grp:
                levels = df[grp].dropna().unique().tolist()
                if len(levels) >= 2:
                    g1, g2 = st.columns(2)
                    l1 = g1.selectbox("1. grup", levels, key="by_l1")
                    l2 = g2.selectbox("2. grup", [x for x in levels if x != l1], key="by_l2")
                    if st.button("Hesapla", type="primary"):
                        res = bayesian.bayesian_ttest(df.loc[df[grp] == l1, val], df.loc[df[grp] == l2, val], alpha=alpha)
                        st.metric("Bayes faktörü (BF₁₀)", f"{res['bf10']:.3g}")
                        interp_box(bayesian.interpret_bayes_ttest(res))
                        claude_expander(str(res), key="by_t3_claude")
            else:
                st.warning("Kategorik grup sütunu yok.")
    else:
        if len(num_cols) < 2:
            st.warning("Korelasyon için 2 sayısal sütun gerekir.")
        else:
            c1, c2 = st.columns(2)
            x = c1.selectbox("X", num_cols, key="by_cx")
            y = c2.selectbox("Y", [c for c in num_cols if c != x], key="by_cy")
            if st.button("Hesapla", type="primary"):
                res = bayesian.bayesian_correlation(df[x], df[y], alpha)
                st.metric("Bayes faktörü (BF₁₀)", f"{res['bf10']:.3g}")
                interp_box(bayesian.interpret_bayes_corr(res))
                claude_expander(str(res), key="by_corr_claude")

# ===== 27. OTOMATİK İÇGÖRÜ =====
with tabs[26]:
    st.subheader("💡 Otomatik içgörü — veriden anlam çıkar")
    st.caption("Tek tıkla: hangi analizi yapacağını bilmene gerek yok. Kayda değer örüntüleri önem sırasıyla bulur.")
    if df.empty:
        st.info("Önce sol menüden veri yükleyin.")
    else:
        if st.button("🔎 Verimi analiz et", type="primary"):
            st.session_state["auto_ins"] = insights.auto_insights(df, alpha)
            st.session_state["auto_prof"] = insights.profile(df)
            st.session_state["auto_flags"] = insights.quality_flags(df)
        ins = st.session_state.get("auto_ins")
        if ins is not None:
            st.markdown("#### 📊 Bulgular (ham parametreler)")
            if ins:
                fact_df = pd.DataFrame([{"Tür": i["tur"], "Bulgu": i["fact"], "Önem": round(i["onem"], 2)} for i in ins])
                st.dataframe(fact_df, use_container_width=True, hide_index=True)
            else:
                st.caption("Güçlü bir örüntü bulunamadı.")
            # --- ayrı yorum katmanı ---
            il = ["### Ne anlama geliyor & ne yapmalı"]
            for i in ins:
                il.append(f"**{i['fact']}**")
                il.append(I.bullet(i["yorum"]))
                il.append(I.bullet("➡️ Öneri: " + i["oneri"]))
                il.append("")
            interp_box(I.joinlines(il) if ins else "Belirgin örüntü yok; testleri elle deneyebilirsin.")

# ===== 28. VERİ PROFİLİ & KALİTE =====
with tabs[27]:
    st.subheader("📋 Veri profili & kalite")
    if df.empty:
        st.info("Önce sol menüden veri yükleyin.")
    else:
        prof = insights.profile(df)
        flags = insights.quality_flags(df)
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Satır", prof["n_rows"]); m2.metric("Sütun", prof["n_cols"])
        m3.metric("Ort. eksik", f"%{prof['total_missing_pct']}"); m4.metric("Yinelenen satır", prof["duplicates"])
        st.markdown("#### 📊 Sütun profili (ham)")
        st.dataframe(prof["columns"].style.format(precision=3), use_container_width=True, hide_index=True)
        if flags:
            st.markdown("#### 🚩 Kalite bayrakları (ham)")
            fdf = pd.DataFrame([{"Seviye": f["seviye"], "Sütun": f["sütun"], "Bulgu": f["fact"],
                                 "Öneri": f["oneri"]} for f in flags])
            st.dataframe(fdf, use_container_width=True, hide_index=True)
        interp_box(insights.summarize_findings(prof, flags, insights.auto_insights(df, alpha, 6)))

# ===== 29. VARSAYIM BEKÇİSİ =====
with tabs[28]:
    st.subheader("✅ Varsayım & geçerlilik bekçisi")
    st.caption("Bir test/model varsayımlarını karşılıyor mu? Sonuca ne kadar güvenmeli? Kontroller ayrı, güven skoru ayrı.")
    if df.empty:
        st.info("Önce sol menüden veri yükleyin.")
    else:
        atype = st.radio("Analiz", ["t-testi / iki grup", "ANOVA / çok grup", "Regresyon (son çalıştırılan)"],
                         horizontal=True)
        res = None
        if atype == "t-testi / iki grup" and num_cols and cat_cols:
            v = st.selectbox("Sayısal değişken", num_cols, key="asm_v")
            g = st.selectbox("Grup (2 düzeyli)", cat_cols, key="asm_g")
            lv = df[g].dropna().unique().tolist()
            if len(lv) >= 2 and st.button("Kontrol et", type="primary", key="asm_btn_t"):
                res = assumptions.for_ttest(df.loc[df[g] == lv[0], v], df.loc[df[g] == lv[1], v], alpha)
        elif atype == "ANOVA / çok grup" and num_cols and cat_cols:
            v = st.selectbox("Sayısal değişken", num_cols, key="asm_av")
            g = st.selectbox("Grup", cat_cols, key="asm_ag")
            if st.button("Kontrol et", type="primary", key="asm_btn_a"):
                res = assumptions.for_anova([grp[v] for _, grp in df.groupby(g)], alpha)
        elif atype == "Regresyon (son çalıştırılan)":
            reg = st.session_state.get("reg_res")
            if reg and "error" not in reg:
                res = assumptions.for_regression(reg["resid"], reg["diag"], alpha)
            else:
                st.info("Önce Regresyon sekmesinden bir model çalıştırın.")
        else:
            st.warning("Bu kontrol için uygun sütunlar yok.")
        if res is not None and not res["checks"].empty:
            st.markdown("#### 📊 Varsayım kontrolleri (ham)")
            st.dataframe(res["checks"], use_container_width=True, hide_index=True)
            st.metric("Güven skoru", f"{res['score']}/100")
            interp_box(assumptions.interpret_score(res))

# ===== 30. NEDENSELLİK =====
with tabs[29]:
    st.subheader("🔗 Nedensellik — korelasyondan nedene")
    st.caption("'X gerçekten Y'yi mi etkiliyor, yoksa sadece ilişkili mi?' Etki tahminleri ayrı, nedensel yorum ayrı.")
    if df.empty:
        st.info("Önce sol menüden veri yükleyin.")
    else:
        cmode = st.radio("Yöntem", ["Karıştırıcı kontrolü", "Fark-içinde-fark (DiD)", "Eğilim skoru eşleştirme"],
                         horizontal=True)
        if cmode == "Karıştırıcı kontrolü":
            if len(num_cols) < 3:
                st.warning("En az 3 sayısal sütun gerekir (X, Y, karıştırıcı).")
            else:
                c1, c2 = st.columns(2)
                x = c1.selectbox("X (neden?)", num_cols, key="cf_x")
                y = c2.selectbox("Y (sonuç)", [c for c in num_cols if c != x], key="cf_y")
                conf = st.multiselect("Karıştırıcı(lar)", [c for c in num_cols if c not in (x, y)],
                                      default=[c for c in num_cols if c not in (x, y)][:1], key="cf_conf")
                if conf and st.button("Kontrol et", type="primary", key="cf_btn"):
                    r = causal.confounding_check(df, x, y, conf, alpha)
                    st.markdown("#### 📊 Katsayılar (ham)")
                    st.dataframe(pd.DataFrame({"": ["Ham (kontrolsüz)", "Kontrol sonrası"],
                                 "β": [r["b_raw"], r["b_adj"]]}).style.format({"β": "{:.4f}"}),
                                 use_container_width=True, hide_index=True)
                    interp_box(causal.interpret_confounding(r))
        elif cmode == "Fark-içinde-fark (DiD)":
            if not num_cols or len(cat_cols) < 2:
                st.warning("Bir sonuç (sayısal), bir grup ve bir zaman sütunu gerekir.")
            else:
                c1, c2, c3 = st.columns(3)
                out = c1.selectbox("Sonuç (Y)", num_cols, key="did_y")
                grp = c2.selectbox("Grup (tedavi/kontrol)", cat_cols, key="did_g")
                tim = c3.selectbox("Zaman (öncesi/sonrası)", [c for c in cat_cols if c != grp], key="did_t")
                gl = df[grp].dropna().unique().tolist(); tl = df[tim].dropna().unique().tolist()
                if len(gl) >= 2 and len(tl) >= 2:
                    cc1, cc2 = st.columns(2)
                    tlab = cc1.selectbox("Tedavi düzeyi", gl, key="did_tl")
                    plab = cc2.selectbox("'Sonrası' düzeyi", tl, key="did_pl")
                    if st.button("DiD çalıştır", type="primary", key="did_btn"):
                        r = causal.difference_in_differences(df, out, grp, tim, tlab, plab, alpha)
                        if "error" in r:
                            st.error(r["error"])
                        else:
                            st.markdown("#### 📊 Grup × zaman ortalamaları (ham)")
                            st.dataframe(r["means"].style.format(precision=3), use_container_width=True)
                            interp_box(causal.interpret_did(r))
        else:
            if len(num_cols) < 2 or not cat_cols:
                st.warning("İkili tedavi + sonuç + ortak değişkenler gerekir.")
            else:
                tcol = st.selectbox("Tedavi (0/1 ya da ikili)", num_cols + cat_cols, key="psm_t")
                out = st.selectbox("Sonuç (sayısal)", [c for c in num_cols if c != tcol], key="psm_y")
                covs = st.multiselect("Ortak değişkenler (karıştırıcı)",
                                      [c for c in num_cols if c not in (tcol, out)],
                                      default=[c for c in num_cols if c not in (tcol, out)][:2], key="psm_c")
                if covs and st.button("Eşleştir ve tahmin et", type="primary", key="psm_btn"):
                    dfp = df.copy()
                    if tcol in cat_cols:
                        lv = dfp[tcol].dropna().unique().tolist()[:2]
                        dfp[tcol] = (dfp[tcol].astype(str) == str(lv[-1])).astype(int)
                    r = causal.propensity_matching(dfp, tcol, out, covs, alpha)
                    if "error" in r:
                        st.error(r["error"])
                    else:
                        st.markdown("#### 📊 Etki tahminleri (ham)")
                        st.dataframe(pd.DataFrame({"": ["Ham fark", "Eşleştirme sonrası (ATT)"],
                                     "Etki": [r["naive"], r["att"]]}).style.format({"Etki": "{:.4f}"}),
                                     use_container_width=True, hide_index=True)
                        interp_box(causal.interpret_psm(r))

# ===== 31. YÖNETİCİ ÖZETİ =====
with tabs[30]:
    st.subheader("📝 Yönetici özeti — analizlerini tek hikayeye dönüştür")
    st.caption("Oturumda çalıştırdığın analizleri + otomatik içgörüleri toplayıp düz-dil bir anlatı üretir.")
    if df.empty:
        st.info("Önce sol menüden veri yükleyin ve birkaç analiz çalıştırın.")
    else:
        if st.button("📝 Özeti oluştur", type="primary"):
            nar = summary_engine.build_narrative(df, dict(st.session_state), alpha)
            st.session_state["narrative"] = nar
        nar = st.session_state.get("narrative")
        if nar:
            st.caption(f"Bu oturumda ~{nar['n_analiz']} analiz sonucu ve otomatik içgörüler kullanıldı.")
            md = summary_engine.render_narrative(nar)
            st.markdown(md)
            st.download_button("⬇️ Özeti indir (.md)", md.encode("utf-8"),
                               "StatLab_yonetici_ozeti.md", "text/markdown")
            claude_expander("Yönetici özeti taslağı:\n" + md, key="narrative_claude")

# ===== 35. MAKRO & FAİZ =====
with tabs[34]:
    st.subheader("🏦 Makro & faiz — getiri eğrisi, VIX, beklentiler")
    st.caption("ABD Hazine getirileri (3A/5Y/10Y/30Y) + korku endeksi VIX. Faizler tüm varlık fiyatlamasının çıpasıdır.")
    mper = st.selectbox("Dönem", ["1y", "2y", "5y", "10y"], index=1, key="macro_period")
    if st.button("Makro veriyi çek", type="primary"):
        with st.spinner("Faiz ve VIX verisi çekiliyor..."):
            try:
                st.session_state["macro"] = finance.fetch_macro(mper)
            except Exception as e:
                st.error(f"Veri çekilemedi: {e}")
    mac = st.session_state.get("macro")
    if mac:
        cols = st.columns(len(mac["yields"]) + 2)
        for i, (k, v) in enumerate(mac["yields"].items()):
            cols[i].metric(k, f"%{v:.2f}")
        cols[-2].metric("Eğim 10Y−3M", f"{mac['slope']:+.2f}")
        cols[-1].metric("VIX", f"{mac['vix']:.1f}")
        # getiri eğrisi
        yc = pd.DataFrame({"Vade": list(mac["yields"].keys()), "Getiri %": list(mac["yields"].values())})
        cfig = px.line(yc, x="Vade", y="Getiri %", markers=True, title="Güncel getiri eğrisi")
        cfig.update_layout(height=320, margin=dict(t=40))
        st.plotly_chart(cfig, use_container_width=True)
        # tarihsel 10Y ve VIX
        hist = mac["history"]
        if "10 Yıl" in hist.columns:
            hfig = px.line(hist[[c for c in ["10 Yıl", "3 Ay"] if c in hist.columns]].dropna(),
                           title="Faiz geçmişi (10Y vs 3A)")
            hfig.update_layout(height=300, margin=dict(t=40), legend_title="")
            st.plotly_chart(hfig, use_container_width=True)
        interp_box(finance.interpret_macro(mac))
        claude_expander(f"Makro: {mac['yields']}, eğim={mac['slope']}, VIX={mac['vix']}. "
                        "Güncel ekonomik-politik bağlam varsa yorumla.", key="macro_claude")

# ===== 36. FUNNEL =====
with tabs[31]:
    st.subheader("🚰 Dönüşüm hunisi (funnel)")
    st.caption("Aşamalar arası dönüşüm ve kayıp. En büyük düşüşün olduğu adım = en yüksek iyileştirme fırsatı.")
    fmode = st.radio("Veri kaynağı", ["Elle gir", "Sütunlardan (0/1 aşamalar)"], horizontal=True)
    if fmode == "Elle gir":
        st.caption("Her aşama ve adedini gir (üstten alta, en geniş aşama en üstte).")
        default = pd.DataFrame({"Aşama": ["Ziyaret", "Sepet", "Ödeme", "Satın alma"],
                                "Adet": [10000, 3200, 1500, 1100]})
        ed = st.data_editor(default, num_rows="dynamic", use_container_width=True, key="funnel_ed")
        if st.button("Huniyi analiz et", type="primary"):
            ed = ed.dropna()
            res = marketing.funnel_from_counts(ed["Aşama"].tolist(), ed["Adet"].tolist())
            _render_funnel(res)
    else:
        if not num_cols:
            st.warning("0/1 aşama sütunları gerekir.")
        else:
            scols = st.multiselect("Aşama sütunları (sırayla)", num_cols, key="funnel_cols")
            if len(scols) >= 2 and st.button("Huniyi analiz et", type="primary"):
                res = marketing.funnel_from_columns(df, scols)
                _render_funnel(res)

# ===== 37. KOHORT / RETENTION =====
with tabs[32]:
    st.subheader("📅 Kohort & elde tutma (retention)")
    st.caption("Kullanıcıların zaman içinde geri dönme oranı. Kohort = aynı dönemde başlayanlar.")
    if df.empty:
        st.info("Kullanıcı-olay verisi yükleyin (kimlik + tarih sütunları).")
    else:
        kc1, kc2, kc3 = st.columns(3)
        idc = kc1.selectbox("Kullanıcı kimliği", list(df.columns), key="coh_id")
        evd = kc2.selectbox("Olay tarihi", list(df.columns), key="coh_ev")
        per = kc3.selectbox("Dönem", ["D (gün)", "W (hafta)", "M (ay)"], index=2, key="coh_per")
        pmap = {"D (gün)": "D", "W (hafta)": "W", "M (ay)": "M"}
        if st.button("Retention hesapla", type="primary"):
            try:
                res = marketing.cohort_retention(df, idc, evd, None, pmap[per], 12)
                st.session_state["coh_res"] = res
            except Exception as e:
                st.error(f"Hesaplanamadı: {e}")
        res = st.session_state.get("coh_res")
        if res is not None and not res["retention"].empty:
            st.markdown("#### 📊 Elde tutma matrisi (ham, %)")
            st.dataframe((res["retention"] * 100).round(1).style.background_gradient(cmap="Greens", axis=None),
                         use_container_width=True)
            hm = px.imshow(res["retention"].values, x=list(res["retention"].columns),
                           y=list(res["retention"].index), color_continuous_scale="Greens",
                           text_auto=".0%", aspect="auto", title="Kohort retention ısı haritası")
            hm.update_layout(height=360, margin=dict(t=40))
            st.plotly_chart(hm, use_container_width=True)
            interp_box(marketing.interpret_retention(res))

# ===== 38. RFM =====
with tabs[33]:
    st.subheader("🎯 RFM segmentasyonu")
    st.caption("Müşterileri Recency (son alışveriş), Frequency (sıklık), Monetary (harcama) ile segmentlere ayırır.")
    if df.empty:
        st.info("Müşteri işlem verisi yükleyin (kimlik + tarih + tutar).")
    else:
        r1, r2, r3 = st.columns(3)
        idc = r1.selectbox("Müşteri kimliği", list(df.columns), key="rfm_id")
        dtc = r2.selectbox("İşlem tarihi", list(df.columns), key="rfm_dt")
        vlc = r3.selectbox("Tutar", num_cols, key="rfm_v") if num_cols else None
        if vlc and st.button("RFM hesapla", type="primary"):
            try:
                res = marketing.rfm(df, idc, dtc, vlc)
                st.session_state["rfm_res"] = res
            except Exception as e:
                st.error(f"Hesaplanamadı: {e}")
        res = st.session_state.get("rfm_res")
        if res is not None:
            pie = px.pie(names=res["segments"].index, values=res["segments"].values,
                         title="Müşteri segmentleri")
            pie.update_layout(height=340, margin=dict(t=40))
            st.plotly_chart(pie, use_container_width=True)
            st.markdown("#### 📊 Müşteri RFM tablosu (ham, ilk 50)")
            st.dataframe(res["table"].head(50), use_container_width=True, hide_index=True)
            interp_box(marketing.interpret_rfm(res))

# ===== 39. DOĞRULAMA (CV) =====
with tabs[35]:
    st.subheader("✔️ Örneklem-dışı doğrulama (çapraz doğrulama / holdout)")
    st.caption("Modelin GÖRMEDİĞİ veride gerçekte ne kadar iyi olacağını ölçer — aşırı uyumu (overfitting) yakalar. "
               "Raporlanması gereken dürüst metrik budur.")
    if len(num_cols) < 2:
        st.warning("Doğrulama için sayısal sütunlar gerekir.")
    else:
        vtask = st.radio("Model", ["Regresyon (OLS)", "Lojistik"], horizontal=True)
        vc1, vc2 = st.columns(2)
        if vtask == "Regresyon (OLS)":
            vy = vc1.selectbox("Bağımlı (Y)", num_cols, key="cv_y")
            vx = st.multiselect("Öngörücüler (X)", [c for c in num_cols if c != vy],
                                default=[c for c in num_cols if c != vy][:3], key="cv_x")
            folds = vc2.slider("Kat sayısı (k)", 3, 10, 5, key="cv_folds")
            if vx and st.button("Çapraz doğrula", type="primary", key="cvreg_btn"):
                res = validation.cv_regression(df, vy, vx, folds)
                if "error" in res:
                    st.error(res["error"])
                else:
                    k1, k2, k3 = st.columns(3)
                    k1.metric("R² (örneklem-içi)", f"{res['r2_in']:.3f}")
                    k2.metric("R² (çapraz doğrulama)", f"{res['r2_cv']:.3f}", delta=f"{-res['gap']:.3f}")
                    k3.metric("CV RMSE", f"{res['rmse_cv']:.3f}")
                    interp_box(validation.interpret_cv(res))
                    ho = validation.holdout(df, vy, vx, "regresyon")
                    interp_box(validation.interpret_holdout(ho))
                    claude_expander(str({k: v for k, v in res.items() if k not in ("pred", "actual", "fold_scores")}), key="cvreg_claude")
        else:
            targets = [c for c in df.columns if df[c].nunique(dropna=True) == 2]
            if not targets:
                st.warning("İkili hedef sütun gerekir.")
            else:
                vy = vc1.selectbox("İkili hedef", targets, key="cvl_y")
                pos = vc2.selectbox("Pozitif sınıf", df[vy].dropna().unique().tolist(), key="cvl_pos")
                vx = st.multiselect("Öngörücüler (sayısal)", [c for c in num_cols if c != vy],
                                    default=[c for c in num_cols if c != vy][:3], key="cvl_x")
                folds = st.slider("Kat sayısı (k)", 3, 10, 5, key="cvl_folds")
                if vx and st.button("Çapraz doğrula", type="primary", key="cvlog_btn"):
                    res = validation.cv_logistic(df, vy, vx, pos, folds)
                    if "error" in res:
                        st.error(res["error"])
                    else:
                        k1, k2, k3 = st.columns(3)
                        k1.metric("AUC (örneklem-içi)", f"{res['auc_in']:.3f}")
                        k2.metric("AUC (çapraz doğrulama)", f"{res['auc_cv']:.3f}", delta=f"{-res['gap']:.3f}")
                        k3.metric("CV doğruluk", f"{res['acc_cv']:.3f}")
                        interp_box(validation.interpret_cv(res))
                        claude_expander(str({k: v for k, v in res.items() if k not in ("proba", "ytrue")}), key="cvlog_claude")

# ===== 40. ARACILIK / DÜZENLEYİCİLİK =====
with tabs[36]:
    st.subheader("🔀 Aracılık & düzenleyicilik")
    st.caption("Aracılık: X→M→Y (mekanizma — 'neden/nasıl?'). Düzenleyicilik: X'in etkisi W'ye göre değişiyor mu?")
    if len(num_cols) < 3:
        st.warning("En az 3 sayısal sütun gerekir.")
    else:
        mmode = st.radio("Analiz", ["Aracılık (mediation)", "Düzenleyicilik (moderation)"], horizontal=True)
        if mmode == "Aracılık (mediation)":
            c1, c2, c3 = st.columns(3)
            mx = c1.selectbox("X (bağımsız)", num_cols, key="med_x")
            mm = c2.selectbox("M (aracı)", [c for c in num_cols if c != mx], key="med_m")
            my = c3.selectbox("Y (sonuç)", [c for c in num_cols if c not in (mx, mm)], key="med_y")
            if st.button("Aracılık analizi", type="primary", key="med_btn"):
                res = mediation.mediation(df, mx, mm, my, alpha)
                if "error" in res:
                    st.error(res["error"])
                else:
                    st.markdown("#### 📊 Yollar (ham)")
                    st.dataframe(res["table"], use_container_width=True, hide_index=True)
                    interp_box(mediation.interpret_mediation(res))
                    claude_expander(f"Aracılık {mx}→{mm}→{my}", key="med_claude")
        else:
            c1, c2, c3 = st.columns(3)
            mx = c1.selectbox("X (bağımsız)", num_cols, key="mod_x")
            mw = c2.selectbox("W (düzenleyici)", [c for c in num_cols if c != mx], key="mod_w")
            my = c3.selectbox("Y (sonuç)", [c for c in num_cols if c not in (mx, mw)], key="mod_y")
            if st.button("Düzenleyicilik analizi", type="primary", key="mod_btn"):
                res = mediation.moderation(df, mx, mw, my, alpha)
                if "error" in res:
                    st.error(res["error"])
                else:
                    st.markdown("#### 📊 Basit eğimler (ham)")
                    st.dataframe(pd.DataFrame({"Koşul": list(res["slopes"].keys()),
                                 "X→Y eğimi": list(res["slopes"].values())}).style.format({"X→Y eğimi": "{:.4f}"}),
                                 use_container_width=True, hide_index=True)
                    interp_box(mediation.interpret_moderation(res))
                    claude_expander(str(res), key="mod_claude")

# ===== 41. BACKTEST =====
with tabs[37]:
    st.subheader("🔁 Portföy backtest (geçmiş simülasyon)")
    rets = st.session_state.get("fin_returns")
    if rets is None or rets.shape[1] < 2:
        st.info("Önce '📈 Getiri & Risk' sekmesinden en az 2 sembol çek (biri benchmark olabilir, ör. SPY).")
    else:
        st.caption("Verilen ağırlıklarla portföyü geçmişte simüle et; periyodik yeniden dengeleme + benchmark karşılaştırması.")
        assets = list(rets.columns)
        default_w = {a: round(1 / len(assets), 2) for a in assets}
        wdf = pd.DataFrame({"Varlık": assets, "Ağırlık": [default_w[a] for a in assets]})
        wed = st.data_editor(wdf, use_container_width=True, key="bt_weights", hide_index=True)
        bc1, bc2 = st.columns(2)
        reb = bc1.selectbox("Yeniden dengeleme", ["Buy&Hold", "M", "Q", "Y"],
                            format_func=lambda r: {"Buy&Hold": "Al-tut", "M": "Aylık", "Q": "Çeyreklik", "Y": "Yıllık"}[r])
        bench = bc2.selectbox("Benchmark (ops.)", ["(yok)"] + assets)
        if st.button("Backtest çalıştır", type="primary"):
            weights = dict(zip(wed["Varlık"], pd.to_numeric(wed["Ağırlık"], errors="coerce").fillna(0)))
            res = finance.backtest(rets, weights, reb, None if bench == "(yok)" else bench)
            eqdf = pd.DataFrame({"Portföy": res["equity"]})
            if "bench_equity" in res:
                eqdf[bench] = res["bench_equity"]
            ef = px.line(eqdf, title="Portföy değeri (backtest)")
            ef.update_layout(height=360, margin=dict(t=40), legend_title="")
            st.plotly_chart(ef, use_container_width=True)
            m = res["metrics"]
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("CAGR", f"%{m['CAGR']*100:.1f}"); k2.metric("Sharpe", f"{m['sharpe']:.2f}")
            k3.metric("Maks. düşüş", f"%{m['max_dd']*100:.1f}")
            if "info_ratio" in res:
                k4.metric("Bilgi oranı", f"{res['info_ratio']:.2f}")
            interp_box(finance.interpret_backtest(res))
            claude_expander(str({"metrics": res["metrics"], "ir": res.get("info_ratio")}), key="bt_claude")

# ===== 42. TAHVİL =====
with tabs[38]:
    st.subheader("💵 Tahvil analizi (fiyat, süre, konveksite)")
    st.caption("Bir tahvili tek tek analiz et: fiyat, YTM, süre (duration = faiz duyarlılığı), konveksite.")
    bc1, bc2, bc3 = st.columns(3)
    face = bc1.number_input("Nominal değer", 100, 1_000_000, 1000, step=100)
    coupon = bc2.number_input("Kupon oranı %", 0.0, 30.0, 5.0, 0.25) / 100
    byears = bc3.number_input("Vade (yıl)", 0.5, 50.0, 10.0, 0.5)
    bc4, bc5 = st.columns(2)
    bytm = bc4.number_input("Piyasa getirisi YTM %", 0.1, 30.0, 4.5, 0.1) / 100
    bfreq = bc5.selectbox("Kupon sıklığı", [1, 2, 4], index=1,
                          format_func=lambda f: {1: "Yıllık", 2: "6 Aylık", 4: "3 Aylık"}[f])
    if st.button("Tahvili analiz et", type="primary"):
        res = finance.bond_analytics(face, coupon, byears, bytm, bfreq)
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Fiyat", f"{res['price']:.2f}")
        k2.metric("Değiştirilmiş süre", f"{res['modified']:.2f}")
        k3.metric("Konveksite", f"{res['convexity']:.1f}")
        k4.metric("Cari getiri", f"%{res['current_yield']*100:.2f}")
        # faiz-fiyat duyarlılığı eğrisi
        dys = np.linspace(-0.03, 0.03, 25)
        chg = [finance.bond_price_change(res, dy) * 100 for dy in dys]
        sfig = px.line(x=dys * 100, y=chg, labels={"x": "Getiri değişimi (bp/100)", "y": "Fiyat değişimi %"},
                       title="Faiz duyarlılığı (süre + konveksite)")
        sfig.update_layout(height=320, margin=dict(t=40))
        st.plotly_chart(sfig, use_container_width=True)
        interp_box(finance.interpret_bond(res))
        claude_expander(str(res), key="bond_claude")

st.divider()
st.caption("StatLab · Python · statsmodels · scipy · Streamlit — otomatik Türkçe yorumlama ile")
