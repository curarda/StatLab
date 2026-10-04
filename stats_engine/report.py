"""PDF rapor üretimi — matplotlib PdfPages (Türkçe karakter güvenli, ek bağımlılık yok).

Kullanım:
    records = [ {"title","text","tables":[df...],"figures":[mpl_fig...]}, ... ]
    pdf_bytes = build_report(meta, records)

Grafik yardımcıları (app bunları çağırıp figure üretir):
    fig_scatter_fit, fig_residuals, fig_corr_heatmap, fig_roc, fig_confusion,
    fig_series_forecast, fig_decompose, fig_hist
"""
from __future__ import annotations

import io
import re
import textwrap
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

plt.rcParams["font.family"] = "DejaVu Sans"  # Türkçe glifleri destekler
ACCENT = "#6366f1"


# ---------------- Grafik yardımcıları ----------------

def fig_hist(values, title="Dağılım"):
    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.hist(np.asarray(values, dtype=float), bins=30, color=ACCENT, alpha=0.85, edgecolor="white")
    ax.set_title(title); ax.set_ylabel("Frekans")
    fig.tight_layout()
    return fig


def fig_scatter_fit(fitted, actual, title="Gerçek vs Tahmin"):
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    fitted = np.asarray(fitted); actual = np.asarray(actual)
    ax.scatter(fitted, actual, alpha=0.6, color=ACCENT, s=18)
    lo, hi = float(min(actual.min(), fitted.min())), float(max(actual.max(), fitted.max()))
    ax.plot([lo, hi], [lo, hi], "r--", lw=1)
    ax.set_xlabel("Tahmin"); ax.set_ylabel("Gözlenen"); ax.set_title(title)
    fig.tight_layout()
    return fig


def fig_residuals(fitted, resid, title="Artık grafiği"):
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    ax.scatter(np.asarray(fitted), np.asarray(resid), alpha=0.6, color=ACCENT, s=18)
    ax.axhline(0, color="r", ls="--", lw=1)
    ax.set_xlabel("Tahmin"); ax.set_ylabel("Artık"); ax.set_title(title)
    fig.tight_layout()
    return fig


def fig_corr_heatmap(corr: pd.DataFrame, title="Korelasyon matrisi"):
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(corr.values, vmin=-1, vmax=1, cmap="RdBu_r")
    ax.set_xticks(range(len(corr.columns))); ax.set_xticklabels(corr.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(corr.index))); ax.set_yticklabels(corr.index, fontsize=8)
    for i in range(len(corr.index)):
        for j in range(len(corr.columns)):
            v = corr.values[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7,
                        color="white" if abs(v) > 0.5 else "black")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    return fig


def fig_roc(fpr, tpr, auc, title="ROC eğrisi"):
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.plot(fpr, tpr, color=ACCENT, lw=2, label=f"AUC = {auc:.3f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Rastgele")
    ax.set_xlabel("Yanlış pozitif oranı"); ax.set_ylabel("Doğru pozitif oranı")
    ax.set_title(title); ax.legend(loc="lower right")
    fig.tight_layout()
    return fig


def fig_confusion(cm: dict, title="Karışıklık matrisi"):
    mat = np.array([[cm["TN"], cm["FP"]], [cm["FN"], cm["TP"]]])
    fig, ax = plt.subplots(figsize=(4.5, 4))
    im = ax.imshow(mat, cmap="Blues")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["Tahmin 0", "Tahmin 1"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["Gerçek 0", "Gerçek 1"])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(mat[i, j]), ha="center", va="center",
                    color="white" if mat[i, j] > mat.max() / 2 else "black", fontsize=13)
    ax.set_title(title)
    fig.tight_layout()
    return fig


def fig_series_forecast(values, fitted=None, forecast=None, ci=None, title="Zaman serisi ve tahmin"):
    fig, ax = plt.subplots(figsize=(8, 3.8))
    n = len(values)
    ax.plot(range(n), values, color="#334155", lw=1.2, label="Gözlem")
    if fitted is not None:
        f = np.asarray(fitted)
        ax.plot(range(len(f)), f, color=ACCENT, lw=1, alpha=0.7, label="Uyum")
    if forecast is not None:
        fc = np.asarray(forecast)
        xf = range(n, n + len(fc))
        ax.plot(xf, fc, color="#dc2626", lw=1.6, label="Tahmin")
        if ci is not None:
            ax.fill_between(xf, ci[0], ci[1], color="#dc2626", alpha=0.15)
    ax.set_title(title); ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    return fig


def fig_decompose(trend, seasonal, resid, title="Mevsimsel ayrıştırma"):
    fig, axes = plt.subplots(3, 1, figsize=(8, 6), sharex=True)
    axes[0].plot(trend, color=ACCENT); axes[0].set_ylabel("Trend")
    axes[1].plot(seasonal, color="#0891b2"); axes[1].set_ylabel("Mevsimsel")
    axes[2].plot(resid, color="#64748b"); axes[2].set_ylabel("Artık")
    axes[0].set_title(title)
    fig.tight_layout()
    return fig


# ---------------- Rapor düzeni ----------------

def _clean_md(text: str) -> list[str]:
    """Basit markdown temizliği → düz metin satırları."""
    out = []
    for line in text.split("\n"):
        line = re.sub(r"\*\*(.+?)\*\*", r"\1", line)
        line = line.replace("### ", "").replace("$", "")
        line = line.replace("- ", "• ").replace("•", "•")
        out.append(line)
    return out


def _cover_page(pdf, meta):
    fig = plt.figure(figsize=(8.27, 11.69))  # A4
    fig.text(0.5, 0.72, "StatLab", ha="center", fontsize=42, weight="bold", color=ACCENT)
    fig.text(0.5, 0.66, "İstatistiksel Analiz Raporu", ha="center", fontsize=20)
    fig.text(0.5, 0.55, f"Veri seti: {meta.get('dataset','—')}", ha="center", fontsize=13)
    fig.text(0.5, 0.51, f"Gözlem: {meta.get('rows','—')} satır × {meta.get('cols','—')} sütun",
             ha="center", fontsize=11, color="#475569")
    fig.text(0.5, 0.47, f"Anlamlılık düzeyi α = {meta.get('alpha',0.05)}", ha="center", fontsize=11, color="#475569")
    fig.text(0.5, 0.08, datetime.now().strftime("%d.%m.%Y %H:%M"), ha="center", fontsize=10, color="#94a3b8")
    fig.text(0.5, 0.05, "Python · statsmodels · scipy", ha="center", fontsize=9, color="#94a3b8")
    pdf.savefig(fig); plt.close(fig)


def _text_page(pdf, title, text):
    lines = _clean_md(text)
    wrapped = []
    for ln in lines:
        if not ln.strip():
            wrapped.append("")
            continue
        indent = "   " if ln.startswith("•") else ""
        for i, w in enumerate(textwrap.wrap(ln, width=95) or [""]):
            wrapped.append((indent if i else "") + w)
    # sayfalara böl
    per_page = 46
    for start in range(0, max(1, len(wrapped)), per_page):
        chunk = wrapped[start:start + per_page]
        fig = plt.figure(figsize=(8.27, 11.69))
        fig.text(0.08, 0.95, title, fontsize=15, weight="bold", color=ACCENT)
        y = 0.90
        for ln in chunk:
            weight = "bold" if ln.strip().endswith(")") and ln.strip()[0].isdigit() else "normal"
            fig.text(0.08, y, ln, fontsize=9.5, va="top", family="DejaVu Sans", weight=weight)
            y -= 0.018
        pdf.savefig(fig); plt.close(fig)


def _table_page(pdf, title, df: pd.DataFrame):
    df = df.copy()
    for c in df.columns:
        if pd.api.types.is_float_dtype(df[c]):
            df[c] = df[c].map(lambda v: f"{v:.4g}" if pd.notna(v) else "")
    fig = plt.figure(figsize=(11.69, 8.27))  # yatay A4
    fig.text(0.05, 0.95, title, fontsize=14, weight="bold", color=ACCENT)
    ax = fig.add_axes([0.03, 0.05, 0.94, 0.85]); ax.axis("off")
    tbl = ax.table(cellText=df.values, colLabels=df.columns, loc="upper center", cellLoc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(8); tbl.scale(1, 1.4)
    for (r, c), cell in tbl.get_celld().items():
        if r == 0:
            cell.set_facecolor(ACCENT); cell.set_text_props(color="white", weight="bold")
        elif r % 2 == 0:
            cell.set_facecolor("#f1f5f9")
    pdf.savefig(fig); plt.close(fig)


def build_report(meta: dict, records: list[dict]) -> bytes:
    """records: [{'title','text','tables':[df],'figures':[fig]}] → PDF bytes."""
    buf = io.BytesIO()
    with PdfPages(buf) as pdf:
        _cover_page(pdf, meta)
        for rec in records:
            title = rec.get("title", "Analiz")
            if rec.get("text"):
                _text_page(pdf, title, rec["text"])
            for df in rec.get("tables", []) or []:
                if isinstance(df, pd.DataFrame) and not df.empty:
                    _table_page(pdf, title + " — tablo", df)
            for fig in rec.get("figures", []) or []:
                if fig is not None:
                    pdf.savefig(fig); plt.close(fig)
        d = pdf.infodict()
        d["Title"] = "StatLab İstatistiksel Analiz Raporu"
        d["Author"] = "StatLab"
    buf.seek(0)
    return buf.getvalue()
