"""Claude API ile derin, doğal dilde istatistiksel yorum (opsiyonel)."""
from __future__ import annotations

import os

DEFAULT_MODEL = "claude-sonnet-5"

SYSTEM = (
    "Sen deneyimli bir istatistik danışmanısın. Sana bir istatistiksel analizin sayısal "
    "sonuçları verilecek. Görevin bu sonuçları Türkçe, açık ve akademik ama anlaşılır bir "
    "dille yorumlamak. Şunları yap: (1) sonucun pratik/uygulamalı anlamını açıkla, "
    "(2) istatistiksel anlamlılık ile etki büyüklüğünü ayır, (3) varsayım ihlallerine ve "
    "sınırlılıklara dikkat çek, (4) olası bir sonraki adımları öner. Sayı uydurma; yalnızca "
    "verilen değerleri kullan. Kısa başlıklar ve madde işaretleri kullan."
)


def available() -> tuple[bool, str]:
    if os.environ.get("ANTHROPIC_API_KEY"):
        try:
            import anthropic  # noqa: F401
            return True, "hazır"
        except ImportError:
            return False, "`anthropic` paketi kurulu değil (pip install anthropic)."
    return False, "ANTHROPIC_API_KEY ortam değişkeni tanımlı değil."


def deep_interpret(context_text: str, question: str | None = None,
                   model: str = DEFAULT_MODEL, api_key: str | None = None) -> str:
    key = api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        return "⚠️ ANTHROPIC_API_KEY tanımlı değil. Sol menüden API anahtarı girin."
    try:
        import anthropic
    except ImportError:
        return "⚠️ `anthropic` paketi kurulu değil."
    client = anthropic.Anthropic(api_key=key)
    user = context_text
    if question:
        user += f"\n\nKullanıcının ek sorusu: {question}"
    try:
        resp = client.messages.create(
            model=model,
            max_tokens=1500,
            system=SYSTEM,
            messages=[{"role": "user", "content": user}],
        )
        return "".join(block.text for block in resp.content if getattr(block, "type", "") == "text")
    except Exception as e:
        return f"⚠️ Claude API hatası: {e}"
