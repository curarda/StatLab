"""py2app ile TAM bağımsız .app paketi (DENEYSEL).

Bilimsel yığının (scipy/statsmodels/streamlit) py2app ile paketlenmesi kırılgandır ve
büyük bir .app üretir. Çoğu kullanıcı için önerilen yol, kök dizindeki `build_app.sh`
ile üretilen hafif başlatıcı .app'tir.

Yine de tam bundle denemek için:
    cd /Users/mac/StatLab
    ./.venv/bin/pip install py2app
    ./.venv/bin/python packaging/setup.py py2app

Not: `streamlit run` bir alt süreç başlattığı için tam bundle'da ek yapılandırma
gerekebilir; sorun yaşarsanız başlatıcı .app'i kullanın.
"""
from setuptools import setup

APP = ["../app.py"]
OPTIONS = {
    "argv_emulation": False,
    "packages": ["streamlit", "pandas", "numpy", "scipy", "statsmodels",
                 "plotly", "matplotlib", "sympy", "openpyxl", "stats_engine"],
    "includes": ["stats_engine"],
    "plist": {"CFBundleName": "StatLab", "CFBundleDisplayName": "StatLab",
              "CFBundleIdentifier": "com.statlab.app", "CFBundleVersion": "1.0.0"},
}

setup(
    app=APP,
    name="StatLab",
    options={"py2app": OPTIONS},
    setup_requires=["py2app"],
)
