#!/usr/bin/env bash
# StatLab.app (hafif başlatıcı) oluşturur — çift tıklayınca uygulamayı açar.
set -e
cd "$(dirname "$0")"

echo "StatLab.app derleniyor..."
rm -rf StatLab.app
osacompile -o StatLab.app packaging/StatLab.applescript
echo "✓ StatLab.app hazır: $(pwd)/StatLab.app"
echo "  Çift tıklayarak açabilir ya da Uygulamalar'a taşıyabilirsiniz."
