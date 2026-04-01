#!/usr/bin/env bash
# Build .deb package for e-clawhisper.
# Requirements: uv, nfpm (go install github.com/goreleaser/nfpm/v2/cmd/nfpm@latest)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
DIST_DIR="$PROJECT_DIR/dist"
ARCH="${ARCH:-amd64}"

cd "$PROJECT_DIR"

##### VERSION #####

VERSION="$(git describe --tags --always 2>/dev/null || echo "0.0.0")"
VERSION="${VERSION#v}"

echo "==> Building e-clawhisper $VERSION ($ARCH)"

##### CLEAN #####

rm -rf "$DIST_DIR"
mkdir -p "$DIST_DIR"

STAGING="$DIST_DIR/.staging"

##### VENV #####

echo "==> Creating production venv..."

uv venv "$STAGING/.venv" --python 3.13 --relocatable --quiet
VIRTUAL_ENV="$STAGING/.venv" uv pip install "$PROJECT_DIR" --quiet

# Make venv self-contained: copy Python binary + stdlib (venv symlinks to ~/.local/share/uv/)
PYTHON_LINK="$STAGING/.venv/bin/python"
PYTHON_REAL="$(readlink -f "$PYTHON_LINK")"
PYTHON_HOME="$(dirname "$(dirname "$PYTHON_REAL")")"

# Copy real binary over symlink
rm -f "$PYTHON_LINK"
cp "$PYTHON_REAL" "$PYTHON_LINK"
chmod 755 "$PYTHON_LINK"

# Copy stdlib into venv (encodings, os, asyncio, etc.) — exclude site-packages (venv owns it)
echo "==> Bundling Python stdlib from $PYTHON_HOME..."
rsync -a --exclude='site-packages' --exclude='__pycache__' \
    "$PYTHON_HOME/lib/python3.13/" "$STAGING/.venv/lib/python3.13/"

# Update pyvenv.cfg so Python finds stdlib in the venv itself
sed -i "s|^home = .*|home = /opt/e-clawhisper/.venv/bin|" "$STAGING/.venv/pyvenv.cfg"

echo "==> Raw venv size: $(du -sh "$STAGING/.venv" | cut -f1)"

##### PRUNE #####

echo "==> Pruning unnecessary packages..."
SITE="$STAGING/.venv/lib/python3.13/site-packages"

# torch / nvidia / triton — we use ONNX Runtime, not PyTorch
# boto3 / botocore / s3transfer — AWS SDK pulled transitively, unused
# sympy — torch compile dep, unused
PRUNE_DIRS=(
    nvidia torch triton
    boto3 botocore s3transfer
    sympy
)
for pkg in "${PRUNE_DIRS[@]}"; do
    rm -rf "$SITE/$pkg" "$SITE/${pkg}.libs" "$SITE/${pkg}"[-_]*.dist-info
done

# Remove __pycache__ and .pyc files
find "$STAGING/.venv" -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find "$STAGING/.venv" -name "*.pyc" -delete 2>/dev/null || true

echo "==> Pruned venv size: $(du -sh "$STAGING/.venv" | cut -f1)"

##### PACKAGE #####

echo "==> Building .deb with nfpm..."
cd "$SCRIPT_DIR"

VERSION="$VERSION" ARCH="$ARCH" nfpm package \
    --config nfpm.yaml \
    --packager deb \
    --target "$DIST_DIR/"

##### CLEAN STAGING #####

rm -rf "$STAGING"

DEB_FILE=$(ls -1t "$DIST_DIR/"*.deb 2>/dev/null | head -1)
echo "==> Package built: $DEB_FILE"
echo "==> Size: $(du -sh "$DEB_FILE" | cut -f1)"
echo ""
echo "Install with: sudo dpkg -i $DEB_FILE"
echo "Then:         sudo apt-get install -f  # resolve system deps"
