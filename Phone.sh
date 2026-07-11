#!/data/data/com.termux/files/usr/bin/bash
# =====================================================================
#  Manga Translation API — Termux Installer & Launcher
#  - Auto-downloads latest release from GitHub via wget
#  - Checks for new versions on startup
#  - 'update' command downloads new zip and refreshes deps
# =====================================================================

set -e

# --- Colors ---
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; BOLD='\033[1m'; NC='\033[0m'
log()  { echo -e "${CYAN}[*]${NC} $1"; }
ok()   { echo -e "${GREEN}[✓]${NC} $1"; }
warn() { echo -e "${YELLOW}[!]${NC} $1"; }
err()  { echo -e "${RED}[✗]${NC} $1"; }

# --- Paths & Config ---
WORK_DIR="$HOME/manga-api"
VENV_DIR="$WORK_DIR/venv"
INSTALL_FLAG="$WORK_DIR/.installed_v1"
VERSION_FILE="$WORK_DIR/.version"
REPO_API="https://api.github.com/repos/Kirogii/MangaAMTL/releases/latest"
USER_AGENT="termux-manga-installer"

# =====================================================================
#  STEP 1 — System packages
# =====================================================================
log "Updating package lists..."
pkg update -y >/dev/null 2>&1 || true

log "Installing system dependencies (5-10 minutes on first run)..."

if ! pkg list-installed 2>/dev/null | grep -q "^tur-repo/"; then
    pkg install -y tur-repo >/dev/null 2>&1 || warn "tur-repo not available — continuing"
fi

pkg install -y \
    python python-pip \
    cmake make clang gcc pkg-config \
    openblas openssl openssl-tool zlib \
    libjpeg-turbo libpng \
    freetype fontconfig \
    protobuf git wget unzip \
    rust libffi ndk-sysroot libc++ \
    2>/dev/null || warn "Some system packages failed — continuing"

ok "System packages installed."

# =====================================================================
#  STEP 2 — Working directory & Fetch latest release
# =====================================================================
mkdir -p "$WORK_DIR"
cd "$WORK_DIR"

log "Fetching latest release info from GitHub..."
wget -q -U "$USER_AGENT" "$REPO_API" -O release.json

ASSET_URL=$(python -c "import json; r=json.load(open('release.json')); print(r['assets'][0]['browser_download_url'] if r.get('assets') else '')" 2>/dev/null)
RELEASE_TAG=$(python -c "import json; r=json.load(open('release.json')); print(r.get('tag_name', 'unknown'))" 2>/dev/null)

if [ -n "$ASSET_URL" ]; then
    log "Downloading release package $RELEASE_TAG..."
    wget -q -U "$USER_AGENT" "$ASSET_URL" -O manga_release.zip
    log "Extracting package..."
    unzip -o manga_release.zip -d "$WORK_DIR" >/dev/null 2>&1
    
    # Handle nested folders (if zip contains a root folder, move its contents up)
    if [ ! -f "$WORK_DIR/app.py" ]; then
        APP_LOC=$(find "$WORK_DIR" -name "app.py" | head -n 1)
        if [ -n "$APP_LOC" ]; then
            APP_DIR=$(dirname "$APP_LOC")
            if [ "$APP_DIR" != "$WORK_DIR" ]; then
                cp -r "$APP_DIR"/* "$WORK_DIR/"
                rm -rf "$APP_DIR"
            fi
        fi
    fi
    rm -f manga_release.zip release.json
    echo "$RELEASE_TAG" > "$VERSION_FILE"
    ok "Release package downloaded and extracted (Version: $RELEASE_TAG)."
else
    warn "No release attachment found. Falling back to raw app.py from main branch..."
    wget -q -U "$USER_AGENT" "https://raw.githubusercontent.com/Kirogii/MangaAMTL/main/app.py" -O "$WORK_DIR/app.py"
    rm -f release.json
    echo "dev-raw" > "$VERSION_FILE"
fi

if [ ! -f "$WORK_DIR/app.py" ]; then
    err "Failed to download app.py from GitHub!"
    exit 1
fi
ok "app.py is present."

# Ensure required directories exist
mkdir -p "$WORK_DIR/models" "$WORK_DIR/models/gguf" "$WORK_DIR/models/colorizer" "$WORK_DIR/fonts"
ok "Required directories created."

# =====================================================================
#  STEP 3 — Python virtual environment
# =====================================================================
if [ ! -d "$VENV_DIR" ]; then
    log "Creating virtual environment..."
    python -m venv "$VENV_DIR"
fi
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

log "Upgrading pip / setuptools / wheel..."
pip install --upgrade pip setuptools wheel >/dev/null 2>&1

# =====================================================================
#  STEP 4 — PyTorch
# =====================================================================
if ! python -c "import torch" 2>/dev/null; then
    log "Installing PyTorch (large download — be patient)..."
    pip install torch torchvision torchaudio || {
        warn "Default PyTorch wheel failed — trying Termux wheel index..."
        pip install --index-url https://termux.dev/python-wheels/ \
            torch torchvision torchaudio || {
            err "PyTorch install failed. Aborting."
            exit 1
        }
    }
    ok "PyTorch installed."
else
    ok "PyTorch already installed."
fi

# =====================================================================
#  STEP 5 — OpenCV (headless)
# =====================================================================
if ! python -c "import cv2" 2>/dev/null; then
    log "Installing OpenCV (headless)..."
    pip install opencv-python-headless || {
        warn "Headless wheel failed — trying opencv-python..."
        pip install opencv-python || warn "OpenCV install issue."
    }
    ok "OpenCV installed."
else
    ok "OpenCV already installed."
fi

# =====================================================================
#  STEP 6 — numpy first (pinned)
# =====================================================================
log "Installing pinned numpy..."
pip install "numpy==1.26.4" >/dev/null 2>&1 || warn "numpy 1.26.4 install issue"

# =====================================================================
#  STEP 7 — llama-cpp-python
# =====================================================================
if ! python -c "import llama_cpp" 2>/dev/null; then
    log "Installing llama-cpp-python (compiles from source, 10-20 min)..."
    export CMAKE_ARGS="-DLLAMA_NATIVE=OFF -DLLAMA_AVX=OFF -DLLAMA_AVX2=OFF"
    export CMAKE_GENERATOR="Unix Makefiles"
    pip install llama-cpp-python || warn "llama-cpp-python install issue"
    ok "llama-cpp-python installed."
else
    ok "llama-cpp-python already installed."
fi

# =====================================================================
#  STEP 8 — Remaining Python requirements
# =====================================================================
log "Installing remaining Python packages (10-30 min)..."
if [ -f "$WORK_DIR/requirements.txt" ]; then
    pip install -r "$WORK_DIR/requirements.txt" 2>&1 | tail -20 || warn "Some packages had install issues."
else
    pip install rich betterproto aiohttp fastapi pydantic pillow \
        "ultralytics==8.3.190" simple-lama-inpainting-updated \
        onnxruntime huggingface-hub fugashi unidic-lite \
        python-multipart uvicorn hayai-ocr transformers accelerate \
        2>&1 | tail -20 || warn "Some packages had install issues."
fi

date > "$INSTALL_FLAG"

# =====================================================================
#  STEP 9 — Create the launcher script (start.sh)
# =====================================================================
log "Creating launcher script..."
cat > "$WORK_DIR/start.sh" << 'EOF'
#!/data/data/com.termux/files/usr/bin/bash
# Manga Translation API launcher
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; BOLD='\033[1m'; NC='\033[0m'

WORK_DIR="$HOME/manga-api"
VENV_DIR="$WORK_DIR/venv"
VERSION_FILE="$WORK_DIR/.version"
REPO_API="https://api.github.com/repos/Kirogii/MangaAMTL/releases/latest"
USER_AGENT="termux-manga-installer"

cd "$WORK_DIR"
source "$VENV_DIR/bin/activate"

# Auto-install missing deps on the fly
if ! python -c "import fastapi, uvicorn, torch, cv2" 2>/dev/null; then
    echo -e "${YELLOW}[!] Dependencies missing — running installer...${NC}"
    bash "$HOME/install_manga.sh"
    source "$VENV_DIR/bin/activate"
fi

# --- Check for updates ---
LOCAL_VERSION="unknown"
if [ -f "$VERSION_FILE" ]; then
    LOCAL_VERSION=$(cat "$VERSION_FILE")
fi

REMOTE_TAG=$(wget -q -T 5 -U "$USER_AGENT" -O - "$REPO_API" | python -c "import json,sys; print(json.load(sys.stdin).get('tag_name','unknown'))" 2>/dev/null)

if [ "$REMOTE_TAG" != "unknown" ] && [ "$REMOTE_TAG" != "$LOCAL_VERSION" ]; then
    echo -e "${YELLOW}════════════════════════════════════════════════════${NC}"
    echo -e "${YELLOW}[!] UPDATE AVAILABLE!${NC} (Local: $LOCAL_VERSION | Remote: $REMOTE_TAG)"
    echo -e "${YELLOW}    Type 'update' and restart 'Manga' to get the latest version.${NC}"
    echo -e "${YELLOW}════════════════════════════════════════════════════${NC}"
    echo ""
else
    echo -e "${GREEN}[✓] You are running the latest version: $LOCAL_VERSION${NC}"
fi

echo -e "${CYAN}══════════════════════════════════════════${NC}"
echo -e "${BOLD}  Manga Translation API${NC}"
echo -e "${CYAN}  Listening on http://0.0.0.0:8000${NC}"
echo -e "${CYAN}  Press Ctrl+C to stop.${NC}"
echo -e "${CYAN}══════════════════════════════════════════${NC}"
echo ""

exec python "$WORK_DIR/app.py"
EOF
chmod +x "$WORK_DIR/start.sh"

# =====================================================================
#  STEP 10 — Create 'Manga' and 'update' commands in $PREFIX/bin
# =====================================================================
log "Creating 'Manga' command..."
MANGA_CMD="$PREFIX/bin/Manga"
cat > "$MANGA_CMD" << 'EOF'
#!/data/data/com.termux/files/usr/bin/bash
WORK_DIR="$HOME/manga-api"
VENV_DIR="$WORK_DIR/venv"
INSTALL_FLAG="$WORK_DIR/.installed_v1"

if [ ! -f "$WORK_DIR/app.py" ]; then
    echo -e "\033[0;31m[✗] app.py not found in $WORK_DIR\033[0m"
    echo "Run the installer again."
    exit 1
fi
if [ ! -d "$VENV_DIR" ] || [ ! -f "$INSTALL_FLAG" ]; then
    echo -e "\033[1;33m[!] First run or missing install — running installer...\033[0m"
    bash "$HOME/install_manga.sh"
fi
exec bash "$WORK_DIR/start.sh" "$@"
EOF
chmod +x "$MANGA_CMD"

log "Creating 'update' command..."
UPDATE_CMD="$PREFIX/bin/update"
cat > "$UPDATE_CMD" << 'EOF'
#!/data/data/com.termux/files/usr/bin/bash
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; NC='\033[0m'

WORK_DIR="$HOME/manga-api"
VENV_DIR="$WORK_DIR/venv"
REPO_API="https://api.github.com/repos/Kirogii/MangaAMTL/releases/latest"
USER_AGENT="termux-manga-installer"
TEMP_DIR=$(mktemp -d)

echo -e "${CYAN}[*]${NC} Checking for updates..."
wget -q -U "$USER_AGENT" "$REPO_API" -O "$TEMP_DIR/release.json"

ASSET_URL=$(python -c "import json; r=json.load(open('$TEMP_DIR/release.json')); print(r['assets'][0]['browser_download_url'] if r.get('assets') else '')" 2>/dev/null)
NEW_TAG=$(python -c "import json; r=json.load(open('$TEMP_DIR/release.json')); print(r.get('tag_name', 'unknown'))" 2>/dev/null)

if [ -z "$ASSET_URL" ]; then
    echo -e "${RED}[✗]${NC} No release attachment found on GitHub to update from."
    rm -rf "$TEMP_DIR"
    exit 1
fi

if [ -f "$WORK_DIR/.version" ] && [ "$(cat "$WORK_DIR/.version")" == "$NEW_TAG" ]; then
    echo -e "${GREEN}[✓]${NC} Already running the latest version ($NEW_TAG)."
    rm -rf "$TEMP_DIR"
    exit 0
fi

echo -e "${CYAN}[*]${NC} Downloading latest release $NEW_TAG..."
wget -q -U "$USER_AGENT" "$ASSET_URL" -O "$TEMP_DIR/manga_update.zip"

echo -e "${CYAN}[*]${NC} Extracting package..."
unzip -o "$TEMP_DIR/manga_update.zip" -d "$WORK_DIR" >/dev/null 2>&1

# Handle nested folders
if [ ! -f "$WORK_DIR/app.py" ]; then
    APP_LOC=$(find "$WORK_DIR" -name "app.py" | head -n 1)
    if [ -n "$APP_LOC" ]; then
        APP_DIR=$(dirname "$APP_LOC")
        if [ "$APP_DIR" != "$WORK_DIR" ]; then
            cp -r "$APP_DIR"/* "$WORK_DIR/"
            rm -rf "$APP_DIR"
        fi
    fi
fi

echo "$NEW_TAG" > "$WORK_DIR/.version"
mkdir -p "$WORK_DIR/models" "$WORK_DIR/models/gguf" "$WORK_DIR/models/colorizer" "$WORK_DIR/fonts"

# Update Python dependencies if requirements.txt changed
if [ -f "$WORK_DIR/requirements.txt" ]; then
    echo -e "${CYAN}[*]${NC} Updating Python dependencies..."
    source "$VENV_DIR/bin/activate"
    pip install -r "$WORK_DIR/requirements.txt" 2>&1 | tail -5
fi

rm -rf "$TEMP_DIR"
echo -e "${GREEN}[✓]${NC} Successfully updated to $NEW_TAG."
echo -e "${YELLOW}[*]${NC} To apply changes, restart the API by typing ${BOLD}Manga${NC}."
EOF
chmod +x "$UPDATE_CMD"

# Add aliases to .bashrc as fallback
if ! grep -q "alias Manga=" "$HOME/.bashrc" 2>/dev/null; then
    echo "alias Manga='$MANGA_CMD'" >> "$HOME/.bashrc"
fi
if ! grep -q "alias update=" "$HOME/.bashrc" 2>/dev/null; then
    echo "alias update='$UPDATE_CMD'" >> "$HOME/.bashrc"
fi

# Save a copy of this installer to $HOME so auto-reinstall on launch works
if [ -f "$0" ] && [ "$(cd "$(dirname "$0")" && pwd)" != "$HOME" ]; then
    cp "$0" "$HOME/install_manga.sh" 2>/dev/null || true
fi

# =====================================================================
#  DONE
# =====================================================================
echo ""
echo -e "${BOLD}${GREEN}════════════════════════════════════════════${NC}"
echo -e "${BOLD}${GREEN}  ✓ Installation Complete!${NC}"
echo -e "${BOLD}${GREEN}════════════════════════════════════════════${NC}"
echo ""
echo -e "  Working dir : ${CYAN}$WORK_DIR${NC}"
echo -e "  Version     : ${CYAN}$(cat $VERSION_FILE)${NC}"
echo ""
echo -e "${YELLOW}Usage:${NC}"
echo -e "  ${BOLD}Manga${NC}    — start the API server (checks for updates on startup)"
echo -e "  ${BOLD}update${NC}   — download the latest release zip and refresh dependencies"
echo ""
echo -e "Run ${BOLD}source ~/.bashrc${NC} if commands aren't immediately found."
echo ""
