#!/usr/bin/env bash
# ============================================================================
#  MangaAMTL - Termux Auto-Installer
#  Repo: https://github.com/Kirogii/MangaAMTL
#
#  This script:
#    - Installs required Termux packages (python, git, wget, unzip)
#    - Downloads the latest release of MangaAMTL from GitHub
#    - Creates a Python venv and installs requirements.txt automatically
#    - Installs a global "Manga" command that:
#         * checks GitHub for a newer release every time you launch it
#         * lets you type "update" to pull + reinstall the newest version
#         * otherwise just starts app.py
#
#  Usage:
#    bash install_manga.sh            (normal / CPU requirements.txt)
#    bash install_manga.sh --cuda     (installs cudarequirements.txt instead)
# ============================================================================

set -u

REPO="Kirogii/MangaAMTL"
API_URL="https://api.github.com/repos/${REPO}/releases/latest"
INSTALL_DIR="${HOME}/MangaAMTL"
VERSION_FILE="${INSTALL_DIR}/.manga_version"
REQ_MODE_FILE="${INSTALL_DIR}/.manga_reqmode"
VENV_DIR="${INSTALL_DIR}/.venv"
BIN_DIR="${PREFIX}/bin"
LAUNCHER="${BIN_DIR}/Manga"

RED="\033[1;31m"; GREEN="\033[1;32m"; YELLOW="\033[1;33m"; CYAN="\033[1;36m"; NC="\033[0m"
info()  { echo -e "${CYAN}[*]${NC} $1"; }
ok()    { echo -e "${GREEN}[+]${NC} $1"; }
warn()  { echo -e "${YELLOW}[!]${NC} $1"; }
err()   { echo -e "${RED}[x]${NC} $1"; }

REQ_MODE="requirements.txt"
if [ "${1:-}" = "--cuda" ]; then
    REQ_MODE="cudarequirements.txt"
fi

# ----------------------------------------------------------------------------
# 1. Install Termux packages
# ----------------------------------------------------------------------------
info "Updating Termux packages..."
pkg update -y && pkg upgrade -y

info "Installing dependencies (python git wget unzip)..."
pkg install -y python git wget unzip

# ----------------------------------------------------------------------------
# 2. Helper: get latest release tag from GitHub API
# ----------------------------------------------------------------------------
get_latest_tag() {
    wget -qO- --timeout=10 "$API_URL" 2>/dev/null | \
    python3 -c "
import json,sys
try:
    data = json.load(sys.stdin)
    print(data.get('tag_name',''))
except Exception:
    print('')
" 2>/dev/null
}

# ----------------------------------------------------------------------------
# 3. Download + extract a given release tag into INSTALL_DIR
# ----------------------------------------------------------------------------
download_and_install() {
    local tag="$1"
    local tmp_dir
    tmp_dir="$(mktemp -d)"
    local zip_path="${tmp_dir}/manga.zip"
    local zip_url="https://github.com/${REPO}/archive/refs/tags/${tag}.zip"

    info "Downloading MangaAMTL ${tag}..."
    if ! wget -q --timeout=120 -O "$zip_path" "$zip_url"; then
        err "Download failed. Check your internet connection or try again later."
        rm -rf "$tmp_dir"
        return 1
    fi

    info "Extracting..."
    if ! unzip -q -o "$zip_path" -d "$tmp_dir"; then
        err "Extraction failed (corrupt zip?)."
        rm -rf "$tmp_dir"
        return 1
    fi

    local extracted_dir
    extracted_dir="$(find "$tmp_dir" -maxdepth 1 -type d -name "MangaAMTL-*" | head -n1)"
    if [ -z "$extracted_dir" ]; then
        err "Could not locate extracted MangaAMTL folder."
        rm -rf "$tmp_dir"
        return 1
    fi

    mkdir -p "$INSTALL_DIR"
    # Copy new files over, but never touch the venv or version files
    cp -rf "$extracted_dir"/. "$INSTALL_DIR"/
    rm -rf "$tmp_dir"
    echo "$tag" > "$VERSION_FILE"
    echo "$REQ_MODE" > "$REQ_MODE_FILE"
    return 0
}

# ----------------------------------------------------------------------------
# 4. Fresh install
# ----------------------------------------------------------------------------
info "Checking latest MangaAMTL release..."
LATEST_TAG="$(get_latest_tag)"

if [ -z "$LATEST_TAG" ]; then
    err "Could not reach GitHub to find the latest release. Aborting."
    exit 1
fi

ok "Latest release: ${LATEST_TAG}"

if ! download_and_install "$LATEST_TAG"; then
    err "Install failed."
    exit 1
fi

# ----------------------------------------------------------------------------
# 5. Python venv + requirements
# ----------------------------------------------------------------------------
info "Setting up Python virtual environment..."
python3 -m venv "$VENV_DIR" 2>/dev/null || warn "venv module unavailable, will install packages globally instead."

if [ -d "$VENV_DIR" ]; then
    # shellcheck disable=SC1091
    source "${VENV_DIR}/bin/activate"
    PIP="pip"
else
    PIP="pip"
fi

info "Upgrading pip..."
 $PIP install --upgrade pip

if [ -f "${INSTALL_DIR}/${REQ_MODE}" ]; then
    info "Installing requirements from ${REQ_MODE} (this can take a while)..."
    $PIP install -r "${INSTALL_DIR}/${REQ_MODE}"
else
    warn "${REQ_MODE} not found in the downloaded release, skipping pip install."
fi

if [ -d "$VENV_DIR" ]; then
    deactivate 2>/dev/null || true
fi

# ----------------------------------------------------------------------------
# 6. Install the "Manga" launcher command
# ----------------------------------------------------------------------------
info "Installing 'Manga' launcher command to ${BIN_DIR}..."
mkdir -p "$BIN_DIR"

cat > "$LAUNCHER" << 'LAUNCHER_EOF'
#!/usr/bin/env bash
# Auto-generated launcher for MangaAMTL. Re-run install_manga.sh to regenerate.

set -u

REPO="Kirogii/MangaAMTL"
API_URL="https://api.github.com/repos/${REPO}/releases/latest"
INSTALL_DIR="${HOME}/MangaAMTL"
VERSION_FILE="${INSTALL_DIR}/.manga_version"
REQ_MODE_FILE="${INSTALL_DIR}/.manga_reqmode"
VENV_DIR="${INSTALL_DIR}/.venv"

RED="\033[1;31m"; GREEN="\033[1;32m"; YELLOW="\033[1;33m"; CYAN="\033[1;36m"; NC="\033[0m"
info()  { echo -e "${CYAN}[*]${NC} $1"; }
ok()    { echo -e "${GREEN}[+]${NC} $1"; }
warn()  { echo -e "${YELLOW}[!]${NC} $1"; }
err()   { echo -e "${RED}[x]${NC} $1"; }

get_latest_tag() {
    wget -qO- --timeout=8 "$API_URL" 2>/dev/null | \
    python3 -c "
import json,sys
try:
    data = json.load(sys.stdin)
    print(data.get('tag_name',''))
except Exception:
    print('')
" 2>/dev/null
}

get_local_tag() {
    [ -f "$VERSION_FILE" ] && cat "$VERSION_FILE" || echo ""
}

do_update() {
    local tag="$1"
    local tmp_dir
    tmp_dir="$(mktemp -d)"
    local zip_path="${tmp_dir}/manga.zip"
    local zip_url="https://github.com/${REPO}/archive/refs/tags/${tag}.zip"
    local req_mode
    req_mode="$( [ -f "$REQ_MODE_FILE" ] && cat "$REQ_MODE_FILE" || echo "requirements.txt" )"

    info "Downloading MangaAMTL ${tag}..."
    if ! wget -q --timeout=120 -O "$zip_path" "$zip_url"; then
        err "Download failed. Update aborted."
        rm -rf "$tmp_dir"
        return 1
    fi

    info "Extracting update..."
    if ! unzip -q -o "$zip_path" -d "$tmp_dir"; then
        err "Extraction failed. Update aborted."
        rm -rf "$tmp_dir"
        return 1
    fi

    local extracted_dir
    extracted_dir="$(find "$tmp_dir" -maxdepth 1 -type d -name "MangaAMTL-*" | head -n1)"
    if [ -z "$extracted_dir" ]; then
        err "Could not locate extracted folder. Update aborted."
        rm -rf "$tmp_dir"
        return 1
    fi

    cp -rf "$extracted_dir"/. "$INSTALL_DIR"/
    rm -rf "$tmp_dir"
    echo "$tag" > "$VERSION_FILE"

    if [ -d "$VENV_DIR" ]; then
        # shellcheck disable=SC1091
        source "${VENV_DIR}/bin/activate"
    fi

    if [ -f "${INSTALL_DIR}/${req_mode}" ]; then
        info "Reinstalling requirements (${req_mode})..."
        pip install --upgrade pip
        pip install -r "${INSTALL_DIR}/${req_mode}"
    fi

    if [ -d "$VENV_DIR" ]; then
        deactivate 2>/dev/null || true
    fi

    ok "Updated to ${tag}."
}

cd "$INSTALL_DIR" || { err "MangaAMTL install directory not found. Re-run install_manga.sh."; exit 1; }

# --- explicit "Manga update" from shell ---
if [ "${1:-}" = "update" ]; then
    LATEST_TAG="$(get_latest_tag)"
    if [ -z "$LATEST_TAG" ]; then
        err "Could not reach GitHub. Check your connection."
        exit 1
    fi
    do_update "$LATEST_TAG"
    exit 0
fi

# --- startup version check ---
LOCAL_TAG="$(get_local_tag)"
LATEST_TAG="$(get_latest_tag)"

if [ -n "$LATEST_TAG" ] && [ "$LATEST_TAG" != "$LOCAL_TAG" ]; then
    warn "New version available: ${LOCAL_TAG:-unknown} -> ${LATEST_TAG}"
    read -r -p "Type 'update' to update now, or press Enter to launch anyway: " ANSWER
    if [ "$ANSWER" = "update" ]; then
        do_update "$LATEST_TAG"
    fi
elif [ -z "$LATEST_TAG" ]; then
    warn "Could not check for updates (offline?). Launching current version (${LOCAL_TAG:-unknown})."
else
    ok "MangaAMTL is up to date (${LOCAL_TAG})."
fi

# --- launch ---
if [ -d "$VENV_DIR" ]; then
    # shellcheck disable=SC1091
    source "${VENV_DIR}/bin/activate"
fi

python app.py

if [ -d "$VENV_DIR" ]; then
    deactivate 2>/dev/null || true
fi
LAUNCHER_EOF

chmod +x "$LAUNCHER"

# Also drop an "update" alias inside the install dir folder for convenience,
# without shadowing a system-wide "update" command.
ok "Installed launcher: ${LAUNCHER}"

echo ""
ok "Installation complete! Installed version: ${LATEST_TAG}"
echo -e "${CYAN}--------------------------------------------------------${NC}"
echo -e "  Type ${GREEN}Manga${NC}         -> launch MangaAMTL (auto-checks for updates)"
echo -e "  Type ${GREEN}Manga update${NC}  -> force an update right now"
echo -e "${CYAN}--------------------------------------------------------${NC}"
