#!/usr/bin/env bash
# ============================================================================
#  MangaAMTL - proot-distro Ubuntu (Termux) Auto-Installer
#  Repo: https://github.com/Kirogii/MangaAMTL
#
#  Run this INSIDE your Ubuntu proot-distro shell, not in bare Termux:
#      termux>            proot-distro login ubuntu
#      ubuntu (root)#      bash install_manga_ubuntu.sh
#
#  This script:
#    - Installs required apt packages (python3, git, wget, unzip, build tools)
#    - Downloads the latest release of MangaAMTL from GitHub
#    - Creates a Python venv and installs requirements.txt automatically
#    - Installs a global "Manga" command that:
#         * checks GitHub for a newer release every time you launch it
#         * lets you type "update" to pull + reinstall the newest version
#         * otherwise just starts app.py
#
#  Usage:
#    bash install_manga_ubuntu.sh            (normal / CPU requirements.txt)
#    bash install_manga_ubuntu.sh --cuda     (installs cudarequirements.txt instead)
# ============================================================================

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

# ----------------------------------------------------------------------------
# 0. Root / sudo detection + BIN_DIR (no $PREFIX under plain Linux/proot)
# ----------------------------------------------------------------------------
if [ "$(id -u)" = "0" ]; then
    SUDO=""
elif command -v sudo >/dev/null 2>&1; then
    SUDO="sudo"
else
    warn "Not running as root and 'sudo' is unavailable. apt/package steps may fail."
    SUDO=""
fi

if [ -w "/usr/local/bin" ] || [ "$(id -u)" = "0" ]; then
    BIN_DIR="/usr/local/bin"
else
    BIN_DIR="${HOME}/.local/bin"
    mkdir -p "$BIN_DIR"
    warn "Not root: installing launcher to ${BIN_DIR}. Make sure it's on your PATH:"
    warn "  echo 'export PATH=\"\$HOME/.local/bin:\$PATH\"' >> ~/.bashrc"
fi
LAUNCHER="${BIN_DIR}/Manga"

export DEBIAN_FRONTEND=noninteractive

REQ_MODE="requirements.txt"
if [ "${1:-}" = "--cuda" ]; then
    REQ_MODE="cudarequirements.txt"
fi

# Packages needed to *compile* things like llama-cpp-python, onnxruntime,
# tokenizers (rust), pillow/opencv (C/C++), sentencepiece, etc.
BUILD_PKGS="build-essential cmake ninja-build pkg-config rustc cargo binutils patchelf libjpeg-turbo8-dev libpng-dev libfreetype6-dev libopenblas-dev"

# Ubuntu already ships prebuilt apt packages for some of the heavier pip
# requirements. Installing these via apt is *much* faster/more reliable than
# letting pip try to compile them from source on-device.
# Format: "pip_name:apt_pkg_name:python_import_name"
TERMUX_ALTS="numpy:python3-numpy:numpy pillow:python3-pil:PIL opencv-python:python3-opencv:cv2 cryptography:python3-cryptography:cryptography"

# ----------------------------------------------------------------------------
# 1. Install apt packages
# ----------------------------------------------------------------------------
info "Updating apt package lists..."
$SUDO apt-get update -y && $SUDO apt-get upgrade -y

info "Installing base dependencies (python3 git wget unzip curl ca-certificates)..."
$SUDO apt-get install -y python3 python3-venv python3-dev python3-pip git wget unzip curl ca-certificates

info "Installing build tools (${BUILD_PKGS})..."
# shellcheck disable=SC2086
$SUDO apt-get install -y $BUILD_PKGS

# ----------------------------------------------------------------------------
# 1b. MangaAMTL requires Python 3.12. Not every Ubuntu release used by
#     proot-distro ships this as the default `python3` (only 24.04 "noble"
#     does; 22.04 "jammy" ships 3.10). Get a real python3.12 binary either way.
# ----------------------------------------------------------------------------
PYTHON_BIN=""
ensure_python312() {
    if command -v python3.12 >/dev/null 2>&1; then
        PYTHON_BIN="python3.12"
        return 0
    fi

    info "python3.12 not found, trying to install it from the distro repos..."
    $SUDO apt-get install -y python3.12 python3.12-venv python3.12-dev >/dev/null 2>&1
    if command -v python3.12 >/dev/null 2>&1; then
        PYTHON_BIN="python3.12"
        return 0
    fi

    warn "python3.12 isn't in this distro's default repos, adding the deadsnakes PPA..."
    $SUDO apt-get install -y software-properties-common gnupg2 ca-certificates
    $SUDO add-apt-repository -y ppa:deadsnakes/ppa
    $SUDO apt-get update -y
    $SUDO apt-get install -y python3.12 python3.12-venv python3.12-dev

    if command -v python3.12 >/dev/null 2>&1; then
        PYTHON_BIN="python3.12"
        return 0
    fi

    err "Could not obtain a python3.12 interpreter. Aborting."
    exit 1
}
ensure_python312
ok "Using ${PYTHON_BIN} ($(${PYTHON_BIN} --version 2>&1))."

# Only apt's python3-numpy/python3-pil/etc are actually usable inside a
# python3.12 venv if the distro's *default* python3 is ALSO 3.12 (true on
# Ubuntu 24.04, false on 22.04 where those apt packages are built for 3.10
# and would silently fail to import from a 3.12 venv). Otherwise skip apt
# natives and let pip pull manylinux/aarch64 wheels directly -- unlike
# Termux, glibc-based Ubuntu can normally use official PyPI wheels for
# numpy/pillow/opencv/cryptography without local compilation anyway.
SYSTEM_PY_VER="$(python3 -c 'import sys; print("%d.%d" % sys.version_info[:2])' 2>/dev/null || echo "")"
if [ "$SYSTEM_PY_VER" = "3.12" ]; then
    SYSTEM_PY_IS_312=1
else
    SYSTEM_PY_IS_312=0
    warn "System default python3 is ${SYSTEM_PY_VER:-unknown}, not 3.12 -- skipping apt-native numpy/pillow/opencv/cryptography shortcuts (pip wheels will be used instead)."
fi

if [ "$SYSTEM_PY_IS_312" = "1" ]; then
    info "Installing apt-native alternatives for heavy pip packages..."
    for triple in $TERMUX_ALTS; do
        apkg="$(echo "$triple" | cut -d: -f2)"
        info "  -> ${apkg}"
        $SUDO apt-get install -y "$apkg" || warn "Could not install ${apkg} via apt, will fall back to pip for it."
    done
fi

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
# 3b. Strip packages we already installed via apt (TERMUX_ALTS) out of a
#     requirements file, so pip doesn't try to rebuild them from source.
#     Writes a filtered copy and echoes its path.
# ----------------------------------------------------------------------------
filter_requirements() {
    local src_file="$1"
    local out_file="${src_file}.filtered"
    cp "$src_file" "$out_file"
    for triple in $TERMUX_ALTS; do
        pipname="$(echo "$triple" | cut -d: -f1)"
        importname="$(echo "$triple" | cut -d: -f3)"
        # Only strip it if the apt install actually succeeded (module importable).
        if python3 -c "import ${importname}" >/dev/null 2>&1; then
            grep -viE "^${pipname}([<>=! ].*)?$" "$out_file" > "${out_file}.tmp" && mv "${out_file}.tmp" "$out_file"
        fi
    done
    echo "$out_file"
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
info "Setting up Python virtual environment with ${PYTHON_BIN}..."
"$PYTHON_BIN" -m venv "$VENV_DIR" 2>/dev/null || warn "venv module unavailable, will install packages globally instead."

if [ -d "$VENV_DIR" ]; then
    # shellcheck disable=SC1091
    source "${VENV_DIR}/bin/activate"
    PIP="pip"
else
    PIP="${PYTHON_BIN} -m pip"
fi

info "Upgrading pip..."
$PIP install --upgrade pip

if [ -f "${INSTALL_DIR}/${REQ_MODE}" ]; then
    if [ "$SYSTEM_PY_IS_312" = "1" ]; then
        FILTERED_REQ="$(filter_requirements "${INSTALL_DIR}/${REQ_MODE}")"
    else
        FILTERED_REQ="${INSTALL_DIR}/${REQ_MODE}"
    fi
    info "Installing requirements from ${REQ_MODE} (this can take a while, some packages will be compiled locally)..."
    $PIP install -r "$FILTERED_REQ"
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
# Auto-generated launcher for MangaAMTL. Re-run install_manga_ubuntu.sh to regenerate.

set -u

REPO="Kirogii/MangaAMTL"
API_URL="https://api.github.com/repos/${REPO}/releases/latest"
INSTALL_DIR="${HOME}/MangaAMTL"
VERSION_FILE="${INSTALL_DIR}/.manga_version"
REQ_MODE_FILE="${INSTALL_DIR}/.manga_reqmode"
VENV_DIR="${INSTALL_DIR}/.venv"

BUILD_PKGS="build-essential cmake ninja-build pkg-config rustc cargo binutils patchelf libjpeg-turbo8-dev libpng-dev libfreetype6-dev libopenblas-dev"
TERMUX_ALTS="numpy:python3-numpy:numpy pillow:python3-pil:PIL opencv-python:python3-opencv:cv2 cryptography:python3-cryptography:cryptography"

if [ "$(id -u)" = "0" ]; then
    SUDO=""
elif command -v sudo >/dev/null 2>&1; then
    SUDO="sudo"
else
    SUDO=""
fi
export DEBIAN_FRONTEND=noninteractive

RED="\033[1;31m"; GREEN="\033[1;32m"; YELLOW="\033[1;33m"; CYAN="\033[1;36m"; NC="\033[0m"
info()  { echo -e "${CYAN}[*]${NC} $1"; }
ok()    { echo -e "${GREEN}[+]${NC} $1"; }
warn()  { echo -e "${YELLOW}[!]${NC} $1"; }
err()   { echo -e "${RED}[x]${NC} $1"; }

PYTHON_BIN=""
ensure_python312() {
    if command -v python3.12 >/dev/null 2>&1; then
        PYTHON_BIN="python3.12"
        return 0
    fi
    $SUDO apt-get install -y python3.12 python3.12-venv python3.12-dev >/dev/null 2>&1
    if command -v python3.12 >/dev/null 2>&1; then
        PYTHON_BIN="python3.12"
        return 0
    fi
    warn "python3.12 not found; adding deadsnakes PPA..."
    $SUDO apt-get install -y software-properties-common gnupg2 ca-certificates >/dev/null 2>&1
    $SUDO add-apt-repository -y ppa:deadsnakes/ppa >/dev/null 2>&1
    $SUDO apt-get update -y >/dev/null 2>&1
    $SUDO apt-get install -y python3.12 python3.12-venv python3.12-dev >/dev/null 2>&1
    if command -v python3.12 >/dev/null 2>&1; then
        PYTHON_BIN="python3.12"
        return 0
    fi
    err "Could not obtain python3.12."
    return 1
}
ensure_python312 || exit 1

SYSTEM_PY_VER="$(python3 -c 'import sys; print("%d.%d" % sys.version_info[:2])' 2>/dev/null || echo "")"
if [ "$SYSTEM_PY_VER" = "3.12" ]; then
    SYSTEM_PY_IS_312=1
else
    SYSTEM_PY_IS_312=0
fi

ensure_build_tools() {
    info "Making sure build tools are present..."
    # shellcheck disable=SC2086
    $SUDO apt-get install -y $BUILD_PKGS >/dev/null 2>&1
    if [ "$SYSTEM_PY_IS_312" = "1" ]; then
        for triple in $TERMUX_ALTS; do
            apkg="$(echo "$triple" | cut -d: -f2)"
            $SUDO apt-get install -y "$apkg" >/dev/null 2>&1
        done
    fi
}

filter_requirements() {
    local src_file="$1"
    if [ "$SYSTEM_PY_IS_312" != "1" ]; then
        echo "$src_file"
        return 0
    fi
    local out_file="${src_file}.filtered"
    cp "$src_file" "$out_file"
    for triple in $TERMUX_ALTS; do
        pipname="$(echo "$triple" | cut -d: -f1)"
        importname="$(echo "$triple" | cut -d: -f3)"
        if python3 -c "import ${importname}" >/dev/null 2>&1; then
            grep -viE "^${pipname}([<>=! ].*)?$" "$out_file" > "${out_file}.tmp" && mv "${out_file}.tmp" "$out_file"
        fi
    done
    echo "$out_file"
}

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
        ensure_build_tools
        FILTERED_REQ="$(filter_requirements "${INSTALL_DIR}/${req_mode}")"
        info "Reinstalling requirements (${req_mode})..."
        pip install --upgrade pip
        pip install -r "$FILTERED_REQ"
    fi

    if [ -d "$VENV_DIR" ]; then
        deactivate 2>/dev/null || true
    fi

    ok "Updated to ${tag}."
}

cd "$INSTALL_DIR" || { err "MangaAMTL install directory not found. Re-run install_manga_ubuntu.sh."; exit 1; }

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

python3 app.py

if [ -d "$VENV_DIR" ]; then
    deactivate 2>/dev/null || true
fi
LAUNCHER_EOF

chmod +x "$LAUNCHER"

ok "Installed launcher: ${LAUNCHER}"

echo ""
ok "Installation complete! Installed version: ${LATEST_TAG}"
echo -e "${CYAN}--------------------------------------------------------${NC}"
echo -e "  Type ${GREEN}Manga${NC}         -> launch MangaAMTL (auto-checks for updates)"
echo -e "  Type ${GREEN}Manga update${NC}  -> force an update right now"
echo -e "${CYAN}--------------------------------------------------------${NC}"
if [ "$BIN_DIR" != "/usr/local/bin" ]; then
    warn "Remember to add ${BIN_DIR} to your PATH if you haven't already."
fi
