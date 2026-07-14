#!/usr/bin/env bash
# ============================================================================
#  MangaAMTL - Plain Ubuntu VPS Installer (no local-LLM build)
#  Repo: https://github.com/Kirogii/MangaAMTL
#
#  Same as the Termux/proot-distro installer, but for a regular Ubuntu VPS,
#  and it SKIPS llama-cpp-python entirely (that's the package that takes
#  forever to build from source when there's no prebuilt wheel for your
#  platform). Everything else you need for cloud-backed translation + OCR +
#  inpainting (fastapi/uvicorn, torch/torchvision, ultralytics,
#  simple-lama-inpainting-updated, onnxruntime, transformers, hayai-ocr,
#  chrome-lens-py, etc.) still gets installed normally, since those all have
#  prebuilt manylinux wheels on x86_64 and don't need local compilation.
#
#  Run:
#      bash install_manga_vps.sh            (normal / CPU requirements.txt)
#      bash install_manga_vps.sh --cuda     (installs cudarequirements.txt)
#
#  Want to skip more packages? Add them (pip name) to EXCLUDE_PKGS below,
#  space-separated.
# ============================================================================

set -u

REPO="Kirogii/MangaAMTL"
API_URL="https://api.github.com/repos/${REPO}/releases/latest"
INSTALL_DIR="${HOME}/MangaAMTL"
VERSION_FILE="${INSTALL_DIR}/.manga_version"
REQ_MODE_FILE="${INSTALL_DIR}/.manga_reqmode"
VENV_DIR="${INSTALL_DIR}/.venv"

# Packages to strip out of whatever requirements file the release ships.
# llama-cpp-python is the one that has no wheel on most VPS platforms and
# ends up building from source (slow). Add more names here if needed.
EXCLUDE_PKGS="llama-cpp-python"

RED="\033[1;31m"; GREEN="\033[1;32m"; YELLOW="\033[1;33m"; CYAN="\033[1;36m"; NC="\033[0m"
info()  { echo -e "${CYAN}[*]${NC} $1"; }
ok()    { echo -e "${GREEN}[+]${NC} $1"; }
warn()  { echo -e "${YELLOW}[!]${NC} $1"; }
err()   { echo -e "${RED}[x]${NC} $1"; }

# ----------------------------------------------------------------------------
# 0. Root / sudo detection + BIN_DIR
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

# Build tools are still useful for the odds and ends that don't ship wheels
# (e.g. sentencepiece/fugashi sdists on some platforms), but nothing here
# should need to compile anything as heavy as llama.cpp.
BUILD_PKGS="build-essential cmake ninja-build pkg-config rustc cargo binutils patchelf libjpeg-turbo8-dev libpng-dev libfreetype6-dev libopenblas-dev"
RUNTIME_PKGS="libgl1 libglib2.0-0"

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

info "Installing runtime libraries (${RUNTIME_PKGS})..."
# shellcheck disable=SC2086
$SUDO apt-get install -y $RUNTIME_PKGS

# ----------------------------------------------------------------------------
# 1b. MangaAMTL requires Python 3.12.
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
    cp -rf "$extracted_dir"/. "$INSTALL_DIR"/
    rm -rf "$tmp_dir"
    echo "$tag" > "$VERSION_FILE"
    echo "$REQ_MODE" > "$REQ_MODE_FILE"
    return 0
}

# ----------------------------------------------------------------------------
# 3b. Strip EXCLUDE_PKGS (e.g. llama-cpp-python) out of a requirements file,
#     so pip never even tries to build it. Writes a filtered copy and echoes
#     its path.
# ----------------------------------------------------------------------------
filter_requirements() {
    local src_file="$1"
    local out_file="${src_file}.filtered"
    cp "$src_file" "$out_file"
    for pipname in $EXCLUDE_PKGS; do
        grep -viE "^${pipname}([<>=! ].*)?$" "$out_file" > "${out_file}.tmp" && mv "${out_file}.tmp" "$out_file"
        info "  -> excluded ${pipname} from install list"
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
# 5. Python venv + requirements (llama-cpp-python excluded)
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
    FILTERED_REQ="$(filter_requirements "${INSTALL_DIR}/${REQ_MODE}")"
    info "Installing requirements from ${REQ_MODE} (llama-cpp-python skipped)..."
    $PIP install -r "$FILTERED_REQ"
else
    warn "${REQ_MODE} not found in the downloaded release, skipping pip install."
fi

if [ -d "$VENV_DIR" ]; then
    deactivate 2>/dev/null || true
fi

warn "llama-cpp-python was skipped. If the app hard-requires it to even start"
warn "(rather than just for local-model translation), you'll need it installed"
warn "manually, or you'll need to point the app at cloud translation only."

# ----------------------------------------------------------------------------
# 6. Install the "Manga" launcher command
# ----------------------------------------------------------------------------
info "Installing 'Manga' launcher command to ${BIN_DIR}..."
mkdir -p "$BIN_DIR"

cat > "$LAUNCHER" << LAUNCHER_EOF
#!/usr/bin/env bash
# Auto-generated launcher for MangaAMTL. Re-run install_manga_vps.sh to regenerate.

set -u

REPO="Kirogii/MangaAMTL"
API_URL="https://api.github.com/repos/\${REPO}/releases/latest"
INSTALL_DIR="\${HOME}/MangaAMTL"
VERSION_FILE="\${INSTALL_DIR}/.manga_version"
REQ_MODE_FILE="\${INSTALL_DIR}/.manga_reqmode"
VENV_DIR="\${INSTALL_DIR}/.venv"

EXCLUDE_PKGS="${EXCLUDE_PKGS}"
BUILD_PKGS="${BUILD_PKGS}"
RUNTIME_PKGS="${RUNTIME_PKGS}"

if [ "\$(id -u)" = "0" ]; then
    SUDO=""
elif command -v sudo >/dev/null 2>&1; then
    SUDO="sudo"
else
    SUDO=""
fi
export DEBIAN_FRONTEND=noninteractive

RED="\033[1;31m"; GREEN="\033[1;32m"; YELLOW="\033[1;33m"; CYAN="\033[1;36m"; NC="\033[0m"
info()  { echo -e "\${CYAN}[*]\${NC} \$1"; }
ok()    { echo -e "\${GREEN}[+]\${NC} \$1"; }
warn()  { echo -e "\${YELLOW}[!]\${NC} \$1"; }
err()   { echo -e "\${RED}[x]\${NC} \$1"; }

PYTHON_BIN=""
ensure_python312() {
    if command -v python3.12 >/dev/null 2>&1; then
        PYTHON_BIN="python3.12"
        return 0
    fi
    \$SUDO apt-get install -y python3.12 python3.12-venv python3.12-dev >/dev/null 2>&1
    if command -v python3.12 >/dev/null 2>&1; then
        PYTHON_BIN="python3.12"
        return 0
    fi
    warn "python3.12 not found; adding deadsnakes PPA..."
    \$SUDO apt-get install -y software-properties-common gnupg2 ca-certificates >/dev/null 2>&1
    \$SUDO add-apt-repository -y ppa:deadsnakes/ppa >/dev/null 2>&1
    \$SUDO apt-get update -y >/dev/null 2>&1
    \$SUDO apt-get install -y python3.12 python3.12-venv python3.12-dev >/dev/null 2>&1
    if command -v python3.12 >/dev/null 2>&1; then
        PYTHON_BIN="python3.12"
        return 0
    fi
    err "Could not obtain python3.12."
    return 1
}
ensure_python312 || exit 1

ensure_build_tools() {
    info "Making sure build tools are present..."
    # shellcheck disable=SC2086
    \$SUDO apt-get install -y \$BUILD_PKGS >/dev/null 2>&1
    # shellcheck disable=SC2086
    \$SUDO apt-get install -y \$RUNTIME_PKGS >/dev/null 2>&1
}

filter_requirements() {
    local src_file="\$1"
    local out_file="\${src_file}.filtered"
    cp "\$src_file" "\$out_file"
    for pipname in \$EXCLUDE_PKGS; do
        grep -viE "^\${pipname}([<>=! ].*)?\$" "\$out_file" > "\${out_file}.tmp" && mv "\${out_file}.tmp" "\$out_file"
    done
    echo "\$out_file"
}

get_latest_tag() {
    wget -qO- --timeout=8 "\$API_URL" 2>/dev/null | \\
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
    [ -f "\$VERSION_FILE" ] && cat "\$VERSION_FILE" || echo ""
}

do_update() {
    local tag="\$1"
    local tmp_dir
    tmp_dir="\$(mktemp -d)"
    local zip_path="\${tmp_dir}/manga.zip"
    local zip_url="https://github.com/\${REPO}/archive/refs/tags/\${tag}.zip"
    local req_mode
    req_mode="\$( [ -f "\$REQ_MODE_FILE" ] && cat "\$REQ_MODE_FILE" || echo "requirements.txt" )"

    info "Downloading MangaAMTL \${tag}..."
    if ! wget -q --timeout=120 -O "\$zip_path" "\$zip_url"; then
        err "Download failed. Update aborted."
        rm -rf "\$tmp_dir"
        return 1
    fi

    info "Extracting update..."
    if ! unzip -q -o "\$zip_path" -d "\$tmp_dir"; then
        err "Extraction failed. Update aborted."
        rm -rf "\$tmp_dir"
        return 1
    fi

    local extracted_dir
    extracted_dir="\$(find "\$tmp_dir" -maxdepth 1 -type d -name "MangaAMTL-*" | head -n1)"
    if [ -z "\$extracted_dir" ]; then
        err "Could not locate extracted folder. Update aborted."
        rm -rf "\$tmp_dir"
        return 1
    fi

    cp -rf "\$extracted_dir"/. "\$INSTALL_DIR"/
    rm -rf "\$tmp_dir"
    echo "\$tag" > "\$VERSION_FILE"

    if [ -d "\$VENV_DIR" ]; then
        # shellcheck disable=SC1091
        source "\${VENV_DIR}/bin/activate"
    fi

    if [ -f "\${INSTALL_DIR}/\${req_mode}" ]; then
        ensure_build_tools
        FILTERED_REQ="\$(filter_requirements "\${INSTALL_DIR}/\${req_mode}")"
        info "Reinstalling requirements (\${req_mode}, llama-cpp-python skipped)..."
        pip install --upgrade pip
        pip install -r "\$FILTERED_REQ"
    fi

    if [ -d "\$VENV_DIR" ]; then
        deactivate 2>/dev/null || true
    fi

    ok "Updated to \${tag}."
}

cd "\$INSTALL_DIR" || { err "MangaAMTL install directory not found. Re-run install_manga_vps.sh."; exit 1; }

if [ "\${1:-}" = "update" ]; then
    LATEST_TAG="\$(get_latest_tag)"
    if [ -z "\$LATEST_TAG" ]; then
        err "Could not reach GitHub. Check your connection."
        exit 1
    fi
    do_update "\$LATEST_TAG"
    exit 0
fi

LOCAL_TAG="\$(get_local_tag)"
LATEST_TAG="\$(get_latest_tag)"

if [ -n "\$LATEST_TAG" ] && [ "\$LATEST_TAG" != "\$LOCAL_TAG" ]; then
    warn "New version available: \${LOCAL_TAG:-unknown} -> \${LATEST_TAG}"
    read -r -p "Type 'update' to update now, or press Enter to launch anyway: " ANSWER
    if [ "\$ANSWER" = "update" ]; then
        do_update "\$LATEST_TAG"
    fi
elif [ -z "\$LATEST_TAG" ]; then
    warn "Could not check for updates (offline?). Launching current version (\${LOCAL_TAG:-unknown})."
else
    ok "MangaAMTL is up to date (\${LOCAL_TAG})."
fi

if [ -d "\$VENV_DIR" ]; then
    # shellcheck disable=SC1091
    source "\${VENV_DIR}/bin/activate"
fi

python3 app.py

if [ -d "\$VENV_DIR" ]; then
    deactivate 2>/dev/null || true
fi
LAUNCHER_EOF

chmod +x "$LAUNCHER"

ok "Installed launcher: ${LAUNCHER}"

echo ""
ok "Installation complete! Installed version: ${LATEST_TAG}"
echo -e "${CYAN}--------------------------------------------------------${NC}"
echo -e "  Type ${GREEN}Manga${NC}         -> launch MangaAMTL (auto-checks for updates)"
echo -e "  Type ${GREEN}Manga update${NC}  -> force an update right now (llama-cpp-python still skipped)"
echo -e "${CYAN}--------------------------------------------------------${NC}"
if [ "$BIN_DIR" != "/usr/local/bin" ]; then
    warn "Remember to add ${BIN_DIR} to your PATH if you haven't already."
fi
