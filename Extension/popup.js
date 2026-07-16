// ============================================================================
// FONT PREVIEW HELPERS
// ============================================================================
const _mtFontByteCache = new Map();

function _mtFontFilenameFromPath(fontPath) {
  return (fontPath || '').split(/[\\/]/).pop();
}

async function getActiveFontFace(serverUrl) {
  const infoRes = await fetch(`${serverUrl}/GetFont`);
  if (!infoRes.ok) throw new Error(`GetFont failed: HTTP ${infoRes.status}`);
  const info = await infoRes.json();
  const filename = _mtFontFilenameFromPath(info.font_path);
  const cacheKey = `${serverUrl}::${filename}`;

  if (_mtFontByteCache.has(cacheKey)) {
    return { face: _mtFontByteCache.get(cacheKey), filename, strokeWidth: info.stroke_width };
  }

  const res = await fetch(`${serverUrl}/v1/font`);
  if (!res.ok) throw new Error(`font fetch failed: HTTP ${res.status}`);
  const buf = await res.arrayBuffer();
  const family = `MTFont_${filename.replace(/[^a-zA-Z0-9]/g, '_')}`;
  const face = new FontFace(family, buf);
  await face.load();
  document.fonts.add(face);
  _mtFontByteCache.set(cacheKey, face);
  return { face, filename, strokeWidth: info.stroke_width };
}

// Load a specific font (by filename) as a FontFace so a preview element can be
// rendered in that font's own typeface. Returns the CSS family name, or null
// if the font could not be loaded.
async function loadFontFaceByName(serverUrl, filename) {
  if (!serverUrl || !filename) return null;
  const cacheKey = `${serverUrl}::${filename}`;
  const family = `MTFont_${filename.replace(/[^a-zA-Z0-9]/g, '_')}`;
  if (_mtFontByteCache.has(cacheKey)) return family;
  try {
    const res = await fetch(`${serverUrl}/v1/font/${encodeURIComponent(filename)}`);
    if (!res.ok) return null;
    const buf = await res.arrayBuffer();
    const face = new FontFace(family, buf);
    await face.load();
    document.fonts.add(face);
    _mtFontByteCache.set(cacheKey, face);
    return family;
  } catch (e) {
    console.warn(`[MangaTranslator] Could not load font preview for ${filename}:`, e);
    return null;
  }
}

function drawFontWeightSwatch(canvas, level, fontFamily) {
  const ctx = canvas.getContext('2d');
  const w = canvas.width, h = canvas.height;
  const dpr = window.devicePixelRatio || 1;

  ctx.clearRect(0, 0, w, h);
  ctx.fillStyle = '#8a8a8a';
  ctx.fillRect(0, 0, w, h);

  const fontSize = (16 + level * 2) * dpr;
  ctx.font = `${fontSize}px "${fontFamily}"`;
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  ctx.lineJoin = 'round';
  const text = 'Aあ';
  const cx = (w * dpr) / 2, cy = (h * dpr) / 2;

  if (level > 0) {
    ctx.lineWidth = level * 2.2 * dpr;
    ctx.strokeStyle = '#ffffff';
    ctx.strokeText(text, cx, cy);
  }
  ctx.fillStyle = '#111111';
  ctx.fillText(text, cx, cy);
}

async function initFontWeightPicker() {
  const container = document.getElementById('fontWeightPicker');
  if (!container) return;
  container.innerHTML = '';

  const { serverUrl, fontWeight } = await chrome.storage.local.get(['serverUrl', 'fontWeight']);
  const selected = fontWeight !== undefined ? parseInt(fontWeight, 10) : 2;
  document.getElementById('fontWeightHidden').value = selected;

  let fontFamily = 'sans-serif';
  if (serverUrl) {
    try {
      const { face } = await getActiveFontFace(serverUrl);
      fontFamily = face.family;
    } catch (e) {
      console.warn('[MangaTranslator] Could not load server font for preview, using fallback:', e);
    }
  }

  const labels = ['Thin', 'Light', 'Regular', 'Bold', 'Heavy'];
  for (let level = 0; level <= 4; level++) {
    const wrap = document.createElement('div');
    wrap.style.cssText = 'display:flex; flex-direction:column; align-items:center; cursor:pointer;';
    wrap.title = `${level + 1} - ${labels[level]}`;

    const dpr = window.devicePixelRatio || 1;
    const canvas = document.createElement('canvas');
    const cssW = 44, cssH = 34;
    canvas.style.width = cssW + 'px';
    canvas.style.height = cssH + 'px';
    canvas.width = cssW * dpr;
    canvas.height = cssH * dpr;
    canvas.dataset.level = level;
    canvas.style.cssText += `border-radius:4px; border:2px solid ${level === selected ? '#28a745' : '#555'}; display:block;`;
    drawFontWeightSwatch(canvas, level, fontFamily);

    const lbl = document.createElement('div');
    lbl.innerText = level + 1;
    lbl.style.cssText = 'font-size:10px; color:#aaa; margin-top:2px;';

    wrap.appendChild(canvas);
    wrap.appendChild(lbl);
    wrap.onclick = () => {
      document.getElementById('fontWeightHidden').value = level;
      chrome.storage.local.set({ fontWeight: String(level) });
      container.querySelectorAll('canvas').forEach(c => {
        c.style.border = `2px solid ${parseInt(c.dataset.level, 10) === level ? '#28a745' : '#555'}`;
      });
    };
    container.appendChild(wrap);
  }
}

// ============================================================================
// FONT FAMILY PICKER
// ============================================================================
function attachWheelHorizontalScroll(container) {
  if (container.dataset.wheelBound === '1') return;
  container.dataset.wheelBound = '1';
  container.addEventListener('wheel', (e) => {
    if (e.deltaY === 0 && e.deltaX === 0) return;
    container.scrollLeft += e.deltaY !== 0 ? e.deltaY : e.deltaX;
    e.preventDefault();
  }, { passive: false });
}

async function initFontFamilyPicker(serverUrl) {
  const container = document.getElementById('fontFamilyScroll');
  if (!container) return;
  attachWheelHorizontalScroll(container);

  if (!serverUrl) {
    container.innerHTML = '<div class="font-loading">Set a Server URL first</div>';
    return;
  }
  container.innerHTML = '<div class="font-loading">Loading fonts…</div>';

  let fonts = [];
  let activeFilename = null;
  try {
    const [fontsRes, activeRes] = await Promise.all([
      fetch(`${serverUrl}/GetFonts`),
      fetch(`${serverUrl}/GetFont`)
    ]);
    const fontsData = await fontsRes.json();
    const activeData = await activeRes.json();
    fonts = fontsData.fonts || [];
    activeFilename = _mtFontFilenameFromPath(activeData.font_path);
  } catch (e) {
    container.innerHTML = `<div class="font-loading error">Could not load fonts: ${e}</div>`;
    return;
  }

  container.innerHTML = '';
  if (fonts.length === 0) {
    container.innerHTML = '<div class="font-loading">No fonts found in server fonts folder.</div>';
    return;
  }

  fonts.forEach(f => {
    const chip = document.createElement('button');
    chip.type = 'button';
    chip.className = 'font-chip' + (f.filename === activeFilename ? ' active' : '');
    chip.innerText = f.name;
    chip.title = `${f.filename} (${f.size_kb} KB)`;
    chip.dataset.filename = f.filename;
    chip.onclick = () => selectFontFamily(serverUrl, f.filename, container);
    container.appendChild(chip);

    // Render each chip in its OWN typeface so the picker is a true preview.
    loadFontFaceByName(serverUrl, f.filename).then(family => {
      if (family) chip.style.fontFamily = `"${family}", sans-serif`;
    });
  });
}

async function selectFontFamily(serverUrl, filename, container) {
  const statusEl = document.getElementById('fontFamilyStatus');
  statusEl.innerText = `Switching to ${filename}...`;
  try {
    const { fontWeight } = await chrome.storage.local.get(['fontWeight']);
    const strokeWidth = fontWeight !== undefined ? parseInt(fontWeight, 10) : 2;

    const res = await fetch(`${serverUrl}/SetFont`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ font_name: filename, stroke_width: strokeWidth })
    });
    const data = await res.json();
    if (!res.ok) {
      statusEl.innerHTML = `<span style="color:#ff4d4d;">Error: ${data.detail}</span>`;
      return;
    }

    statusEl.innerText = `Active: ${filename}`;
    chrome.storage.local.set({ fontFamily: filename });
    container.querySelectorAll('.font-chip').forEach(c => {
      c.classList.toggle('active', c.dataset.filename === filename);
    });

    await initFontWeightPicker();
  } catch (e) {
    statusEl.innerHTML = `<span style="color:#ff4d4d;">Error: ${e}</span>`;
  }
}

// ============================================================================
// INPAINTING & OCR MODE HELPERS
// ============================================================================
async function syncInpaintModeFromServer(serverUrl) {
  const statusEl = document.getElementById('inpaintModeStatus');
  if (!serverUrl) return;
  try {
    const res = await fetch(`${serverUrl}/GetInpaintMode`);
    const data = await res.json();
    document.getElementById('inpaintMode').value = data.inpaint_mode || 'low';
    chrome.storage.local.set({ inpaintMode: data.inpaint_mode || 'low' });
    if (data.inpaint_mode === 'high') {
      statusEl.innerText = data.high_model_downloaded
        ? `High model ready (${data.high_model_size_mb} MB)`
        : 'High model will download on first use';
    } else if (data.inpaint_mode === 'none') {
      statusEl.innerText = 'None mode active (no model loaded)';
    } else {
      statusEl.innerText = '';
    }
  } catch (e) {
    console.warn('[MangaTranslator] Could not fetch inpaint mode from server:', e);
  }
}

async function pushInpaintMode(serverUrl, mode) {
  const statusEl = document.getElementById('inpaintModeStatus');
  if (!serverUrl) {
    statusEl.innerHTML = `<span style="color:#ff4d4d;">Set a Server URL first</span>`;
    return;
  }
  statusEl.innerText = mode === 'high' ? 'Switching (may download model)...' : 'Switching...';
  try {
    const res = await fetch(`${serverUrl}/SetInpaintMode`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ mode })
    });
    const data = await res.json();
    if (res.ok) {
      if (data.inpaint_mode === 'high') {
        statusEl.innerText = `High model ready (${data.high_model_size_mb} MB)`;
      } else if (data.inpaint_mode === 'none') {
        statusEl.innerText = 'None mode active (no model loaded)';
      } else {
        statusEl.innerText = 'Low mode active';
      }
    } else {
      statusEl.innerHTML = `<span style="color:#ff4d4d;">Error: ${data.detail}</span>`;
    }
  } catch (e) {
    statusEl.innerHTML = `<span style="color:#ff4d4d;">Error: ${e}</span>`;
  }
}

async function syncOcrModeFromServer(serverUrl) {
  const statusEl = document.getElementById('ocrModeStatus');
  if (!serverUrl) return;
  try {
    const res = await fetch(`${serverUrl}/GetOcrMode`);
    const data = await res.json();
    document.getElementById('ocrMode').value = data.ocr_mode || 'hayai';
    chrome.storage.local.set({ ocrMode: data.ocr_mode || 'hayai' });
    statusEl.innerText = data.ocr_mode === 'lens' ? 'Google Lens active' : (data.ocr_mode === 'glm' ? 'GLM active' : 'Hayai active');
  } catch (e) {
    console.warn('[MangaTranslator] Could not fetch OCR mode from server:', e);
  }
}

async function pushOcrMode(serverUrl, mode) {
  const statusEl = document.getElementById('ocrModeStatus');
  if (!serverUrl) {
    statusEl.innerHTML = `<span style="color:#ff4d4d;">Set a Server URL first</span>`;
    return;
  }
  statusEl.innerText = 'Switching...';
  try {
    const res = await fetch(`${serverUrl}/SetOcrMode`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ mode })
    });
    const data = await res.json();
    if (res.ok) {
      statusEl.innerText = data.ocr_mode === 'lens' ? 'Google Lens active' : (data.ocr_mode === 'glm' ? 'GLM active' : 'Hayai active');
      chrome.storage.local.set({ ocrMode: data.ocr_mode });
    } else {
      statusEl.innerHTML = `<span style="color:#ff4d4d;">Error: ${data.detail}</span>`;
    }
  } catch (e) {
    statusEl.innerHTML = `<span style="color:#ff4d4d;">Error: ${e}</span>`;
  }
}

// ============================================================================
// CLOUD MODE — offload everything to the cloud, use minimum local resources.
// Forces Google Lens OCR + OpenRouter translation + no local inpainting model,
// and disables colorization (a heavy local model). The local-only controls are
// disabled while cloud mode is on so the choices can't drift out of sync.
// ============================================================================
function applyCloudMode(on) {
  const ocrModeEl   = document.getElementById('ocrMode');
  const modelTypeEl = document.getElementById('modelType');
  const inpaintEl   = document.getElementById('inpaintMode');
  const colorizeEl  = document.getElementById('colorize');
  const orBox       = document.getElementById('openrouterBox');

  if (on) {
    ocrModeEl.value = 'lens';
    modelTypeEl.value = 'openrouter';
    inpaintEl.value = 'none';
    colorizeEl.checked = false;
    orBox.style.display = 'block';          // still need the API key/model fields
  }
  // Lock the local-resource controls while cloud mode is active.
  [ocrModeEl, inpaintEl, colorizeEl].forEach(el => { el.disabled = on; });
  modelTypeEl.disabled = on;
}

// Push the cloud-mode server settings (lens + openrouter + none) so the backend
// stops loading local models. Reuses the OpenRouter model + API key the user
// already saved so they don't have to re-enter them. Best-effort; errors are
// logged, not fatal.
async function pushCloudModeToServer(serverUrl) {
  if (!serverUrl) return;

  // Reuse previously entered OpenRouter details (prefer the live fields, fall
  // back to what's cached in storage) so cloud mode works without re-typing.
  const liveModel = document.getElementById('openrouterModel').value.trim();
  const liveKey = document.getElementById('openrouterKey').value.trim();
  const cached = await chrome.storage.local.get(['openrouterModel', 'openrouterApiKey']);
  const model = liveModel || cached.openrouterModel || '';
  const apiKey = liveKey || cached.openrouterApiKey || '';

  const statusEl = document.getElementById('status');
  try {
    // One call flips the backend to lens + openrouter + none. The server reuses
    // any key it already has, so model/api_key are optional overrides here.
    const body = { enabled: true };
    if (model) body.model = model;
    if (apiKey) body.api_key = apiKey;

    const res = await fetch(`${serverUrl}/SetCloudMode`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
    const data = await res.json();
    if (!res.ok) {
      if (statusEl) statusEl.innerText = data.detail || 'Cloud mode failed to enable.';
      return;
    }
    // Re-cache so both popups + options stay in sync.
    chrome.storage.local.set({
      modelType: 'openrouter', ocrMode: 'lens', inpaintMode: 'none',
      openrouterModel: data.openrouter_model || model,
      openrouterApiKey: apiKey || cached.openrouterApiKey,
    });
    if (statusEl) statusEl.innerText = `Cloud mode on — Lens + ${data.openrouter_model || 'OpenRouter'}`;
  } catch (e) {
    console.warn('[MangaTranslator] Cloud mode: failed to enable on server:', e);
    if (statusEl) statusEl.innerText = 'Could not reach server to enable cloud mode.';
  }
}

// Tell the server to leave cloud mode (best-effort).
async function disableCloudModeOnServer(serverUrl) {
  if (!serverUrl) return;
  try {
    await fetch(`${serverUrl}/SetCloudMode`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ enabled: false }),
    });
  } catch (e) {
    console.warn('[MangaTranslator] Failed to disable cloud mode on server:', e);
  }
}

// ============================================================================
// INIT — autoload all cached settings into dropdowns/fields
// ============================================================================
document.addEventListener('DOMContentLoaded', async () => {
  const targetSel = document.getElementById('targetLang');
  const ocrLangSel = document.getElementById('ocrLang');

  const data = await chrome.storage.local.get(
    ['serverUrl', 'ocrLang', 'colorize', 'targetLang', 'fontWeight', 'modelType',
     'openrouterModel', 'openrouterApiKey', 'inpaintMode', 'ocrMode', 'cloudMode']
  );

  // Populate language dropdowns from the built-in list first so they're never
  // empty, then refresh from the server (which may add/rename languages).
  mtPopulateLangSelect(targetSel, data.targetLang || 'en');
  mtPopulateLangSelect(ocrLangSel, data.ocrLang || 'ja');

  document.getElementById('serverUrl').value = data.serverUrl || 'http://localhost:7860';
  document.getElementById('colorize').checked = data.colorize !== false;
  document.getElementById('inpaintMode').value = data.inpaintMode || 'low';
  document.getElementById('ocrMode').value = data.ocrMode || 'hayai';

  const modelType = data.modelType || 'local';
  document.getElementById('modelType').value = modelType;
  document.getElementById('openrouterBox').style.display = modelType === 'openrouter' ? 'block' : 'none';
  if (data.openrouterModel) {
    document.getElementById('openrouterModel').value = data.openrouterModel;
  }
  // ★ Load cached API key (displays as •••• because input is type="password")
  if (data.openrouterApiKey) {
    document.getElementById('openrouterKey').value = data.openrouterApiKey;
  }

  // ★ Cloud Mode — restore the toggle and apply its locked state.
  const cloudOn = data.cloudMode === true;
  document.getElementById('cloudMode').checked = cloudOn;
  applyCloudMode(cloudOn);

  syncInpaintModeFromServer(data.serverUrl);
  syncOcrModeFromServer(data.serverUrl);

  const initUrl = data.serverUrl || '';
  initFontFamilyPicker(initUrl);
  initFontWeightPicker();

  // Refresh language lists from the server (falls back to built-in on error).
  if (initUrl) {
    const langs = await mtFetchLanguages(initUrl);
    mtPopulateLangSelect(targetSel, targetSel.value || data.targetLang || 'en', langs);
    mtPopulateLangSelect(ocrLangSel, ocrLangSel.value || data.ocrLang || 'ja', langs);
  }
});

document.getElementById('serverUrl').addEventListener('change', (e) => {
  const url = e.target.value.trim().replace(/\/$/, '');
  const container = document.getElementById('fontWeightPicker');
  if (container) container.innerHTML = '<div style="font-size:11px; color:#888;">Loading preview…</div>';
  initFontFamilyPicker(url);
  initFontWeightPicker();
  syncInpaintModeFromServer(url);
  syncOcrModeFromServer(url);
});

document.getElementById('modelType').addEventListener('change', (e) => {
  const isOpenRouter = e.target.value === 'openrouter';
  document.getElementById('openrouterBox').style.display = isOpenRouter ? 'block' : 'none';
  chrome.storage.local.set({ modelType: e.target.value });
});

document.getElementById('inpaintMode').addEventListener('change', (e) => {
  const serverUrl = document.getElementById('serverUrl').value.trim().replace(/\/$/, '');
  chrome.storage.local.set({ inpaintMode: e.target.value });
  pushInpaintMode(serverUrl, e.target.value);
});

document.getElementById('ocrMode').addEventListener('change', (e) => {
  const serverUrl = document.getElementById('serverUrl').value.trim().replace(/\/$/, '');
  chrome.storage.local.set({ ocrMode: e.target.value });
  pushOcrMode(serverUrl, e.target.value);
});

document.getElementById('cloudMode').addEventListener('change', async (e) => {
  const on = e.target.checked;
  applyCloudMode(on);
  const serverUrl = document.getElementById('serverUrl').value.trim().replace(/\/$/, '');

  if (on) {
    // Persist the forced cloud selections so every surface stays in sync.
    chrome.storage.local.set({
      cloudMode: true,
      ocrMode: 'lens',
      modelType: 'openrouter',
      inpaintMode: 'none',
      colorize: false,
    });
    await pushCloudModeToServer(serverUrl);
  } else {
    // Leaving cloud mode: keep the current (now re-enabled) selections as-is.
    chrome.storage.local.set({ cloudMode: false });
    await disableCloudModeOnServer(serverUrl);
  }
});

// ============================================================================
// SET MODEL — pushes to server AND caches the API key
// ============================================================================
document.getElementById('setModelBtn').addEventListener('click', async () => {
  const serverUrl = document.getElementById('serverUrl').value.trim().replace(/\/$/, '');
  const model = document.getElementById('openrouterModel').value.trim();
  const apiKey = document.getElementById('openrouterKey').value.trim();
  const statusEl = document.getElementById('modelStatus');

  if (!serverUrl) {
    alert("Please set your FastAPI Server URL first!");
    return;
  }
  if (!model) {
    alert("Please enter an OpenRouter model ID.");
    return;
  }

  statusEl.innerText = 'Setting model...';
  try {
    const body = { model_type: 'openrouter', model: model };
    if (apiKey) body.api_key = apiKey;

    const res = await fetch(`${serverUrl}/SetModelType`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body)
    });
    const data = await res.json();
    if (res.ok) {
      // ★ Cache model + API key so they persist across popup reopens
      chrome.storage.local.set({ modelType: 'openrouter', openrouterModel: model, openrouterApiKey: apiKey });
      statusEl.innerText = `Active: ${data.openrouter_model}`;
    } else {
      statusEl.innerHTML = `<span style="color:#ff4d4d;">Error: ${data.detail}</span>`;
    }
  } catch (err) {
    statusEl.innerHTML = `<span style="color:#ff4d4d;">Error: ${err}</span>`;
  }
});

// ============================================================================
// SAVE SETTINGS — caches EVERYTHING to chrome.storage.local
// ============================================================================
document.getElementById('saveBtn').addEventListener('click', async () => {
  const url = document.getElementById('serverUrl').value.trim().replace(/\/$/, '');
  const ocrMode = document.getElementById('ocrMode').value;
  const ocrLang = document.getElementById('ocrLang').value;
  const colorize = document.getElementById('colorize').checked;
  const targetLang = document.getElementById('targetLang').value;
  const fontWeight = document.getElementById('fontWeightHidden').value;
  const modelType = document.getElementById('modelType').value;
  const inpaintMode = document.getElementById('inpaintMode').value;
  const openrouterModel = document.getElementById('openrouterModel').value.trim();
  const openrouterApiKey = document.getElementById('openrouterKey').value.trim();
  const cloudMode = document.getElementById('cloudMode').checked;

  // ★ Cache all settings including the API key
  chrome.storage.local.set({
    serverUrl: url,
    ocrMode: ocrMode,
    ocrLang: ocrLang,
    colorize: colorize,
    targetLang: targetLang,
    fontWeight: fontWeight,
    modelType: modelType,
    inpaintMode: inpaintMode,
    openrouterModel: openrouterModel,
    openrouterApiKey: openrouterApiKey,
    cloudMode: cloudMode
  }, () => {
    const status = document.getElementById('status');
    status.innerText = 'Settings saved & cached!';
    setTimeout(() => status.innerText = '', 2000);
  });

  if (url) {
    try {
      await fetch(`${url}/SetFont`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ stroke_width: parseInt(fontWeight, 10) })
      });
    } catch (e) {
      console.warn('[MangaTranslator] Failed to push font weight to server:', e);
    }

    await pushInpaintMode(url, inpaintMode);
    await pushOcrMode(url, ocrMode);

    if (modelType === 'local') {
      try {
        await fetch(`${url}/SetModelType`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ model_type: 'local' })
        });
      } catch (e) {
        console.warn('[MangaTranslator] Failed to switch server to local model:', e);
      }
    }
  }
});

document.getElementById('translateBtn').addEventListener('click', async () => {
  // Snapshot every current control so the choices persist across popup reopens
  // and stay in sync with the content-script popup and options page.
  const cloudMode = document.getElementById('cloudMode').checked;
  const settings = {
    serverUrl: document.getElementById('serverUrl').value.trim().replace(/\/$/, ''),
    ocrMode: document.getElementById('ocrMode').value,
    ocrLang: document.getElementById('ocrLang').value,
    targetLang: document.getElementById('targetLang').value,
    colorize: document.getElementById('colorize').checked,
    inpaintMode: document.getElementById('inpaintMode').value,
    modelType: document.getElementById('modelType').value,
    fontWeight: document.getElementById('fontWeightHidden').value,
    openrouterModel: document.getElementById('openrouterModel').value.trim(),
    openrouterApiKey: document.getElementById('openrouterKey').value.trim(),
    cloudMode: cloudMode,
  };

  // In cloud mode, make sure the server is actually configured for the cloud
  // path (Google Lens + OpenRouter + no local inpaint) using the OpenRouter
  // details the user already entered — no need to re-type them.
  if (cloudMode && settings.serverUrl) {
    await pushCloudModeToServer(settings.serverUrl);
  }

  chrome.storage.local.set(settings, () => {
    chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
      chrome.tabs.sendMessage(tabs[0].id, {
        action: "translateAllImages",
        ocrLang: settings.ocrLang,
        targetLang: settings.targetLang
      });
      window.close();
    });
  });
});

document.getElementById('optionsBtn').addEventListener('click', () => {
  chrome.runtime.openOptionsPage();
});

// ============================================================================
// SETTINGS PANEL TOGGLE — keep the popup clean; reveal detail on demand so the
// Translate button stays the focal point.
// ============================================================================
(function initSettingsToggle() {
  const toggle = document.getElementById('settingsToggleBtn');
  const panel = document.getElementById('settingsPanel');
  if (!toggle || !panel) return;

  const setOpen = (open) => {
    panel.style.display = open ? 'block' : 'none';
    toggle.classList.toggle('open', open);
    toggle.setAttribute('aria-expanded', open ? 'true' : 'false');
    chrome.storage.local.set({ settingsPanelOpen: open });
  };

  toggle.addEventListener('click', () => {
    setOpen(panel.style.display === 'none');
  });

  // Restore last open/closed state.
  chrome.storage.local.get(['settingsPanelOpen'], (d) => setOpen(d.settingsPanelOpen === true));
})();