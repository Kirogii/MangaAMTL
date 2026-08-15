// ============================================================================
// FONT PREVIEW HELPERS
// ============================================================================
const _mtFontByteCache = new Map();

function _mtFontFilenameFromPath(fontPath) {
  return (fontPath || '').split(/[\\/]/).pop();
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
    const res = await fetch(`${serverUrl}/SetFont`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ font_name: filename })
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
  } catch (e) {
    statusEl.innerHTML = `<span style="color:#ff4d4d;">Error: ${e}</span>`;
  }
}

// ============================================================================
// STYLE FONT PICKERS
// ============================================================================
const STYLE_FONT_FIELDS = [
  { style: 'bold', selectId: 'styleFontBold', storageKey: 'styleFontBold' },
  { style: 'italic', selectId: 'styleFontItalic', storageKey: 'styleFontItalic' },
  { style: 'regular', selectId: 'styleFontRegular', storageKey: 'styleFontRegular' },
];

// Fill the three per-style <select>s with the server's font list, restore the
// saved choice for each, and persist changes. An empty value means "use the
// main font" and is always kept as the first option.
async function initStyleFontPickers(serverUrl, saved) {
  const statusEl = document.getElementById('styleFontStatus');
  const selects = STYLE_FONT_FIELDS.map(f => ({
    ...f,
    el: document.getElementById(f.selectId),
    savedValue: (saved && saved[f.storageKey]) || '',
  })).filter(f => f.el);
  if (selects.length === 0) return;

  const bind = (f) => {
    f.el.value = f.savedValue;
    f.el.onchange = () => chrome.storage.local.set({ [f.storageKey]: f.el.value });
  };

  if (!serverUrl) {
    if (statusEl) statusEl.innerText = 'Set a Server URL to list fonts.';
    selects.forEach(bind);
    return;
  }

  if (statusEl) statusEl.innerText = 'Loading fonts…';
  let fonts = [];
  try {
    const res = await fetch(`${serverUrl}/GetFonts`);
    const data = await res.json();
    fonts = data.fonts || [];
  } catch (e) {
    if (statusEl) statusEl.innerHTML = `<span style="color:#ff4d4d;">Could not load fonts: ${e}</span>`;
    selects.forEach(bind);
    return;
  }

  selects.forEach(f => {
    // Keep the pre-seeded "Main font" option, drop anything stale after it.
    while (f.el.options.length > 1) f.el.remove(1);
    fonts.forEach(font => {
      const opt = document.createElement('option');
      opt.value = font.filename;
      opt.textContent = font.name;
      f.el.appendChild(opt);
      loadFontFaceByName(serverUrl, font.filename).then(family => {
        if (family) opt.style.fontFamily = `"${family}", sans-serif`;
      });
    });
    bind(f);
  });

  if (statusEl) {
    statusEl.innerText = fonts.length
      ? `${fonts.length} fonts available.`
      : 'No fonts found in server fonts folder.';
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

function ocrModeLabel(mode) {
  switch (mode) {
    case 'lens': return 'Google Lens active';
    case 'glm': return 'GLM active';
    case 'openai_endpoint': return 'OpenAI Endpoint OCR active';
    case 'google_ai': return 'Google AI Studio OCR active (all text)';
    case 'local_vision': return 'Local GGUF Vision OCR active';
    default: return 'Hayai active';
  }
}

async function syncOcrModeFromServer(serverUrl) {
  const statusEl = document.getElementById('ocrModeStatus');
  if (!serverUrl) return;
  try {
    const res = await fetch(`${serverUrl}/GetOcrMode`);
    const data = await res.json();
    document.getElementById('ocrMode').value = data.ocr_mode || 'hayai';
    document.getElementById('openaiOcrBox').style.display = data.ocr_mode === 'openai_endpoint' ? 'block' : 'none';
    document.getElementById('googleAiOcrBox').style.display = data.ocr_mode === 'google_ai' ? 'block' : 'none';
    chrome.storage.local.set({ ocrMode: data.ocr_mode || 'hayai' });
    statusEl.innerText = ocrModeLabel(data.ocr_mode);
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
      statusEl.innerText = ocrModeLabel(data.ocr_mode);
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
  applyContextLevelGate();
}

// ============================================================================
// CONTEXT LEVEL GATE — High rides on a vision request, which only OpenRouter
// can make. Cloud mode forces the backend to OpenRouter, so it counts as
// eligible even before the modelType select is read back.
// ============================================================================
// Assigned by the DOMContentLoaded handler once the context controls exist;
// the gate calls it so the style-font button follows a forced level change.
let applyCtxVisibility = () => {};

function applyContextLevelGate() {
  const ctxLevelEl = document.getElementById('contextLevel');
  if (!ctxLevelEl) return;
  const cloudOn = document.getElementById('cloudMode')?.checked === true;
  const isOpenRouter = cloudOn || document.getElementById('modelType')?.value === 'openrouter';

  const highOpt = ctxLevelEl.querySelector('option[value="high"]');
  if (highOpt) highOpt.disabled = !isOpenRouter;

  if (!isOpenRouter && ctxLevelEl.value === 'high') {
    ctxLevelEl.value = 'low';
    chrome.storage.local.set({ contextLevel: 'low' });
  }

  const note = document.getElementById('contextHighNote');
  if (note) note.style.display = isOpenRouter ? 'none' : 'block';

  applyCtxVisibility();
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
// Unified settings push — single POST to /SetAllSettings before Translate.
// Handles the cloud-mode-restart problem: if the backend was restarted (its
// in-memory settings reset to defaults) while cloud mode was on in the
// extension, this re-applies the correct state in one shot.
// ============================================================================
async function pushAllSettings(serverUrl, settings) {
  if (!serverUrl) {
    return { ok: false, error: 'Set a FastAPI Server URL first.' };
  }
  try {
    const res = await fetch(`${serverUrl}/SetAllSettings`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(settings),
    });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) {
      const error = data.detail || `Server returned HTTP ${res.status}`;
      console.warn('[MangaTranslator] /SetAllSettings failed:', error);
      return { ok: false, error };
    }
    console.log('[MangaTranslator] Settings applied:', data.applied);
    return { ok: true, data };
  } catch (e) {
    const error = `Could not apply settings: ${e.message || e}`;
    console.warn('[MangaTranslator] /SetAllSettings error:', e);
    return { ok: false, error };
  }
}

// Build the full settings payload from the current popup control values.
// In cloud mode this forces ocr=lens / model_type=openrouter / inpaint=none.
function buildSettingsPayload() {
  const cloudMode = document.getElementById('cloudMode').checked;
  const contextMode = document.getElementById('contextMode').value === 'on';
  const contextLevel = document.getElementById('contextLevel').value === 'high' ? 'high' : 'low';
  const payload = {
    cloud_mode: cloudMode,
    free_openrouter: document.getElementById('freeOpenRouter').checked,
    context_aware: contextMode,
    context_level: contextMode ? contextLevel : 'low',
    style_aware: contextMode && contextLevel === 'high',
    style_fonts: {
      bold: document.getElementById('styleFontBold').value || '',
      italic: document.getElementById('styleFontItalic').value || '',
      regular: document.getElementById('styleFontRegular').value || '',
    },
  };

  if (cloudMode) {
    // Cloud-forced state
    payload.model_type = 'openrouter';
    payload.ocr_mode = 'lens';
    payload.inpaint_mode = 'none';
    const model = document.getElementById('openrouterModel').value.trim();
    const apiKey = document.getElementById('openrouterKey').value.trim();
    if (model) payload.openrouter_model = model;
    if (apiKey) payload.openrouter_api_key = apiKey;
  } else {
    // User's actual choices
    payload.ocr_mode = document.getElementById('ocrMode').value;
    const ocrEndpoint = document.getElementById('openaiOcrEndpoint').value.trim();
    const ocrModel = document.getElementById('openaiOcrModel').value.trim();
    const ocrKey = document.getElementById('openaiOcrKey').value.trim();
    if (payload.ocr_mode === 'openai_endpoint') {
      payload.openai_ocr_endpoint = ocrEndpoint;
      payload.openai_ocr_model = ocrModel;
      payload.openai_ocr_api_key = ocrKey;
    }
    if (payload.ocr_mode === 'google_ai') {
      const selectedModel = document.getElementById('googleAiOcrModel').value;
      payload.google_ai_ocr_api_key = document.getElementById('googleAiOcrKey').value.trim();
      payload.google_ai_ocr_model = selectedModel === 'custom'
        ? document.getElementById('googleAiOcrCustomModel').value.trim()
        : selectedModel;
      payload.google_ai_ocr_rpm = parseInt(document.getElementById('googleAiOcrRpm').value, 10) || 5;
    }
    payload.inpaint_mode = document.getElementById('inpaintMode').value;
    payload.model_type = document.getElementById('modelType').value;
    const model = document.getElementById('openrouterModel').value.trim();
    const apiKey = document.getElementById('openrouterKey').value.trim();
    if (model) payload.openrouter_model = model;
    if (apiKey) payload.openrouter_api_key = apiKey;
    // Active font filename, if known
    const activeFont = document.querySelector('#fontFamilyScroll .font-chip.active');
    if (activeFont && activeFont.dataset.filename) {
      payload.font_filename = activeFont.dataset.filename;
    }
  }
  return payload;
}

// ============================================================================
// INIT — autoload all cached settings into dropdowns/fields
// ============================================================================
function updateLanguageWarning() {
  const source = document.getElementById('ocrLang');
  const target = document.getElementById('targetLang');
  const warning = document.getElementById('languageWarning');
  if (!source || !target || !warning) return false;
  const same = source.value && source.value === target.value;
  warning.innerText = same
    ? 'Source and target languages match. Select the manga\'s original language for accurate OCR.'
    : '';
  return same;
}

document.addEventListener('DOMContentLoaded', async () => {
  const targetSel = document.getElementById('targetLang');
  const ocrLangSel = document.getElementById('ocrLang');

  const data = await chrome.storage.local.get(
    ['serverUrl', 'ocrLang', 'colorize', 'targetLang', 'modelType',
     'openrouterModel', 'openrouterApiKey', 'inpaintMode', 'ocrMode', 'cloudMode',
     'openaiOcrEndpoint', 'openaiOcrModel', 'openaiOcrApiKey',
     'googleAiOcrApiKey', 'googleAiOcrModel', 'googleAiOcrCustomModel', 'googleAiOcrRpm',
     'combineAmount', 'freeOpenRouter',
     'contextMode', 'contextLevel', 'styleFontBold', 'styleFontItalic', 'styleFontRegular',
     'skipSfx', 'contextAware']
  );

  // Populate language dropdowns from the built-in list first so they're never
  // empty, then refresh from the server (which may add/rename languages).
  mtPopulateLangSelect(targetSel, data.targetLang || 'en');
  mtPopulateLangSelect(ocrLangSel, data.ocrLang || 'ja');

  document.getElementById('serverUrl').value = data.serverUrl || 'http://localhost:7860';
  document.getElementById('colorize').checked = data.colorize !== false;
  document.getElementById('inpaintMode').value = data.inpaintMode || 'low';
  document.getElementById('ocrMode').value = data.ocrMode || 'hayai';
  document.getElementById('openaiOcrBox').style.display = data.ocrMode === 'openai_endpoint' ? 'block' : 'none';
  document.getElementById('openaiOcrEndpoint').value = data.openaiOcrEndpoint || 'https://api.openai.com/v1';
  document.getElementById('openaiOcrModel').value = data.openaiOcrModel || 'gpt-4o-mini';
  document.getElementById('openaiOcrKey').value = data.openaiOcrApiKey || '';
  document.getElementById('googleAiOcrBox').style.display = data.ocrMode === 'google_ai' ? 'block' : 'none';
  document.getElementById('googleAiOcrKey').value = data.googleAiOcrApiKey || '';
  const googleModel = data.googleAiOcrModel || 'gemini-2.5-flash-lite';
  const googleModelSelect = document.getElementById('googleAiOcrModel');
  const knownGoogleModel = Array.from(googleModelSelect.options).some(option => option.value === googleModel);
  googleModelSelect.value = knownGoogleModel ? googleModel : 'custom';
  document.getElementById('googleAiOcrCustomModel').value = data.googleAiOcrCustomModel || (knownGoogleModel ? '' : googleModel);
  document.getElementById('googleAiOcrCustomModel').style.display = googleModelSelect.value === 'custom' ? 'block' : 'none';
  document.getElementById('googleAiOcrRpm').value = data.googleAiOcrRpm || 5;

  // ★ Combine slider — restore value + live label
  const combineAmount = parseInt(data.combineAmount || '1', 10);
  const combineSlider = document.getElementById('combineAmount');
  const combineLabel = document.getElementById('combineAmountVal');
  if (combineSlider && combineLabel) {
    combineSlider.value = combineAmount;
    combineLabel.textContent = combineAmount;
    combineSlider.oninput = () => {
      combineLabel.textContent = combineSlider.value;
      chrome.storage.local.set({ combineAmount: String(combineSlider.value) });
    };
  }

  // ★ Context (merged Skip SFX + Context Aware) — restore.
  // Users upgrading from the split settings inherit "on" if either was set.
  const ctxModeEl = document.getElementById('contextMode');
  const ctxLevelEl = document.getElementById('contextLevel');
  const ctxLevelRow = document.getElementById('contextLevelRow');
  const styleFontsBtn = document.getElementById('styleFontsBtn');
  const styleFontsBox = document.getElementById('styleFontsBox');
  if (ctxModeEl && ctxLevelEl) {
    const ctxMode = data.contextMode
      || ((data.skipSfx === true || data.contextAware === true) ? 'on' : 'off');
    const ctxLevel = data.contextLevel === 'high' ? 'high' : 'low';
    ctxModeEl.value = ctxMode;
    ctxLevelEl.value = ctxLevel;

    const applyCtxVisibilityLocal = () => {
      const on = ctxModeEl.value === 'on';
      const high = on && ctxLevelEl.value === 'high';
      if (ctxLevelRow) ctxLevelRow.style.display = on ? 'block' : 'none';
      if (styleFontsBtn) styleFontsBtn.style.display = high ? 'block' : 'none';
      if (styleFontsBox && !high) styleFontsBox.style.display = 'none';
    };
    // Hand the gate a way to refresh this once it forces High → Low.
    applyCtxVisibility = applyCtxVisibilityLocal;
    applyCtxVisibilityLocal();

    ctxModeEl.onchange = () => {
      chrome.storage.local.set({ contextMode: ctxModeEl.value });
      applyCtxVisibilityLocal();
    };
    ctxLevelEl.onchange = () => {
      chrome.storage.local.set({ contextLevel: ctxLevelEl.value });
      applyCtxVisibilityLocal();
    };
    if (styleFontsBtn && styleFontsBox) {
      styleFontsBtn.onclick = () => {
        styleFontsBox.style.display = styleFontsBox.style.display === 'block' ? 'none' : 'block';
      };
    }
  }

  // ★ Free OpenRouter — restore (default off)
  const freeOrEl = document.getElementById('freeOpenRouter');
  if (freeOrEl) {
    freeOrEl.checked = data.freeOpenRouter === true;
    freeOrEl.onchange = () => chrome.storage.local.set({ freeOpenRouter: freeOrEl.checked });
  }

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
  initStyleFontPickers(initUrl, data);

  // Refresh language lists from the server (falls back to built-in on error).
  if (initUrl) {
    const langs = await mtFetchLanguages(initUrl);
    mtPopulateLangSelect(targetSel, targetSel.value || data.targetLang || 'en', langs);
    mtPopulateLangSelect(ocrLangSel, ocrLangSel.value || data.ocrLang || 'ja', langs);
  }
  updateLanguageWarning();
  targetSel.addEventListener('change', updateLanguageWarning);
  ocrLangSel.addEventListener('change', updateLanguageWarning);
});

document.getElementById('serverUrl').addEventListener('change', (e) => {
  const url = e.target.value.trim().replace(/\/$/, '');
  initFontFamilyPicker(url);
  chrome.storage.local.get(['styleFontBold', 'styleFontItalic', 'styleFontRegular'], (saved) => {
    initStyleFontPickers(url, saved || {});
  });
  syncInpaintModeFromServer(url);
  syncOcrModeFromServer(url);
});

document.getElementById('modelType').addEventListener('change', (e) => {
  const isOpenRouter = e.target.value === 'openrouter';
  document.getElementById('openrouterBox').style.display = isOpenRouter ? 'block' : 'none';
  chrome.storage.local.set({ modelType: e.target.value });
  applyContextLevelGate();
});

document.getElementById('inpaintMode').addEventListener('change', (e) => {
  const serverUrl = document.getElementById('serverUrl').value.trim().replace(/\/$/, '');
  chrome.storage.local.set({ inpaintMode: e.target.value });
  pushInpaintMode(serverUrl, e.target.value);
});

document.getElementById('ocrMode').addEventListener('change', (e) => {
  const serverUrl = document.getElementById('serverUrl').value.trim().replace(/\/$/, '');
  document.getElementById('openaiOcrBox').style.display = e.target.value === 'openai_endpoint' ? 'block' : 'none';
  document.getElementById('googleAiOcrBox').style.display = e.target.value === 'google_ai' ? 'block' : 'none';
  chrome.storage.local.set({ ocrMode: e.target.value });
  if (!['openai_endpoint', 'google_ai'].includes(e.target.value)) pushOcrMode(serverUrl, e.target.value);
});

document.getElementById('googleAiOcrModel').addEventListener('change', (e) => {
  document.getElementById('googleAiOcrCustomModel').style.display = e.target.value === 'custom' ? 'block' : 'none';
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
  const modelType = document.getElementById('modelType').value;
  const inpaintMode = document.getElementById('inpaintMode').value;
  const openrouterModel = document.getElementById('openrouterModel').value.trim();
  const openrouterApiKey = document.getElementById('openrouterKey').value.trim();
  const cloudMode = document.getElementById('cloudMode').checked;
  const combineAmount = document.getElementById('combineAmount').value;
  const freeOpenRouter = document.getElementById('freeOpenRouter').checked;
  const openaiOcrEndpoint = document.getElementById('openaiOcrEndpoint').value.trim();
  const openaiOcrModel = document.getElementById('openaiOcrModel').value.trim();
  const openaiOcrApiKey = document.getElementById('openaiOcrKey').value.trim();
  const googleAiOcrModelSelect = document.getElementById('googleAiOcrModel').value;
  const googleAiOcrModel = googleAiOcrModelSelect === 'custom'
    ? document.getElementById('googleAiOcrCustomModel').value.trim()
    : googleAiOcrModelSelect;
  const googleAiOcrApiKey = document.getElementById('googleAiOcrKey').value.trim();
  const googleAiOcrRpm = parseInt(document.getElementById('googleAiOcrRpm').value, 10) || 5;
  const contextMode = document.getElementById('contextMode').value === 'on' ? 'on' : 'off';
  const contextLevel = document.getElementById('contextLevel').value === 'high' ? 'high' : 'low';

  // ★ Cache all settings including the API key
  chrome.storage.local.set({
    serverUrl: url,
    ocrMode: ocrMode,
    ocrLang: ocrLang,
    colorize: colorize,
    targetLang: targetLang,
    modelType: modelType,
    inpaintMode: inpaintMode,
    openrouterModel: openrouterModel,
    openrouterApiKey: openrouterApiKey,
    cloudMode: cloudMode,
    combineAmount: combineAmount,
    freeOpenRouter: freeOpenRouter,
    openaiOcrEndpoint: openaiOcrEndpoint,
    openaiOcrModel: openaiOcrModel,
    openaiOcrApiKey: openaiOcrApiKey,
    googleAiOcrApiKey: googleAiOcrApiKey,
    googleAiOcrModel: googleAiOcrModel,
    googleAiOcrCustomModel: googleAiOcrModelSelect === 'custom' ? googleAiOcrModel : '',
    googleAiOcrRpm: googleAiOcrRpm,
    contextMode: contextMode,
    contextLevel: contextLevel,
    styleFontBold: document.getElementById('styleFontBold').value || '',
    styleFontItalic: document.getElementById('styleFontItalic').value || '',
    styleFontRegular: document.getElementById('styleFontRegular').value || ''
  }, () => {
    const status = document.getElementById('status');
    status.innerText = 'Settings saved & cached!';
    setTimeout(() => status.innerText = '', 2000);
  });

  // ★ Unified settings push via /SetAllSettings (single call, handles drift)
  if (url) {
    await pushAllSettings(url, buildSettingsPayload());
  }
});

async function ensureTranslationContentScript(tab) {
  const url = tab.url || '';
  let protocol = '';
  try { protocol = new URL(url).protocol; } catch (error) {}
  if (!['http:', 'https:', 'file:'].includes(protocol)) {
    throw new Error('This browser page does not allow extension scripts. Open the manga on a normal website tab.');
  }

  try {
    const ping = await chrome.tabs.sendMessage(tab.id, { action: 'mangaTranslatorPing' });
    if (ping?.ready === true) return;
  } catch (error) {}

  try {
    await chrome.scripting.executeScript({
      target: { tabId: tab.id },
      files: ['languages.js', 'content.js', 'reader.js'],
    });
  } catch (error) {
    const recoveredPing = await chrome.tabs.sendMessage(tab.id, { action: 'mangaTranslatorPing' }).catch(() => null);
    if (recoveredPing?.ready !== true) throw error;
  }

  const ping = await chrome.tabs.sendMessage(tab.id, { action: 'mangaTranslatorPing' });
  if (!ping?.ready) {
    throw new Error('The manga page did not initialize the translation script.');
  }
}

function translationStartError(error) {
  const message = error instanceof Error ? error.message : String(error || 'Unknown startup error');
  if (/No eligible manga images/i.test(message)) return message;
  if (/This browser page does not allow|Cannot access contents|Cannot access a chrome|Missing host permission/i.test(message)) {
    return 'Chrome blocked access to this page. Open the manga in a normal http/https tab. For local files, enable “Allow access to file URLs” for the extension.';
  }
  return message;
}

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
    openrouterModel: document.getElementById('openrouterModel').value.trim(),
    openrouterApiKey: document.getElementById('openrouterKey').value.trim(),
    openaiOcrEndpoint: document.getElementById('openaiOcrEndpoint').value.trim(),
    openaiOcrModel: document.getElementById('openaiOcrModel').value.trim(),
    openaiOcrApiKey: document.getElementById('openaiOcrKey').value.trim(),
    googleAiOcrApiKey: document.getElementById('googleAiOcrKey').value.trim(),
    googleAiOcrModel: document.getElementById('googleAiOcrModel').value === 'custom'
      ? document.getElementById('googleAiOcrCustomModel').value.trim()
      : document.getElementById('googleAiOcrModel').value,
    googleAiOcrCustomModel: document.getElementById('googleAiOcrCustomModel').value.trim(),
    googleAiOcrRpm: parseInt(document.getElementById('googleAiOcrRpm').value, 10) || 5,
    cloudMode: cloudMode,
    combineAmount: document.getElementById('combineAmount').value,
    freeOpenRouter: document.getElementById('freeOpenRouter').checked,
    contextMode: document.getElementById('contextMode').value === 'on' ? 'on' : 'off',
    contextLevel: document.getElementById('contextLevel').value === 'high' ? 'high' : 'low',
    styleFontBold: document.getElementById('styleFontBold').value || '',
    styleFontItalic: document.getElementById('styleFontItalic').value || '',
    styleFontRegular: document.getElementById('styleFontRegular').value || '',
  };

  const status = document.getElementById('status');
  const translateButton = document.getElementById('translateBtn');
  if (settings.ocrLang === settings.targetLang) {
    status.innerText = 'Source and target match; using automatic OCR language detection.';
    updateLanguageWarning();
  }
  if (settings.ocrMode === 'google_ai' && !settings.googleAiOcrApiKey) {
    status.innerText = 'Google AI Studio OCR requires an API key.';
    return;
  }
  if (settings.ocrMode === 'google_ai' && !settings.googleAiOcrModel) {
    status.innerText = 'Select or enter a Gemini model ID.';
    return;
  }

  translateButton.disabled = true;
  status.innerText = 'Applying settings...';
  const settingsResult = await pushAllSettings(settings.serverUrl, buildSettingsPayload());
  if (!settingsResult.ok) {
    status.innerText = `Settings error: ${settingsResult.error}`;
    translateButton.disabled = false;
    return;
  }

  await chrome.storage.local.set(settings);
  status.innerText = 'Starting translation...';
  const tabs = await chrome.tabs.query({ active: true, currentWindow: true });
  if (!tabs.length || !tabs[0].id) {
    status.innerText = 'No active browser tab was found.';
    translateButton.disabled = false;
    return;
  }

  try {
    await ensureTranslationContentScript(tabs[0]);
    const response = await chrome.tabs.sendMessage(tabs[0].id, {
      action: "translateAllImages",
      ocrLang: settings.ocrLang === settings.targetLang ? 'auto' : settings.ocrLang,
      targetLang: settings.targetLang,
      combineAmount: settings.combineAmount,
      contextMode: settings.contextMode,
      contextLevel: settings.contextLevel,
      styleFonts: {
        bold: settings.styleFontBold,
        italic: settings.styleFontItalic,
        regular: settings.styleFontRegular,
      },
    });
    if (!response || response.ok !== true) {
      throw new Error(response?.error || 'The page did not acknowledge the translation request.');
    }
    window.close();
  } catch (error) {
    console.error('[MangaTranslator] Could not start translation in active tab:', error);
    status.innerText = `Could not start: ${translationStartError(error)}`;
    translateButton.disabled = false;
  }
});

document.getElementById('optionsBtn').addEventListener('click', () => {
  chrome.runtime.openOptionsPage();
});

// ── Clear Image Cache ───────────────────────────────────────────────────
// Removes all mtImgCache_* keys from chrome.storage.local. These are
// page images cached client-side by the reader for offline/fast reading.
document.getElementById('clearCacheBtn').addEventListener('click', () => {
  chrome.storage.local.get(null, (all) => {
    const keysToRemove = Object.keys(all).filter(k => k.startsWith('mtImgCache_'));
    if (keysToRemove.length === 0) {
      const status = document.getElementById('status');
      status.innerText = 'No cached images to clear.';
      setTimeout(() => status.innerText = '', 2000);
      return;
    }
    chrome.storage.local.remove(keysToRemove, () => {
      const status = document.getElementById('status');
      status.innerText = `Cleared ${keysToRemove.length} cached images.`;
      setTimeout(() => status.innerText = '', 2000);
      console.log(`[MangaTranslator] Cleared ${keysToRemove.length} cached images.`);
    });
  });
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