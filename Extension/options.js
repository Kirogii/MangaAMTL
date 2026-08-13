// ============================================================================
// FONT PREVIEW HELPERS
// ============================================================================
const _mtFontByteCache = new Map();

function _mtFontFilenameFromPath(fontPath) {
  return (fontPath || '').split(/[\\/]/).pop();
}

// Load an arbitrary font file (by filename) as a FontFace so each chip in the
// font picker can be rendered in its OWN typeface — a true preview.
async function loadFontFaceByName(serverUrl, filename) {
  if (!serverUrl || !filename) return null;
  const cacheKey = `${serverUrl}::${filename}`;
  const family = `MTFontOptions_${filename.replace(/[^a-zA-Z0-9]/g, '_')}`;
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
// STYLE FONT PICKERS (Context Aware → High mode)
// ============================================================================
const STYLE_FONT_FIELDS = [
  { style: 'bold',   selectId: 'optStyleFontBold',   storageKey: 'styleFontBold' },
  { style: 'italic', selectId: 'optStyleFontItalic', storageKey: 'styleFontItalic' },
  { style: 'regular', selectId: 'optStyleFontRegular', storageKey: 'styleFontRegular' },
];

async function initStyleFontPickers(serverUrl, saved) {
  const statusEl = document.getElementById('optStyleFontStatus');
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
    if (statusEl) statusEl.innerHTML = `<span class="error">Could not load fonts: ${e}</span>`;
    selects.forEach(bind);
    return;
  }

  selects.forEach(f => {
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
  const container = document.getElementById('optFontFamilyScroll');
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
  const statusEl = document.getElementById('optFontFamilyStatus');
  statusEl.innerText = `Switching to ${filename}...`;
  try {
    const res = await fetch(`${serverUrl}/SetFont`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ font_name: filename })
    });
    const data = await res.json();
    if (!res.ok) {
      statusEl.innerHTML = `<span class="error">Error: ${data.detail}</span>`;
      return;
    }

    statusEl.innerText = `Active: ${filename}`;
    chrome.storage.local.set({ fontFamily: filename });
    container.querySelectorAll('.font-chip').forEach(c => {
      c.classList.toggle('active', c.dataset.filename === filename);
    });
  } catch (e) {
    statusEl.innerHTML = `<span class="error">Error: ${e}</span>`;
  }
}

// ============================================================================
// OCR / INPAINTING / MODEL TYPE SYNC
// ============================================================================
async function syncModelTypeFromServer(serverUrl) {
  if (!serverUrl) return;
  try {
    const res = await fetch(`${serverUrl}/GetModelType`);
    const data = await res.json();
    document.getElementById('optModelType').value = data.model_type || 'local';
    document.getElementById('optOpenrouterRow').style.display = data.model_type === 'openrouter' ? 'block' : 'none';
    if (data.openrouter_model) {
      document.getElementById('optOpenrouterModel').value = data.openrouter_model;
    }
    chrome.storage.local.set({ modelType: data.model_type || 'local' });
    applyContextLevelGate();
  } catch (e) {
    console.warn('[MangaTranslator] Could not fetch model type from server:', e);
  }
}

// ============================================================================
// CONTEXT LEVEL GATE — High rides on a vision request (the page image is sent
// alongside the text), which only the OpenRouter backend can make. On the local
// GGUF backend the server degrades High to Low, so the option is disabled here
// rather than silently doing nothing.
// ============================================================================
// Assigned by the DOMContentLoaded handler once the context controls exist; the
// gate calls it so the style-font button follows a forced level change.
let applyCtxVisibility = () => {};

function applyContextLevelGate() {
  const ctxLevelEl = document.getElementById('optContextLevel');
  if (!ctxLevelEl) return;
  const isOpenRouter = document.getElementById('optModelType')?.value === 'openrouter';

  const highOpt = ctxLevelEl.querySelector('option[value="high"]');
  if (highOpt) highOpt.disabled = !isOpenRouter;

  if (!isOpenRouter && ctxLevelEl.value === 'high') {
    ctxLevelEl.value = 'low';
    chrome.storage.local.set({ contextLevel: 'low' });
  }

  const note = document.getElementById('optContextHighNote');
  if (note) note.style.display = isOpenRouter ? 'none' : 'block';

  applyCtxVisibility();
}

async function syncInpaintModeFromServer(serverUrl) {
  const statusEl = document.getElementById('optInpaintStatus');
  if (!serverUrl) return;
  try {
    const res = await fetch(`${serverUrl}/GetInpaintMode`);
    const data = await res.json();
    document.getElementById('optInpaintMode').value = data.inpaint_mode || 'low';
    chrome.storage.local.set({ inpaintMode: data.inpaint_mode || 'low' });
    if (data.inpaint_mode === 'high') {
      statusEl.innerText = data.high_model_downloaded ? `High model ready (${data.high_model_size_mb} MB)` : 'High model will download on first use';
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
  const statusEl = document.getElementById('optInpaintStatus');
  if (!serverUrl) {
    statusEl.innerHTML = `<span class="error">Set a Server URL first</span>`;
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
      statusEl.innerHTML = `<span class="error">Error: ${data.detail}</span>`;
    }
  } catch (e) {
    statusEl.innerHTML = `<span class="error">Error: ${e}</span>`;
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
  const statusEl = document.getElementById('optOcrStatus');
  if (!serverUrl) return;
  try {
    const res = await fetch(`${serverUrl}/GetOcrMode`);
    const data = await res.json();
    document.getElementById('optOcrMode').value = data.ocr_mode || 'hayai';
    document.getElementById('optOpenaiOcrRow').style.display = data.ocr_mode === 'openai_endpoint' ? 'block' : 'none';
    document.getElementById('optGoogleAiOcrRow').style.display = data.ocr_mode === 'google_ai' ? 'block' : 'none';
    chrome.storage.local.set({ ocrMode: data.ocr_mode || 'hayai' });
    statusEl.innerText = ocrModeLabel(data.ocr_mode);
  } catch (e) {
    console.warn('[MangaTranslator] Could not fetch OCR mode from server:', e);
  }
}

async function pushOcrMode(serverUrl, mode) {
  const statusEl = document.getElementById('optOcrStatus');
  if (!serverUrl) {
    statusEl.innerHTML = `<span class="error">Set a Server URL first</span>`;
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
      statusEl.innerHTML = `<span class="error">Error: ${data.detail}</span>`;
    }
  } catch (e) {
    statusEl.innerHTML = `<span class="error">Error: ${e}</span>`;
  }
}

// ============================================================================
// INIT — autoload ALL cached settings into dropdowns/fields on open
// ============================================================================
document.addEventListener('DOMContentLoaded', () => {
  chrome.storage.local.get(
    ['serverUrl', 'modelType', 'openrouterModel', 'openrouterApiKey', 'inpaintMode', 'ocrMode',
     'openaiOcrEndpoint', 'openaiOcrModel', 'openaiOcrApiKey',
     'googleAiOcrApiKey', 'googleAiOcrModel', 'googleAiOcrCustomModel', 'googleAiOcrRpm',
     'combineAmount', 'freeOpenRouter', 'contextMode', 'contextLevel',
     'styleFontBold', 'styleFontItalic', 'styleFontRegular', 'skipSfx', 'contextAware'],
    (data) => {
      const url = data.serverUrl || 'http://localhost:7860';
      document.getElementById('optServerUrl').value = url;

      // ★ Autoload cached dropdown selections
      if (data.ocrMode) document.getElementById('optOcrMode').value = data.ocrMode;
      document.getElementById('optOpenaiOcrRow').style.display = data.ocrMode === 'openai_endpoint' ? 'block' : 'none';
      document.getElementById('optOpenaiOcrEndpoint').value = data.openaiOcrEndpoint || 'https://api.openai.com/v1';
      document.getElementById('optOpenaiOcrModel').value = data.openaiOcrModel || 'gpt-4o-mini';
      document.getElementById('optOpenaiOcrKey').value = data.openaiOcrApiKey || '';
      document.getElementById('optGoogleAiOcrRow').style.display = data.ocrMode === 'google_ai' ? 'block' : 'none';
      document.getElementById('optGoogleAiOcrKey').value = data.googleAiOcrApiKey || '';
      const googleModel = data.googleAiOcrModel || 'gemini-2.5-flash-lite';
      const googleModelSelect = document.getElementById('optGoogleAiOcrModel');
      const knownGoogleModel = Array.from(googleModelSelect.options).some(option => option.value === googleModel);
      googleModelSelect.value = knownGoogleModel ? googleModel : 'custom';
      document.getElementById('optGoogleAiOcrCustomModel').value = data.googleAiOcrCustomModel || (knownGoogleModel ? '' : googleModel);
      document.getElementById('optGoogleAiOcrCustomModel').style.display = googleModelSelect.value === 'custom' ? 'block' : 'none';
      document.getElementById('optGoogleAiOcrRpm').value = data.googleAiOcrRpm || 5;
      if (data.inpaintMode) document.getElementById('optInpaintMode').value = data.inpaintMode;

      const cachedModelType = data.modelType || 'local';
      document.getElementById('optModelType').value = cachedModelType;
      document.getElementById('optOpenrouterRow').style.display = cachedModelType === 'openrouter' ? 'block' : 'none';
      if (data.openrouterModel) {
        document.getElementById('optOpenrouterModel').value = data.openrouterModel;
      }
      // ★ Load cached API key (displays as •••• because input is type="password")
      if (data.openrouterApiKey) {
        document.getElementById('optOpenrouterKey').value = data.openrouterApiKey;
      }

      // ★ Combine slider — restore value + live label
      const combineAmount = parseInt(data.combineAmount || '1', 10);
      const combineSlider = document.getElementById('optCombineAmount');
      const combineLabel = document.getElementById('optCombineAmountVal');
      if (combineSlider && combineLabel) {
        combineSlider.value = combineAmount;
        combineLabel.textContent = combineAmount;
        combineSlider.oninput = () => {
          combineLabel.textContent = combineSlider.value;
          chrome.storage.local.set({ combineAmount: String(combineSlider.value) });
        };
      }

      // ★ Context Aware (merged Skip SFX + old Text Context Aware) — restore
      const ctxModeEl = document.getElementById('optContextMode');
      const ctxLevelEl = document.getElementById('optContextLevel');
      const ctxLevelRow = document.getElementById('optContextLevelRow');
      const styleFontsBtn = document.getElementById('optStyleFontsBtn');
      const styleFontsBox = document.getElementById('optStyleFontsBox');
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
        applyContextLevelGate();
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
      const freeOrEl = document.getElementById('optFreeOpenRouter');
      if (freeOrEl) {
        freeOrEl.checked = data.freeOpenRouter === true;
        freeOrEl.onchange = () => chrome.storage.local.set({ freeOpenRouter: freeOrEl.checked });
      }

      initFontFamilyPicker(url);
      initStyleFontPickers(url, data);
      // Server syncs run after cache load — if server is online they keep things in sync
      syncModelTypeFromServer(url);
      syncInpaintModeFromServer(url);
      syncOcrModeFromServer(url);
    }
  );
});

// ============================================================================
// SAVE URL (individual button, still works)
// ============================================================================
document.getElementById('mtSaveUrlBtn').addEventListener('click', () => {
  const url = document.getElementById('optServerUrl').value.trim().replace(/\/$/, '');
  chrome.storage.local.set({ serverUrl: url }, () => {
    const status = document.getElementById('mtUrlStatus');
    status.innerText = 'URL Saved!';
    setTimeout(() => status.innerText = '', 2000);
  });
  initFontFamilyPicker(url);
  chrome.storage.local.get(['styleFontBold', 'styleFontItalic', 'styleFontRegular'], (saved) => {
    initStyleFontPickers(url, saved || {});
  });
  syncModelTypeFromServer(url);
  syncInpaintModeFromServer(url);
  syncOcrModeFromServer(url);
});

// ============================================================================
// ★ SAVE ALL SETTINGS — caches everything to chrome.storage.local
// ============================================================================
document.getElementById('saveAllBtn').addEventListener('click', () => {
  const url = document.getElementById('optServerUrl').value.trim().replace(/\/$/, '');
  const ocrMode = document.getElementById('optOcrMode').value;
  const modelType = document.getElementById('optModelType').value;
  const openrouterModel = document.getElementById('optOpenrouterModel').value.trim();
  const openrouterApiKey = document.getElementById('optOpenrouterKey').value.trim();
  const inpaintMode = document.getElementById('optInpaintMode').value;
  const combineAmount = document.getElementById('optCombineAmount').value;
  const freeOpenRouter = document.getElementById('optFreeOpenRouter').checked;
  const openaiOcrEndpoint = document.getElementById('optOpenaiOcrEndpoint').value.trim();
  const openaiOcrModel = document.getElementById('optOpenaiOcrModel').value.trim();
  const openaiOcrApiKey = document.getElementById('optOpenaiOcrKey').value.trim();
  const googleAiOcrModelSelect = document.getElementById('optGoogleAiOcrModel').value;
  const googleAiOcrModel = googleAiOcrModelSelect === 'custom'
    ? document.getElementById('optGoogleAiOcrCustomModel').value.trim()
    : googleAiOcrModelSelect;
  const googleAiOcrApiKey = document.getElementById('optGoogleAiOcrKey').value.trim();
  const googleAiOcrRpm = parseInt(document.getElementById('optGoogleAiOcrRpm').value, 10) || 5;
  const contextMode = document.getElementById('optContextMode').value;
  const contextLevel = document.getElementById('optContextLevel').value;
  const styleFontBold = document.getElementById('optStyleFontBold').value;
  const styleFontItalic = document.getElementById('optStyleFontItalic').value;
  const styleFontRegular = document.getElementById('optStyleFontRegular').value;

  chrome.storage.local.set({
    serverUrl: url,
    ocrMode: ocrMode,
    modelType: modelType,
    openrouterModel: openrouterModel,
    openrouterApiKey: openrouterApiKey,
    inpaintMode: inpaintMode,
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
    styleFontBold: styleFontBold,
    styleFontItalic: styleFontItalic,
    styleFontRegular: styleFontRegular
  }, () => {
    const btn = document.getElementById('saveAllBtn');
    const originalText = btn.innerText;
    btn.innerText = '✓ Settings Saved & Cached!';
    setTimeout(() => { btn.innerText = originalText; }, 2000);
  });
});

document.getElementById('optModelType').addEventListener('change', (e) => {
  const isOpenRouter = e.target.value === 'openrouter';
  document.getElementById('optOpenrouterRow').style.display = isOpenRouter ? 'block' : 'none';
  chrome.storage.local.set({ modelType: e.target.value });
  applyContextLevelGate();

  if (!isOpenRouter) {
    const serverUrl = document.getElementById('optServerUrl').value.trim().replace(/\/$/, '');
    if (serverUrl) {
      fetch(`${serverUrl}/SetModelType`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ model_type: 'local' })
      }).catch(e => console.warn('[MangaTranslator] Failed to switch to local model:', e));
    }
  }
});

// ============================================================================
// SET MODEL — pushes to server AND caches API key
// ============================================================================
document.getElementById('optSetModelBtn').addEventListener('click', async () => {
  const serverUrl = document.getElementById('optServerUrl').value.trim().replace(/\/$/, '');
  const model = document.getElementById('optOpenrouterModel').value.trim();
  const apiKey = document.getElementById('optOpenrouterKey').value.trim();
  const statusEl = document.getElementById('optModelTypeStatus');

  if (!serverUrl) { alert("Please set your FastAPI Server URL first!"); return; }
  if (!model) { alert("Please enter an OpenRouter model ID."); return; }

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
      // ★ Cache model + API key
      chrome.storage.local.set({ modelType: 'openrouter', openrouterModel: model, openrouterApiKey: apiKey });
      statusEl.innerText = `Active: ${data.openrouter_model}`;
    } else {
      statusEl.innerHTML = `<span class="error">Error: ${data.detail}</span>`;
    }
  } catch (e) {
    statusEl.innerHTML = `<span class="error">Error: ${e}</span>`;
  }
});

document.getElementById('optInpaintMode').addEventListener('change', (e) => {
  const serverUrl = document.getElementById('optServerUrl').value.trim().replace(/\/$/, '');
  chrome.storage.local.set({ inpaintMode: e.target.value });
  pushInpaintMode(serverUrl, e.target.value);
});

document.getElementById('optOcrMode').addEventListener('change', (e) => {
  const serverUrl = document.getElementById('optServerUrl').value.trim().replace(/\/$/, '');
  document.getElementById('optOpenaiOcrRow').style.display = e.target.value === 'openai_endpoint' ? 'block' : 'none';
  document.getElementById('optGoogleAiOcrRow').style.display = e.target.value === 'google_ai' ? 'block' : 'none';
  chrome.storage.local.set({ ocrMode: e.target.value });
  if (e.target.value === 'openai_endpoint' || e.target.value === 'google_ai') {
    const payload = { cloud_mode: false, ocr_mode: e.target.value };
    if (e.target.value === 'openai_endpoint') {
      payload.openai_ocr_endpoint = document.getElementById('optOpenaiOcrEndpoint').value.trim();
      payload.openai_ocr_model = document.getElementById('optOpenaiOcrModel').value.trim();
      payload.openai_ocr_api_key = document.getElementById('optOpenaiOcrKey').value.trim();
    } else {
      const selectedModel = document.getElementById('optGoogleAiOcrModel').value;
      payload.google_ai_ocr_api_key = document.getElementById('optGoogleAiOcrKey').value.trim();
      payload.google_ai_ocr_model = selectedModel === 'custom'
        ? document.getElementById('optGoogleAiOcrCustomModel').value.trim()
        : selectedModel;
      payload.google_ai_ocr_rpm = parseInt(document.getElementById('optGoogleAiOcrRpm').value, 10) || 5;
    }
    fetch(`${serverUrl}/SetAllSettings`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    }).catch(err => console.warn('[MangaTranslator] Failed to configure cloud OCR:', err));
  } else {
    pushOcrMode(serverUrl, e.target.value);
  }
});

document.getElementById('optGoogleAiOcrModel').addEventListener('change', (e) => {
  document.getElementById('optGoogleAiOcrCustomModel').style.display = e.target.value === 'custom' ? 'block' : 'none';
});

// ============================================================================
// GGUF MODEL LIST / SWITCH / INSTALL
// ============================================================================
document.getElementById('refreshModelsBtn').addEventListener('click', async () => {
  const serverUrl = document.getElementById('optServerUrl').value.trim().replace(/\/$/, '');
  const tableBody = document.querySelector('#modelsTable tbody');
  tableBody.innerHTML = '<tr><td colspan="5" style="text-align:center;">Loading...</td></tr>';

  try {
    const res = await fetch(`${serverUrl}/v1/listmodels`);
    const data = await res.json();
    tableBody.innerHTML = '';

    if (data.models && data.models.length > 0) {
      data.models.forEach(m => {
        const tr = document.createElement('tr');
        tr.innerHTML = `
          <td>${m.repo_id}</td>
          <td>${m.filename}</td>
          <td><span class="vision-pill${m.vision_capable ? '' : ' text-only'}">${m.vision_capable ? 'Vision OCR' : 'Text only'}</span></td>
          <td>${m.size_mb}</td>
          <td><button class="success" data-repo="${m.repo_id}" data-file="${m.filename}">Switch</button></td>
        `;
        tableBody.appendChild(tr);
      });

      document.querySelectorAll('#modelsTable button.success').forEach(btn => {
        btn.addEventListener('click', async (e) => {
          const repo = e.target.dataset.repo;
          const file = e.target.dataset.file;
          document.getElementById('modelStatus').innerText = `Switching to ${repo}/${file}...`;
          try {
            const res = await fetch(`${serverUrl}/v1/changemodel`, {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify({ repo_id: repo, filename: file })
            });
            const data = await res.json();
            if (res.ok) {
              document.getElementById('modelStatus').innerText = `Active: ${data.repo_id}/${data.filename}`;
              document.getElementById('optModelType').value = 'local';
              document.getElementById('optOpenrouterRow').style.display = 'none';
              document.getElementById('optInpaintMode').value = 'low';
              chrome.storage.local.set({ modelType: 'local', cloudMode: false, inpaintMode: 'low' });
            } else {
              document.getElementById('modelStatus').innerHTML = `<span class="error">Error: ${data.detail}</span>`;
            }
          } catch (err) {
            document.getElementById('modelStatus').innerHTML = `<span class="error">Error: ${err}</span>`;
          }
        });
      });
    } else {
      tableBody.innerHTML = '<tr><td colspan="5" style="text-align:center;">No models found in API server.</td></tr>';
    }
  } catch (e) {
    tableBody.innerHTML = `<tr><td colspan="5" class="error" style="text-align:center;">Error fetching models: ${e}</td></tr>`;
  }
});

document.getElementById('installModelBtn').addEventListener('click', async () => {
  const serverUrl = document.getElementById('optServerUrl').value.trim().replace(/\/$/, '');
  const repo = document.getElementById('customRepo').value.trim();
  const file = document.getElementById('customFile').value.trim();

  if (!repo) {
    alert("Please enter a Repo ID.");
    return;
  }

  document.getElementById('modelStatus').innerText = `Downloading & switching to ${repo}/${file || 'auto'}... (This may take a while)`;
  try {
    const res = await fetch(`${serverUrl}/v1/changemodel`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ repo_id: repo, filename: file || null })
    });
    const data = await res.json();
    if (res.ok) {
      document.getElementById('modelStatus').innerText = `Success! Active model: ${data.repo_id}/${data.filename}`;
      document.getElementById('optModelType').value = 'local';
      document.getElementById('optOpenrouterRow').style.display = 'none';
      chrome.storage.local.set({ modelType: 'local', cloudMode: false });
      document.getElementById('refreshModelsBtn').click();
    } else {
      document.getElementById('modelStatus').innerHTML = `<span class="error">Error: ${data.detail}</span>`;
    }
  } catch (err) {
    document.getElementById('modelStatus').innerHTML = `<span class="error">Error: ${err}</span>`;
  }
});