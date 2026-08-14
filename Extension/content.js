(function() {
  let isTranslating = false;
  let floatBtn, floatPopup;

  // ========================================================================
  // FONT PREVIEW HELPERS
  // ========================================================================
  let _mtFontFace = null;
  let _mtFontFaceServerUrl = null;

  async function loadServerFontFace(serverUrl) {
    if (_mtFontFace && _mtFontFaceServerUrl === serverUrl) return _mtFontFace;
    const res = await fetch(`${serverUrl}/v1/font`);
    if (!res.ok) throw new Error(`font fetch failed: HTTP ${res.status}`);
    const buf = await res.arrayBuffer();
    const face = new FontFace('MTPreviewFont', buf);
    await face.load();
    document.fonts.add(face);
    _mtFontFace = face;
    _mtFontFaceServerUrl = serverUrl;
    return face;
  }

  // ========================================================================
  // FONT FAMILY PICKER (in-page popup)
  // ========================================================================
  const _mtFontFamilyCache = new Map();

  function _mtFilenameFromPath(p) { return (p || '').split(/[\\/]/).pop(); }

  async function loadFontFaceByName(serverUrl, filename) {
    if (!serverUrl || !filename) return null;
    const cacheKey = `${serverUrl}::${filename}`;
    const family = `MTFamily_${filename.replace(/[^a-zA-Z0-9]/g, '_')}`;
    if (_mtFontFamilyCache.has(cacheKey)) return family;
    try {
      const res = await fetch(`${serverUrl}/v1/font/${encodeURIComponent(filename)}`);
      if (!res.ok) return null;
      const buf = await res.arrayBuffer();
      const face = new FontFace(family, buf);
      await face.load();
      document.fonts.add(face);
      _mtFontFamilyCache.set(cacheKey, face);
      return family;
    } catch (e) {
      console.warn(`[MangaTranslator] Could not load font preview for ${filename}:`, e);
      return null;
    }
  }

  async function initFontFamilyPicker(serverUrl) {
    const container = document.getElementById('mtFontFamilyScroll');
    if (!container) return;

    if (!serverUrl) {
      container.innerHTML = '<div style="font-size:11px;color:#888;">Set a Server URL in Advanced Settings first</div>';
      return;
    }
    container.innerHTML = '<div style="font-size:11px;color:#888;">Loading fonts…</div>';

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
      activeFilename = _mtFilenameFromPath(activeData.font_path);
    } catch (e) {
      container.innerHTML = `<div style="font-size:11px;color:#ff4d4d;">Could not load fonts: ${e}</div>`;
      return;
    }

    container.innerHTML = '';
    if (fonts.length === 0) {
      container.innerHTML = '<div style="font-size:11px;color:#888;">No fonts found in server fonts folder.</div>';
      return;
    }

    fonts.forEach(f => {
      const chip = document.createElement('button');
      chip.type = 'button';
      chip.dataset.filename = f.filename;
      chip.textContent = f.name;
      chip.title = `${f.filename} (${f.size_kb} KB)`;
      const isActive = f.filename === activeFilename;
      chip.style.cssText = `
        flex: 0 0 auto; padding: 6px 11px; border-radius: 14px; cursor: pointer;
        font-size: 12px; white-space: nowrap; color: ${isActive ? '#fff' : '#ccc'};
        background: #2a2a3c; border: 2px solid ${isActive ? '#28a745' : '#555'};
      `;
      chip.onclick = () => selectFontFamily(serverUrl, f.filename, container);
      container.appendChild(chip);

      loadFontFaceByName(serverUrl, f.filename).then(family => {
        if (family) chip.style.fontFamily = `"${family}", sans-serif`;
      });
    });
  }

  async function selectFontFamily(serverUrl, filename, container) {
    const statusEl = document.getElementById('mtFontFamilyStatus');
    if (statusEl) statusEl.innerText = `Switching to ${filename}...`;
    try {
      const res = await fetch(`${serverUrl}/SetFont`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ font_name: filename })
      });
      const data = await res.json();
      if (!res.ok) {
        if (statusEl) statusEl.innerHTML = `<span style="color:#ff4d4d;">Error: ${data.detail}</span>`;
        return;
      }
      if (statusEl) statusEl.innerText = `Active: ${filename}`;
      chrome.storage.local.set({ fontFamily: filename });
      container.querySelectorAll('button').forEach(c => {
        const on = c.dataset.filename === filename;
        c.style.borderColor = on ? '#28a745' : '#555';
        c.style.color = on ? '#fff' : '#ccc';
      });
    } catch (e) {
      if (statusEl) statusEl.innerHTML = `<span style="color:#ff4d4d;">Error: ${e}</span>`;
    }
  }

  // Listen for messages from popup
  chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
    if (message.action === "translateAllImages") {
      if (isTranslating) {
        sendResponse({ ok: false, error: 'Translation is already running on this page.' });
        return;
      }
      const imageCount = findAllTranslatableImages().length;
      if (imageCount === 0) {
        sendResponse({
          ok: false,
          error: 'No eligible manga images found. Images must be visible, at least 200x200 on the page, and at least 700,000 source pixels.',
        });
        return;
      }
      // Popup forwards combineAmount + contextMode/contextLevel/styleFonts; fall back to cached.
      const opts = {
        combineAmount: parseInt(message.combineAmount, 10) || 1,
        contextMode: message.contextMode,
        contextLevel: message.contextLevel,
        styleFonts: message.styleFonts || null,
      };
      startTranslationProcess(message.ocrLang, message.targetLang, opts)
        .catch((error) => console.error('[MangaTranslator] Translation failed to start:', error));
      sendResponse({ ok: true, started: true, imageCount });
    }
  });

  // Inject keyframes for spinner
  const style = document.createElement('style');
  style.innerHTML = `
    @keyframes mt-spin {
      0% { transform: rotate(0deg); }
      100% { transform: rotate(360deg); }
    }
  `;
  document.head.appendChild(style);

  // ========================================================================
  // ★ INPAINTING & OCR MODE SYNC HELPERS — 1:1 with popup.js
  // ========================================================================
  async function syncInpaintModeFromServer(serverUrl) {
    const statusEl = document.getElementById('mtInpaintModeStatus');
    if (!serverUrl) return;
    try {
      const res = await fetch(`${serverUrl}/GetInpaintMode`);
      const data = await res.json();
      const el = document.getElementById('mtInpaintMode');
      if (el) el.value = data.inpaint_mode || 'low';
      chrome.storage.local.set({ inpaintMode: data.inpaint_mode || 'low' });
      if (statusEl) {
        if (data.inpaint_mode === 'high') {
          statusEl.innerText = data.high_model_downloaded
            ? `High model ready (${data.high_model_size_mb} MB)`
            : 'High model will download on first use';
        } else if (data.inpaint_mode === 'none') {
          statusEl.innerText = 'None mode active (no model loaded)';
        } else {
          statusEl.innerText = '';
        }
      }
    } catch (e) {
      console.warn('[MangaTranslator] Could not fetch inpaint mode from server:', e);
    }
  }

  async function pushInpaintMode(serverUrl, mode) {
    const statusEl = document.getElementById('mtInpaintModeStatus');
    if (!serverUrl) {
      if (statusEl) statusEl.innerHTML = `<span style="color:#ff4d4d;">Set a Server URL first</span>`;
      return;
    }
    if (statusEl) statusEl.innerText = mode === 'high' ? 'Switching (may download model)...' : 'Switching...';
    try {
      const res = await fetch(`${serverUrl}/SetInpaintMode`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ mode })
      });
      const data = await res.json();
      if (res.ok) {
        if (statusEl) {
          if (data.inpaint_mode === 'high') {
            statusEl.innerText = `High model ready (${data.high_model_size_mb} MB)`;
          } else if (data.inpaint_mode === 'none') {
            statusEl.innerText = 'None mode active (no model loaded)';
          } else {
            statusEl.innerText = 'Low mode active';
          }
        }
      } else {
        if (statusEl) statusEl.innerHTML = `<span style="color:#ff4d4d;">Error: ${data.detail}</span>`;
      }
    } catch (e) {
      if (statusEl) statusEl.innerHTML = `<span style="color:#ff4d4d;">Error: ${e}</span>`;
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
    const statusEl = document.getElementById('mtOcrModeStatus');
    if (!serverUrl) return;
    try {
      const res = await fetch(`${serverUrl}/GetOcrMode`);
      const data = await res.json();
      const el = document.getElementById('mtOcrModeSelect');
      if (el) el.value = data.ocr_mode || 'hayai';
      chrome.storage.local.set({ ocrMode: data.ocr_mode || 'hayai' });
      if (statusEl) {
        statusEl.innerText = ocrModeLabel(data.ocr_mode);
      }
    } catch (e) {
      console.warn('[MangaTranslator] Could not fetch OCR mode from server:', e);
    }
  }

  async function pushOcrMode(serverUrl, mode) {
    const statusEl = document.getElementById('mtOcrModeStatus');
    if (!serverUrl) {
      if (statusEl) statusEl.innerHTML = `<span style="color:#ff4d4d;">Set a Server URL first</span>`;
      return;
    }
    if (statusEl) statusEl.innerText = 'Switching...';
    try {
      const res = await fetch(`${serverUrl}/SetOcrMode`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ mode })
      });
      const data = await res.json();
      if (res.ok) {
        if (statusEl) statusEl.innerText = ocrModeLabel(data.ocr_mode);
        chrome.storage.local.set({ ocrMode: data.ocr_mode });
      } else {
        if (statusEl) statusEl.innerHTML = `<span style="color:#ff4d4d;">Error: ${data.detail}</span>`;
      }
    } catch (e) {
      if (statusEl) statusEl.innerHTML = `<span style="color:#ff4d4d;">Error: ${e}</span>`;
    }
  }

  // ========================================================================
  // ★ CLOUD MODE HELPERS — 1:1 with popup.js
  // ========================================================================
  function applyCloudModeToPopup(on) {
    const ocrModeEl   = document.getElementById('mtOcrModeSelect');
    const modelTypeEl = document.getElementById('mtModelTypeSelect');
    const inpaintEl   = document.getElementById('mtInpaintMode');
    const colorizeEl  = document.getElementById('mtColorize');
    const orRow       = document.getElementById('mtOpenrouterRow');

    if (on) {
      if (ocrModeEl) ocrModeEl.value = 'lens';
      if (modelTypeEl) modelTypeEl.value = 'openrouter';
      if (inpaintEl) inpaintEl.value = 'none';
      if (colorizeEl) colorizeEl.checked = false;
      if (orRow) orRow.style.display = 'block';
    }
    // Lock the local-resource controls while cloud mode is active.
    [ocrModeEl, inpaintEl, colorizeEl].forEach(el => { if (el) el.disabled = on; });
    if (modelTypeEl) modelTypeEl.disabled = on;
  }

  async function pushCloudModeFromPopup(serverUrl) {
    if (!serverUrl) return;
    const statusEl = document.getElementById('mtStatus');
    const liveModel = document.getElementById('mtOpenrouterModel').value.trim();
    const liveKey = document.getElementById('mtOpenrouterKey').value.trim();
    const cached = await chrome.storage.local.get(['openrouterModel', 'openrouterApiKey']);
    const model = liveModel || cached.openrouterModel || '';
    const apiKey = liveKey || cached.openrouterApiKey || '';
    try {
      const body = { enabled: true };
      if (model) body.model = model;
      if (apiKey) body.api_key = apiKey;
      const res = await fetch(`${serverUrl}/SetCloudMode`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body)
      });
      const data = await res.json();
      if (!res.ok) { if (statusEl) statusEl.innerText = data.detail || 'Cloud mode failed to enable.'; return; }
      chrome.storage.local.set({
        modelType: 'openrouter', ocrMode: 'lens', inpaintMode: 'none', colorize: false,
        openrouterModel: data.openrouter_model || model, openrouterApiKey: apiKey || cached.openrouterApiKey
      });
      if (statusEl) statusEl.innerText = `Cloud mode on — Lens + ${data.openrouter_model || 'OpenRouter'}`;
    } catch (e) {
      if (statusEl) statusEl.innerText = 'Could not reach server to enable cloud mode.';
    }
  }

  async function disableCloudModeOnServer(serverUrl) {
    if (!serverUrl) return;
    try {
      await fetch(`${serverUrl}/SetCloudMode`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ enabled: false })
      });
    } catch (e) {
      console.warn('[MangaTranslator] Failed to disable cloud mode on server:', e);
    }
  }

  // ========================================================================
  // Unified settings push — single POST to /SetAllSettings before Translate.
  // Mirrors popup.js. Handles the cloud-mode-restart problem: if the backend
  // was restarted (in-memory settings reset) while cloud mode was on in the
  // extension, this re-applies the correct state in one call.
  // ========================================================================
  async function pushAllSettings(serverUrl, settings) {
    if (!serverUrl) return;
    try {
      const res = await fetch(`${serverUrl}/SetAllSettings`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(settings),
      });
      const data = await res.json();
      if (!res.ok) {
        console.warn('[MangaTranslator] /SetAllSettings failed:', data.detail || res.status);
      } else {
        console.log('[MangaTranslator] Settings applied:', data.applied);
      }
    } catch (e) {
      console.warn('[MangaTranslator] /SetAllSettings error:', e);
    }
  }

  // Build the full settings payload from the floating popup's current values.
  // In cloud mode this forces ocr=lens / model_type=openrouter / inpaint=none.
  function buildSettingsPayload() {
    const cloudMode = document.getElementById('mtCloudMode').checked;
    const ctxOn = document.getElementById('mtContextMode')?.value === 'on';
    const ctxLevel = document.getElementById('mtContextLevel')?.value === 'high' ? 'high' : 'low';
    const payload = {
      cloud_mode: cloudMode,
      free_openrouter: document.getElementById('mtFreeOpenRouter')?.checked || false,
      context_aware: ctxOn,
      context_level: ctxLevel,
      style_aware: ctxOn && ctxLevel === 'high',
      style_fonts: {
        bold: document.getElementById('mtStyleFontBold')?.value || '',
        italic: document.getElementById('mtStyleFontItalic')?.value || '',
        regular: document.getElementById('mtStyleFontRegular')?.value || '',
      },
    };

    if (cloudMode) {
      payload.model_type = 'openrouter';
      payload.ocr_mode = 'lens';
      payload.inpaint_mode = 'none';
      const model = document.getElementById('mtOpenrouterModel').value.trim();
      const apiKey = document.getElementById('mtOpenrouterKey').value.trim();
      if (model) payload.openrouter_model = model;
      if (apiKey) payload.openrouter_api_key = apiKey;
    } else {
      payload.ocr_mode = document.getElementById('mtOcrModeSelect').value;
      payload.inpaint_mode = document.getElementById('mtInpaintMode').value;
      payload.model_type = document.getElementById('mtModelTypeSelect').value;
      const model = document.getElementById('mtOpenrouterModel').value.trim();
      const apiKey = document.getElementById('mtOpenrouterKey').value.trim();
      if (model) payload.openrouter_model = model;
      if (apiKey) payload.openrouter_api_key = apiKey;
      // Active font filename, if known (chip with green border)
      const activeChip = Array.from(document.querySelectorAll('#mtFontFamilyScroll button[data-filename]'))
        .find(c => c.style.borderColor === 'rgb(40, 167, 69)' || c.style.borderColor === '#28a745');
      if (activeChip && activeChip.dataset.filename) {
        payload.font_filename = activeChip.dataset.filename;
      }
    }
    return payload;
  }

  // ========================================================================
  // Helper: load all cached settings into the floating popup — 1:1 with popup.js
  // ========================================================================
  async function loadCachedSettingsIntoPopup() {
    const sel = document.getElementById('mtTargetLangSelect');
    const ocrLangSel = document.getElementById('mtOcrLangSelect');

    const data = await chrome.storage.local.get(
      ['serverUrl', 'ocrLang', 'colorize', 'targetLang', 'modelType',
       'openrouterModel', 'openrouterApiKey', 'inpaintMode', 'ocrMode', 'cloudMode',
       'combineAmount', 'freeOpenRouter', 'contextMode', 'contextLevel',
       'styleFontBold', 'styleFontItalic', 'styleFontRegular', 'skipSfx', 'contextAware']
    );

    // Populate language dropdowns from built-in list, then refresh from server
    if (typeof mtPopulateLangSelect === 'function') {
      mtPopulateLangSelect(sel, data.targetLang || 'en');
      mtPopulateLangSelect(ocrLangSel, data.ocrLang || 'ja');
      if (data.serverUrl && typeof mtFetchLanguages === 'function') {
        const langs = await mtFetchLanguages(data.serverUrl);
        mtPopulateLangSelect(sel, sel.value || data.targetLang || 'en', langs);
        mtPopulateLangSelect(ocrLangSel, ocrLangSel.value || data.ocrLang || 'ja', langs);
      }
    }
    if (sel && data.targetLang) sel.value = data.targetLang;

    // ★ Colorize checkbox — 1:1 with popup.js
    const colorizeEl = document.getElementById('mtColorize');
    if (colorizeEl) colorizeEl.checked = data.colorize !== false;

    // ★ Inpaint mode — 1:1 with popup.js
    const inpaintEl = document.getElementById('mtInpaintMode');
    if (inpaintEl) inpaintEl.value = data.inpaintMode || 'low';

    // ★ OCR mode — 1:1 with popup.js
    const ocrModeSel = document.getElementById('mtOcrModeSelect');
    if (ocrModeSel) ocrModeSel.value = data.ocrMode || 'hayai';
    if (ocrLangSel && data.ocrLang) ocrLangSel.value = data.ocrLang;

    // Model type + OpenRouter fields
    const modelTypeSel = document.getElementById('mtModelTypeSelect');
    const openrouterRow = document.getElementById('mtOpenrouterRow');
    if (modelTypeSel) {
      modelTypeSel.value = data.modelType || 'local';
      if (openrouterRow) openrouterRow.style.display = modelTypeSel.value === 'openrouter' ? 'block' : 'none';
    }
    if (data.openrouterModel) {
      const orModelInput = document.getElementById('mtOpenrouterModel');
      if (orModelInput) orModelInput.value = data.openrouterModel;
    }
    if (data.openrouterApiKey) {
      const orKeyInput = document.getElementById('mtOpenrouterKey');
      if (orKeyInput) orKeyInput.value = data.openrouterApiKey;
    }

    // ★ Cloud Mode — restore toggle + apply locked state — 1:1 with popup.js
    const cloudOn = data.cloudMode === true;
    const cloudEl = document.getElementById('mtCloudMode');
    if (cloudEl) {
      cloudEl.checked = cloudOn;
      applyCloudModeToPopup(cloudOn);
    }

    // ★ Combine slider — restore value + live label — 1:1 with popup.js
    const combineAmount = parseInt(data.combineAmount || '1', 10);
    const combineSlider = document.getElementById('mtCombineAmount');
    const combineLabel = document.getElementById('mtCombineAmountVal');
    if (combineSlider && combineLabel) {
      combineSlider.value = combineAmount;
      combineLabel.textContent = combineAmount;
    }

    // ★ Context (merged Skip SFX + Context Aware) — restore + level/style-font gating
    const ctxModeEl = document.getElementById('mtContextMode');
    const ctxLevelEl = document.getElementById('mtContextLevel');
    const ctxLevelRow = document.getElementById('mtContextLevelRow');
    const styleFontsBtn = document.getElementById('mtStyleFontsBtn');
    const styleFontsBox = document.getElementById('mtStyleFontsBox');
    if (ctxModeEl && ctxLevelEl) {
      const ctxMode = data.contextMode
        || ((data.skipSfx === true || data.contextAware === true) ? 'on' : 'off');
      const ctxLevel = data.contextLevel === 'high' ? 'high' : 'low';
      ctxModeEl.value = ctxMode;
      ctxLevelEl.value = ctxLevel;
      const applyCtxVisibility = () => {
        const on = ctxModeEl.value === 'on';
        const high = on && ctxLevelEl.value === 'high';
        if (ctxLevelRow) ctxLevelRow.style.display = on ? 'block' : 'none';
        if (styleFontsBtn) styleFontsBtn.style.display = high ? 'block' : 'none';
        if (styleFontsBox && !high) styleFontsBox.style.display = 'none';
      };
      applyCtxVisibility();
      ctxModeEl.onchange = () => {
        chrome.storage.local.set({ contextMode: ctxModeEl.value });
        applyCtxVisibility();
      };
      ctxLevelEl.onchange = () => {
        chrome.storage.local.set({ contextLevel: ctxLevelEl.value });
        applyCtxVisibility();
      };
      if (styleFontsBtn && styleFontsBox) {
        styleFontsBtn.onclick = () => {
          styleFontsBox.style.display = styleFontsBox.style.display === 'block' ? 'none' : 'block';
        };
      }
    }

    // ★ Free OpenRouter — restore (default off) — 1:1 with popup.js
    const freeOrEl = document.getElementById('mtFreeOpenRouter');
    if (freeOrEl) freeOrEl.checked = data.freeOpenRouter === true;

    // ★ Sync from server (best-effort) — 1:1 with popup.js
    if (data.serverUrl) {
      syncInpaintModeFromServer(data.serverUrl);
      syncOcrModeFromServer(data.serverUrl);
    }
  }

  function injectUI() {
    // 1. Floating Button
    floatBtn = document.createElement('button');
    floatBtn.title = 'Manga Translator';
    floatBtn.setAttribute('aria-label', 'Manga Translator');
    floatBtn.innerHTML = `
      <svg viewBox="0 0 24 24" width="24" height="24" fill="none" stroke="#ffffff"
           stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
        <path d="m5 8 6 6"/><path d="m4 14 6-6 2-3"/><path d="M2 5h12"/>
        <path d="M7 2h1"/><path d="m22 22-5-10-5 10"/><path d="M14 18h6"/>
      </svg>`;
    floatBtn.style.cssText = `
      position: fixed; top: 50%; left: 15px; transform: translateY(-50%);
      z-index: 2147483647; width: 46px; height: 46px;
      background: rgba(20,20,31,0.35); color: #fff;
      border: 1px solid rgba(255,255,255,0.25); border-radius: 12px;
      cursor: pointer; box-shadow: 0 2px 10px rgba(0,0,0,0.25);
      display: flex; align-items: center; justify-content: center;
      backdrop-filter: blur(6px); -webkit-backdrop-filter: blur(6px);
      opacity: 0.55; transition: opacity .15s, background .15s, border-color .15s;
      padding: 0;
    `;
    floatBtn.onmouseover = () => {
      floatBtn.style.opacity = '1';
      floatBtn.style.background = 'rgba(34,165,82,0.85)';
      floatBtn.style.borderColor = 'rgba(255,255,255,0.55)';
    };
    floatBtn.onmouseout = () => {
      floatBtn.style.opacity = '0.55';
      floatBtn.style.background = 'rgba(20,20,31,0.35)';
      floatBtn.style.borderColor = 'rgba(255,255,255,0.25)';
    };
    floatBtn.onclick = (e) => { e.stopPropagation(); toggleFloatPopup(); };
    document.body.appendChild(floatBtn);

    // 2. Popup Menu
    floatPopup = document.createElement('div');
    floatPopup.style.cssText = `
      position: fixed; top: 50%; left: 78px; transform: translateY(-50%);
      z-index: 2147483647; padding: 16px; background: #16161f;
      border-radius: 12px; box-shadow: 0 8px 28px rgba(0,0,0,0.55);
      font-family: 'Segoe UI', Arial, sans-serif; display: none; width: 300px;
      color: #e6e6ec; border: 1px solid #2c2c3a; font-size: 14px; line-height: 1.4;
      max-height: 88vh; overflow-y: auto; overflow-x: hidden; box-sizing: border-box;
    `;
    const ICON = {
      arrow: '<svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align:middle;margin-right:8px;"><path d="M5 12h14"/><path d="m12 5 7 7-7 7"/></svg>',
      globe: '<svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align:middle;margin-right:6px;"><circle cx="12" cy="12" r="10"/><path d="M2 12h20"/><path d="M12 2a15.3 15.3 0 0 1 4 10 15.3 15.3 0 0 1-4 10 15.3 15.3 0 0 1-4-10 15.3 15.3 0 0 1 4-10z"/></svg>',
      cloud: '<svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align:middle;margin-right:6px;"><path d="M17.5 19a4.5 4.5 0 1 0 0-9h-1.8A7 7 0 1 0 4 15.9"/></svg>',
      scan: '<svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align:middle;margin-right:6px;"><path d="M4 7V5a2 2 0 0 1 2-2h2"/><path d="M4 17v2a2 2 0 0 0 2 2h2"/><path d="M16 3h2a2 2 0 0 1 2 2v2"/><path d="M16 21h2a2 2 0 0 0 2-2v-2"/><path d="M7 12h10"/></svg>',
      type: '<svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align:middle;margin-right:6px;"><path d="M4 20h16"/><path d="m6 16 6-12 6 12"/><path d="M8 12h8"/></svg>',
      brain: '<svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align:middle;margin-right:6px;"><path d="M12 2a3 3 0 0 0-3 3v7.5"/><path d="M12 2a3 3 0 0 1 3 3v.5"/><path d="M9 12.5A3 3 0 1 0 6 17a3 3 0 0 0 3 1"/><path d="M15 6a3 3 0 1 1 3 5"/><path d="M9 18a3 3 0 0 0 6 0v-6"/><path d="M18 11a3 3 0 1 1-3 5"/></svg>',
      gear: '<svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align:middle;margin-right:8px;"><circle cx="12" cy="12" r="3"/><path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 1 1-2.83 2.83l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-4 0v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 1 1-2.83-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1 0-4h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 1 1 2.83-2.83l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 4 0v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 1 1 2.83 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 0 4h-.09a1.65 1.65 0 0 0-1.51 1z"/></svg>',
      wrench: '<svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align:middle;margin-right:8px;"><path d="M14.7 6.3a1 1 0 0 0 0 1.4l1.6 1.6a1 1 0 0 0 1.4 0l3.77-3.77a6 6 0 0 1-7.94 7.94l-6.91 6.91a2.12 2.12 0 0 1-3-3l6.91-6.91a6 6 0 0 1 7.94-7.94l-3.76 3.76z"/></svg>',
      inpaint: '<svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align:middle;margin-right:6px;"><path d="m15 5 4 4"/><path d="M13 7 8.7 2.7a2.4 2.4 0 0 0-3.4 0L2.7 5.3a2.4 2.4 0 0 0 0 3.4L7 13"/><path d="m8 6 2-2"/><path d="M18 12h.01"/><path d="M18 21a3 3 0 0 0 3-3c0-1.5-1-3-3-5-2 2-3 3.5-3 5a3 3 0 0 0 3 3z"/><path d="M11 15 6 20a2.83 2.83 0 0 1-4-4l5-5"/></svg>',
      palette: '<svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align:middle;margin-right:6px;"><circle cx="13.5" cy="6.5" r=".5" fill="currentColor"/><circle cx="17.5" cy="10.5" r=".5" fill="currentColor"/><circle cx="8.5" cy="7.5" r=".5" fill="currentColor"/><circle cx="6.5" cy="12.5" r=".5" fill="currentColor"/><path d="M12 2C6.5 2 2 6.5 2 12s4.5 10 10 10c.926 0 1.648-.746 1.648-1.688 0-.437-.18-.835-.437-1.125-.29-.289-.438-.652-.438-1.125a1.64 1.64 0 0 1 1.668-1.668h1.996c3.051 0 5.563-2.512 5.563-5.563C22 6.012 17.5 2 12 2z"/></svg>',
    };
    const selCss = "width:100%; padding:9px; margin:0 0 12px 0; box-sizing:border-box; border:1px solid #3a3a4c; border-radius:6px; background:#22222e; color:#e6e6ec; font-size:13px; cursor:pointer;";
    const inCss  = "width:100%; padding:9px; margin:6px 0; box-sizing:border-box; border:1px solid #3a3a4c; border-radius:6px; background:#22222e; color:#e6e6ec; font-size:13px;";
    const labCss = "display:flex; align-items:center; font-weight:600; color:#a9a9bd; margin-bottom:4px; font-size:13px;";
    const secCss = "display:flex; align-items:center; font-size:12px; text-transform:uppercase; letter-spacing:.5px; color:#7a7a8c; margin:6px 0 8px 0;";
    const hintCss = "font-size:11px; color:#8a8a9c; margin:-6px 0 12px 0;";
    floatPopup.innerHTML = `
      <div style="display:flex; align-items:center; gap:8px; margin-bottom:14px;">
        <svg viewBox="0 0 24 24" width="20" height="20" fill="none" stroke="#22a552" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="m5 8 6 6"/><path d="m4 14 6-6 2-3"/><path d="M2 5h12"/><path d="M7 2h1"/><path d="m22 22-5-10-5 10"/><path d="M14 18h6"/></svg>
        <div style="flex:1; font-weight:700; color:#fff; font-size:16px;">Manga Translator</div>
        <button id="mtReaderBtn" title="Open Reader" aria-label="Open Reader" style="width:34px; height:34px; padding:0; flex:0 0 auto; background:#2a2a3c; border:1px solid #444; border-radius:8px; color:#fff; cursor:pointer; display:flex; align-items:center; justify-content:center;">
          <svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M2 3h6a4 4 0 0 1 4 4v14a3 3 0 0 0-3-3H2z"/><path d="M22 3h-6a4 4 0 0 0-4 4v14a3 3 0 0 1 3-3h7z"/></svg>
        </button>
      </div>

      <button id="mtStartBtn" style="
        width:100%; box-sizing:border-box; padding:15px; background:#22a552; color:#fff; border:none;
        border-radius:8px; cursor:pointer; font-weight:600; font-size:16px; margin-bottom:12px;
        box-shadow:0 3px 10px rgba(34,165,82,0.35); display:flex; align-items:center; justify-content:center;
      ">${ICON.arrow}<span>Translate All</span></button>

      <div id="mtStatus" style="font-size:12px; color:#46c877; text-align:center; min-height:15px; margin-bottom:4px;"></div>
    `;
    document.body.appendChild(floatPopup);

    // ── Wire the Translate All button (reads everything from cache) ──
    wireTranslateButton();

    // ── Wire the Reader button (book icon) ──
    const readerBtn = document.getElementById('mtReaderBtn');
    if (readerBtn) {
      readerBtn.onclick = () => {
        if (typeof window.__mtOpenReader === 'function') {
          window.__mtOpenReader();
        } else {
          console.warn('[MangaTranslator] Reader not loaded; reloading page may help.');
        }
      };
    }
  }

  // ========================================================================
  // Translate button — reads ALL settings from chrome.storage.local
  // (the floating popup no longer has settings controls; popup.html is the
  // only settings UI. This button just triggers translation using cached
  // values, and pushes them to the backend via /SetAllSettings first.)
  // ========================================================================
  function wireTranslateButton() {
    const btn = document.getElementById('mtStartBtn');
    if (!btn) return;
    btn.onclick = async () => {
      const stored = await chrome.storage.local.get([
        'serverUrl', 'ocrMode', 'ocrLang', 'targetLang', 'inpaintMode',
        'modelType', 'cloudMode', 'colorize',
        'combineAmount', 'freeOpenRouter', 'contextMode', 'contextLevel',
        'styleFontBold', 'styleFontItalic', 'styleFontRegular', 'skipSfx', 'contextAware',
        'openrouterModel', 'openrouterApiKey',
        'openaiOcrEndpoint', 'openaiOcrModel', 'openaiOcrApiKey',
        'googleAiOcrApiKey', 'googleAiOcrModel', 'googleAiOcrRpm', 'fontFamily',
      ]);

      const ctxOn = (stored.contextMode
        || ((stored.skipSfx === true || stored.contextAware === true) ? 'on' : 'off')) === 'on';
      const ctxLevel = stored.contextLevel === 'high' ? 'high' : 'low';
      const styleFonts = {
        bold: stored.styleFontBold || '',
        italic: stored.styleFontItalic || '',
        regular: stored.styleFontRegular || '',
      };

      if (!stored.serverUrl) {
        alert("Please set your FastAPI Server URL in the extension popup first!");
        return;
      }

      // Unified settings push BEFORE translation.
      const payload = {
        cloud_mode: stored.cloudMode === true,
        free_openrouter: stored.freeOpenRouter === true,
        context_aware: ctxOn,
        context_level: ctxLevel,
        style_aware: ctxOn && ctxLevel === 'high',
        style_fonts: styleFonts,
      };
      if (stored.cloudMode) {
        payload.model_type = 'openrouter';
        payload.ocr_mode = 'lens';
        payload.inpaint_mode = 'none';
        if (stored.openrouterModel) payload.openrouter_model = stored.openrouterModel;
        if (stored.openrouterApiKey) payload.openrouter_api_key = stored.openrouterApiKey;
      } else {
        payload.ocr_mode = stored.ocrMode || 'hayai';
        payload.inpaint_mode = stored.inpaintMode || 'low';
        payload.model_type = stored.modelType || 'local';
        if (payload.ocr_mode === 'openai_endpoint') {
          payload.openai_ocr_endpoint = stored.openaiOcrEndpoint || 'https://api.openai.com/v1';
          payload.openai_ocr_model = stored.openaiOcrModel || 'gpt-4o-mini';
          payload.openai_ocr_api_key = stored.openaiOcrApiKey || '';
        }
        if (payload.ocr_mode === 'google_ai') {
          payload.google_ai_ocr_api_key = stored.googleAiOcrApiKey || '';
          payload.google_ai_ocr_model = stored.googleAiOcrModel || 'gemini-2.5-flash-lite';
          payload.google_ai_ocr_rpm = parseInt(stored.googleAiOcrRpm || 5, 10);
        }
        if (stored.openrouterModel) payload.openrouter_model = stored.openrouterModel;
        if (stored.openrouterApiKey) payload.openrouter_api_key = stored.openrouterApiKey;
        if (stored.fontFamily) payload.font_filename = stored.fontFamily;
      }
      await pushAllSettings(stored.serverUrl, payload);

      floatPopup.style.display = 'none';
      startTranslationProcess(stored.ocrLang || 'ja', stored.targetLang || 'en', {
        combineAmount: parseInt(stored.combineAmount || '1', 10) || 1,
        contextMode: ctxOn ? 'on' : 'off',
        contextLevel: ctxLevel,
        styleFonts,
      });
    };
  }

  function toggleFloatPopup() {
    if (floatPopup.style.display === 'block') {
      floatPopup.style.display = 'none';
    } else {
      floatPopup.style.display = 'block';
    }
  }

  // ========================================================================
  // SETTINGS MODAL (unchanged)
  // ========================================================================
  function openSettingsModal() {
    if (document.getElementById('mtSettingsOverlay')) return;

    const overlay = document.createElement('div');
    overlay.id = 'mtSettingsOverlay';
    overlay.style.cssText = `
      position: fixed; top: 0; left: 0; width: 100%; height: 100%;
      background: rgba(0,0,0,0.7); z-index: 2147483648;
      display: flex; align-items: center; justify-content: center;
    `;

    const modal = document.createElement('div');
    modal.style.cssText = `
      background: #1e1e2e; padding: 20px; border-radius: 8px;
      width: 650px; max-height: 80vh; overflow-y: auto;
      position: relative; box-shadow: 0 4px 20px rgba(0,0,0,0.5);
      font-family: Arial, sans-serif; color: #e0e0e0; border: 1px solid #444;
    `;

    const closeBtn = document.createElement('button');
    closeBtn.innerHTML = '✖';
    closeBtn.style.cssText = `
      position: absolute; top: 10px; right: 15px;
      background: transparent; border: none; color: #aaa;
      font-size: 24px; cursor: pointer; font-weight: bold; z-index: 10;
    `;
    closeBtn.onclick = () => overlay.remove();
    modal.appendChild(closeBtn);

    modal.insertAdjacentHTML('beforeend', `
      <h2 style="margin-top: 0; color: #ffffff;">Advanced Manga Translator Settings</h2>

      <div style="background: #2a2a3c; padding: 15px; border-radius: 8px; margin-bottom: 20px; border: 1px solid #333;">
        <h3 style="margin-top: 0; color: #ffffff;">API Server</h3>
        <label style="font-weight: bold; color: #aaaaaa; display: block; margin-bottom: 5px;">FastAPI Server URL:</label>
        <input type="text" id="mtOptServerUrl" placeholder="http://localhost:7860"
          style="width: 100%; padding: 8px; margin-bottom: 10px; box-sizing: border-box;
                 border: 1px solid #555; border-radius: 4px; background: #1e1e2e; color: #fff;">
        <button id="mtSaveUrlBtn"
          style="padding: 10px 15px; background: #0066cc; color: white; border: none; border-radius: 4px; cursor: pointer; font-weight: bold;">
          Save URL
        </button>
        <div id="mtUrlStatus" style="margin-top: 10px; font-size: 14px; color: #28a745;"></div>
      </div>

      <div style="background: #2a2a3c; padding: 15px; border-radius: 8px; margin-bottom: 20px; border: 1px solid #333;">
        <h3 style="margin-top: 0; color: #ffffff;">Translation GGUF Model</h3>
        <div style="display: flex; gap: 10px; align-items: center; margin-bottom: 10px; flex-wrap: wrap;">
          <button id="mtRefreshModelsBtn"
            style="padding: 10px 15px; background: #3a3f4b; color: white; border: 1px solid #555; border-radius: 4px; cursor: pointer; font-weight: bold;">
            Refresh List
          </button>
        </div>
        <table id="mtModelsTable" style="width: 100%; border-collapse: collapse; margin-top: 10px;">
          <thead>
            <tr>
              <th style="padding: 8px; border: 1px solid #444; text-align: left; background: #1e1e2e; color: #fff;">Repo ID</th>
              <th style="padding: 8px; border: 1px solid #444; text-align: left; background: #1e1e2e; color: #fff;">Filename</th>
              <th style="padding: 8px; border: 1px solid #444; text-align: left; background: #1e1e2e; color: #fff;">Capability</th>
              <th style="padding: 8px; border: 1px solid #444; text-align: left; background: #1e1e2e; color: #fff;">Size (MB)</th>
              <th style="padding: 8px; border: 1px solid #444; text-align: left; background: #1e1e2e; color: #fff;">Action</th>
            </tr>
          </thead>
          <tbody>
            <tr><td colspan="5" style="text-align:center; padding: 8px; border: 1px solid #444; color: #aaa;">Click "Refresh List" to load models...</td></tr>
          </tbody>
        </table>

        <hr style="margin: 20px 0; border: 0; border-top: 1px solid #444;">

        <h3 style="margin-top: 0; color: #ffffff;">Install Custom Model</h3>
        <label style="font-weight: bold; color: #aaaaaa; display: block; margin-bottom: 5px;">
          Repo ID (e.g. hugging-quants/Llama-3.2-1B-Instruct-GGUF):
        </label>
        <input type="text" id="mtCustomRepo" placeholder="repo_id"
          style="width: 100%; padding: 8px; margin-bottom: 10px; box-sizing: border-box;
                 border: 1px solid #555; border-radius: 4px; background: #1e1e2e; color: #fff;">
        <label style="font-weight: bold; color: #aaaaaa; display: block; margin-bottom: 5px;">
          Filename (leave blank to auto-find):
        </label>
        <input type="text" id="mtCustomFile" placeholder="filename.gguf"
          style="width: 100%; padding: 8px; margin-bottom: 10px; box-sizing: border-box;
                 border: 1px solid #555; border-radius: 4px; background: #1e1e2e; color: #fff;">
        <button id="mtInstallModelBtn"
          style="padding: 10px 15px; background: #28a745; color: white; border: none; border-radius: 4px; cursor: pointer; font-weight: bold;">
          Download &amp; Switch
        </button>
        <div id="mtModelInstallStatus" style="margin-top: 10px; font-size: 14px; color: #28a745;"></div>
      </div>
    `);

    overlay.appendChild(modal);
    document.body.appendChild(overlay);

    overlay.addEventListener('click', (e) => { if (e.target === overlay) overlay.remove(); });
    initSettingsModalLogic(modal);
  }

  function initSettingsModalLogic(modal) {
    chrome.storage.local.get(['serverUrl'], (data) => {
      modal.querySelector('#mtOptServerUrl').value = data.serverUrl || 'http://localhost:7860';
    });

    modal.querySelector('#mtSaveUrlBtn').addEventListener('click', () => {
      const url = modal.querySelector('#mtOptServerUrl').value.trim().replace(/\/$/, '');
      chrome.storage.local.set({ serverUrl: url }, () => {
        const status = modal.querySelector('#mtUrlStatus');
        status.innerText = 'URL Saved!';
        setTimeout(() => status.innerText = '', 2000);
        // Re-sync everything from the new server URL
        loadCachedSettingsIntoPopup();
        initFontFamilyPicker(url);
        syncInpaintModeFromServer(url);
        syncOcrModeFromServer(url);
      });
    });

    modal.querySelector('#mtRefreshModelsBtn').addEventListener('click', async () => {
      const serverUrl  = modal.querySelector('#mtOptServerUrl').value.trim().replace(/\/$/, '');
      const tableBody  = modal.querySelector('#mtModelsTable tbody');
      tableBody.innerHTML = '<tr><td colspan="5" style="text-align:center; padding: 8px; border: 1px solid #444; color: #aaa;">Loading...</td></tr>';
      try {
        const res  = await fetch(`${serverUrl}/v1/listmodels`);
        const data = await res.json();
        tableBody.innerHTML = '';
        if (data.models && data.models.length > 0) {
          data.models.forEach(m => {
            const tr = document.createElement('tr');
            tr.innerHTML = `
              <td style="padding: 8px; border: 1px solid #444; color: #ccc;">${m.repo_id}</td>
              <td style="padding: 8px; border: 1px solid #444; color: #ccc;">${m.filename}</td>
              <td style="padding: 8px; border: 1px solid #444;"><span style="display:inline-block;padding:2px 7px;border-radius:10px;font-size:11px;font-weight:bold;background:${m.vision_capable ? '#164d2b' : '#30303d'};color:${m.vision_capable ? '#7dffad' : '#aaa'};border:1px solid ${m.vision_capable ? '#287a47' : '#555'};">${m.vision_capable ? 'Vision OCR' : 'Text only'}</span></td>
              <td style="padding: 8px; border: 1px solid #444; color: #ccc;">${m.size_mb}</td>
              <td style="padding: 8px; border: 1px solid #444;">
                <button class="mt-switch-btn" data-repo="${m.repo_id}" data-file="${m.filename}"
                  style="padding: 5px 10px; background: #28a745; color: white; border: none; border-radius: 4px; cursor: pointer;">
                  Switch
                </button>
              </td>
            `;
            tableBody.appendChild(tr);
          });
          tableBody.querySelectorAll('.mt-switch-btn').forEach(btn => {
            btn.addEventListener('click', async (e) => {
              const repo = e.target.dataset.repo;
              const file = e.target.dataset.file;
              modal.querySelector('#mtModelInstallStatus').innerText = `Switching to ${repo}/${file}...`;
              try {
                const res  = await fetch(`${serverUrl}/v1/changemodel`, {
                  method: 'POST',
                  headers: { 'Content-Type': 'application/json' },
                  body: JSON.stringify({ repo_id: repo, filename: file }),
                });
                const data = await res.json();
                if (res.ok) {
                  modal.querySelector('#mtModelInstallStatus').innerText = `Active: ${data.repo_id}/${data.filename}`;
                  const modelType = document.getElementById('mtModelTypeSelect');
                  const cloudMode = document.getElementById('mtCloudMode');
                  if (modelType) modelType.value = 'local';
                  if (cloudMode) cloudMode.checked = false;
                  const inpaintMode = document.getElementById('mtInpaintMode');
                  if (inpaintMode) inpaintMode.value = 'low';
                  chrome.storage.local.set({ modelType: 'local', cloudMode: false, inpaintMode: 'low' });
                } else {
                  modal.querySelector('#mtModelInstallStatus').innerHTML = `<span style="color:#ff4d4d;">Error: ${data.detail}</span>`;
                }
              } catch (err) {
                modal.querySelector('#mtModelInstallStatus').innerHTML = `<span style="color:#ff4d4d;">Error: ${err}</span>`;
              }
            });
          });
        } else {
          tableBody.innerHTML = '<tr><td colspan="5" style="text-align:center; padding: 8px; border: 1px solid #444; color: #aaa;">No models found.</td></tr>';
        }
      } catch (e) {
        tableBody.innerHTML = `<tr><td colspan="5" style="text-align:center; padding: 8px; border: 1px solid #444; color:#ff4d4d;">Error: ${e}</td></tr>`;
      }
    });

    modal.querySelector('#mtInstallModelBtn').addEventListener('click', async () => {
      const serverUrl = modal.querySelector('#mtOptServerUrl').value.trim().replace(/\/$/, '');
      const repo = modal.querySelector('#mtCustomRepo').value.trim();
      const file = modal.querySelector('#mtCustomFile').value.trim();
      if (!repo) { alert("Please enter a Repo ID."); return; }
      modal.querySelector('#mtModelInstallStatus').innerText = `Downloading & switching to ${repo}/${file || 'auto'}...`;
      try {
        const res  = await fetch(`${serverUrl}/v1/changemodel`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ repo_id: repo, filename: file || null }),
        });
        const data = await res.json();
        if (res.ok) {
          modal.querySelector('#mtModelInstallStatus').innerText = `Success! Active: ${data.repo_id}/${data.filename}`;
          const modelType = document.getElementById('mtModelTypeSelect');
          const cloudMode = document.getElementById('mtCloudMode');
          if (modelType) modelType.value = 'local';
          if (cloudMode) cloudMode.checked = false;
          const inpaintMode = document.getElementById('mtInpaintMode');
          if (inpaintMode) inpaintMode.value = 'low';
          chrome.storage.local.set({ modelType: 'local', cloudMode: false, inpaintMode: 'low' });
          modal.querySelector('#mtRefreshModelsBtn').click();
        } else {
          modal.querySelector('#mtModelInstallStatus').innerHTML = `<span style="color:#ff4d4d;">Error: ${data.detail}</span>`;
        }
      } catch (err) {
        modal.querySelector('#mtModelInstallStatus').innerHTML = `<span style="color:#ff4d4d;">Error: ${err}</span>`;
      }
    });
  }

  injectUI();

  // Expose the image finder + translate trigger for reader.js so the reader
  // can pull the same translatable-image list the popup uses, and trigger
  // translation from its own toolbar.
  window.__mtFindTranslatableImages = findAllTranslatableImages;
  window.__mtStartTranslation = (ocrLang, targetLang, opts) =>
    startTranslationProcess(ocrLang, targetLang, opts);

  // ── Swipe up to return to reader ──────────────────────────────────────
  // If the user recently exited the reader (within 30s) and swipes up from
  // the bottom of the page, re-open the reader at their last position.
  if ('ontouchstart' in window || navigator.maxTouchPoints > 0) {
    let swipeStartY = 0;
    document.addEventListener('touchstart', (e) => {
      const t = e.changedTouches[0];
      swipeStartY = t.clientY;
    }, { passive: true });
    document.addEventListener('touchend', (e) => {
      const t = e.changedTouches[0];
      const dy = t.clientY - swipeStartY;
      // Swipe up from the bottom 15% of the screen.
      if (dy < -80 && Math.abs(dy) > Math.abs(t.clientX - (e.changedTouches[0] && 0)) &&
          swipeStartY > window.innerHeight * 0.85) {
        // Check if the reader was recently active.
        const exitTime = parseInt(sessionStorage.getItem('mtReaderExitTime') || '0', 10);
        if (exitTime && (Date.now() - exitTime) < 30000) {
          if (typeof window.__mtOpenReader === 'function') {
            console.log('[MangaTranslator] Swipe-up detected — reopening reader.');
            window.__mtOpenReader();
          }
        }
      }
    }, { passive: true });
  }

  // ========================================================================
  // MAIN TRANSLATION PROCESS (unchanged)
  // ========================================================================
  async function startTranslationProcess(selectedOcrLang, selectedTargetLang, opts = {}) {
    if (isTranslating) {
      console.warn("[MangaTranslator] Already translating, ignoring request.");
      return;
    }
    isTranslating = true;
    floatPopup.style.display = 'none';

    const stored = await chrome.storage.local.get([
      'serverUrl', 'ocrLang', 'colorize', 'targetLang', 'combineAmount',
      'contextMode', 'contextLevel', 'styleFontBold', 'styleFontItalic', 'styleFontRegular',
      'skipSfx', 'contextAware',
    ]);

    if (!stored.serverUrl) {
      alert("Please set your FastAPI Server URL in the extension popup or Advanced Settings!");
      isTranslating = false;
      return;
    }

    const serverUrl      = stored.serverUrl;
    const targetOcr      = selectedOcrLang   || stored.ocrLang   || 'ja';
    const targetLanguage = selectedTargetLang || stored.targetLang || 'en';
    const colorize       = stored.colorize !== false;
    // Combine amount: explicit opts first, then cached, then default 1.
    const combineAmount  = Math.max(1, Math.min(20, parseInt(opts.combineAmount ?? stored.combineAmount ?? '1', 10) || 1));
    // Merged context setting: explicit opts first, then cached, then legacy migration.
    const contextMode = opts.contextMode
      || stored.contextMode
      || ((stored.skipSfx === true || stored.contextAware === true) ? 'on' : 'off');
    const contextOn = contextMode === 'on';
    const contextLevel = (opts.contextLevel || stored.contextLevel) === 'high' ? 'high' : 'low';
    const styleAware = contextOn && contextLevel === 'high';
    const styleFonts = opts.styleFonts || {
      bold: stored.styleFontBold || '',
      italic: stored.styleFontItalic || '',
      regular: stored.styleFontRegular || '',
    };
    const groupOpts = { contextOn, contextLevel, styleAware, styleFonts };

    console.log(`[MangaTranslator] Starting — OCR Lang: ${targetOcr}, Lang: ${targetLanguage}, Colorize: ${colorize}, Combine: ${combineAmount}, Context: ${contextMode}/${contextLevel}, Server: ${serverUrl}`);

    let images = findAllTranslatableImages();
    if (images.length === 0) {
      alert("No suitable manga images found on this page. (Images must be at least 700k pixels and visible)");
      isTranslating = false;
      return;
    }

    images.sort((a, b) => a.getBoundingClientRect().top - b.getBoundingClientRect().top);

    console.log(`[MangaTranslator] Found ${images.length} images to translate.`);

    // ── Chunk into groups of `combineAmount` (27 / 10 → [10, 10, 7]) ──
    const groups = [];
    for (let i = 0; i < images.length; i += combineAmount) {
      groups.push(images.slice(i, i + combineAmount));
    }
    console.log(`[MangaTranslator] Grouped into ${groups.length} group(s): ${groups.map(g => g.length).join(', ')}`);

    const overlay = createProgressOverlay(images.length, targetOcr, colorize, targetLanguage);

    let processedImages = 0;
    let failedImages = 0;

    // ── Dispatch in waves of WAVE_SIZE concurrent jobs ────────────────────
    // Each group is an independent server job: processImage and
    // processImageGroup share no state, and the backend's Lens client already
    // caps its own fan-out with an internal semaphore, so several jobs can be
    // in flight safely.
    //
    // Waves rather than a rolling pool, by request: every job in a wave has to
    // finish before the next wave is dispatched. That also bounds peak memory
    // to WAVE_SIZE stitched canvases instead of the whole chapter, and keeps
    // page order roughly intact so the reader fills in top-to-bottom.
    const WAVE_SIZE = 3;

    async function runGroup(group, gIdx) {
      const spinner = createSpinner(group[0]);
      group.forEach(img => {
        img.style.outline = '4px solid yellow';
        img.style.outlineOffset = '-4px';
      });

      try {
        if (group.length === 1) {
          await processImage(group[0], serverUrl, colorize, targetLanguage, targetOcr, groupOpts);
          console.log(`[MangaTranslator] ✅ Done: ${group[0].dataset.mtTargetSrc}`);
        } else {
          await processImageGroup(group, serverUrl, colorize, targetLanguage, targetOcr, groupOpts);
          console.log(`[MangaTranslator] ✅ Group ${gIdx + 1}/${groups.length} done (${group.length} images)`);
        }
      } catch (e) {
        failedImages += group.length;
        console.error(`[MangaTranslator] ❌ Group ${gIdx + 1} failed before image replacement:`, e);
      }

      group.forEach(img => { img.style.outline = ''; });
      spinner.remove();
      // Counter is only touched after an await, and JS runs these callbacks on
      // one thread, so += needs no guarding.
      processedImages += group.length;
      updateOverlay(overlay, processedImages, images.length);
    }

    for (let wStart = 0; wStart < groups.length; wStart += WAVE_SIZE) {
      const wave = groups.slice(wStart, wStart + WAVE_SIZE);
      if (combineAmount > 1) {
        updateOverlayGroup(overlay, wStart + 1, groups.length, wave[0].length, processedImages, images.length);
      } else {
        updateOverlay(overlay, processedImages, images.length);
      }
      console.log(`[MangaTranslator] Wave ${Math.floor(wStart / WAVE_SIZE) + 1}: dispatching ${wave.length} job(s) concurrently`);
      await Promise.all(wave.map((g, i) => runGroup(g, wStart + i)));
    }

    // Every page keeps its own image — combine groups are sliced back apart
    // after translation, so there is nothing to hide or restore here.
    if (failedImages > 0) {
      overlay.innerText = `Translation finished with ${failedImages} failed image${failedImages === 1 ? '' : 's'}. Open the extension service-worker and page consoles for the exact handoff error.`;
      overlay.style.background = 'rgba(120, 24, 24, 0.97)';
      setTimeout(() => overlay.remove(), 12000);
    } else {
      overlay.innerText = `Done! (OCR Lang: ${targetOcr}, Lang: ${targetLanguage}, Colorize: ${colorize ? 'On' : 'Off'}, Combine: ${combineAmount})`;
      setTimeout(() => overlay.remove(), 4000);
    }
    isTranslating = false;
    // Final sweep signal: individual pages already fired mt-image-translated,
    // but a run can also fail partway or finish out of order. The reader does
    // one full reconcile here so nothing is left showing a stale source.
    try {
      document.dispatchEvent(new CustomEvent('mt-translation-complete'));
    } catch (e) {}
  }

  // ========================================================================
  // PROCESS A GROUP OF IMAGES (combine > ocr > translate > overlay > uncombine)
  // ========================================================================
  // Fetches each image's full-res bytes, stitches them vertically into one
  // canvas (centered on the widest, white background), sends the stitched
  // image through the normal translate flow so the backend can OCR across
  // page boundaries, then slices the returned overlaid image back apart along
  // the original page seams and puts each slice on its own <img>. Every page
  // in the group keeps its own element — nothing is hidden or removed.
  async function processImageGroup(group, serverUrl, colorize, targetLang, ocrLang, opts = {}) {
    // 1. Fetch each image's full-res bytes.
    const fetched = [];
    for (const img of group) {
      try {
        const r = await sendRuntimeMessage(
          { type: "fetchImage", url: img.dataset.mtTargetSrc },
          `Group image fetch timed out: ${img.dataset.mtTargetSrc}`,
        );
        if (r.success) fetched.push({ img, dataUrl: r.base64 });
        else console.warn(`[MangaTranslator] Fetch failed in group: ${img.dataset.mtTargetSrc}`);
      } catch (e) {
        console.warn(`[MangaTranslator] Fetch error in group: ${e}`);
      }
    }
    if (fetched.length === 0) throw new Error("No images in group could be fetched");

    // 2. Load each into an HTMLImageElement to measure dimensions.
    const loaded = await Promise.all(fetched.map(f => new Promise((resolve, reject) => {
      const el = new Image();
      el.onload = () => resolve({ img: f.img, el });
      el.onerror = () => reject(new Error("img load failed"));
      el.src = f.dataUrl;
    })));

    // 3. Compute stitched canvas dimensions.
    const maxW = Math.max(...loaded.map(l => l.el.naturalWidth));
    const totalH = loaded.reduce((sum, l) => sum + l.el.naturalHeight, 0);
    const canvas = document.createElement('canvas');
    canvas.width = maxW;
    canvas.height = totalH;
    const ctx = canvas.getContext('2d');
    ctx.fillStyle = '#ffffff';
    ctx.fillRect(0, 0, maxW, totalH);

    // 4. Draw each image centered horizontally, stacked vertically. Record
    //    each page's placement so the translated result can be sliced back
    //    apart along the exact same seams (the uncombine step).
    const seams = [];
    let y = 0;
    for (const l of loaded) {
      const x = Math.floor((maxW - l.el.naturalWidth) / 2);
      ctx.drawImage(l.el, 0, 0, l.el.naturalWidth, l.el.naturalHeight, x, y, l.el.naturalWidth, l.el.naturalHeight);
      seams.push({ img: l.img, x, y, w: l.el.naturalWidth, h: l.el.naturalHeight });
      y += l.el.naturalHeight;
    }
    const stitchedDataUrl = canvas.toDataURL('image/png');

    // 5. Send the stitched image for translation. OCR + translation run over
    //    the whole strip so dialogue that spans a page break is read as one
    //    unit, and the backend overlays the text before returning it.
    const initialSubmitResponse = await sendRuntimeMessage({
      type: "submitImage",
      serverUrl:   serverUrl,
      base64Data:  stitchedDataUrl,
      colorize:    colorize,
      targetLang:  targetLang,
      ocrLang:     ocrLang,
      contextMode:  opts.contextOn ? 'on' : 'off',
      contextLevel: opts.contextLevel || 'low',
      styleAware:   opts.styleAware === true,
      styleFonts:   opts.styleFonts || null,
    }, "Grouped backend submission timed out", 60000);
    const submitResponse = await finishBackendTranslation(serverUrl, initialSubmitResponse);
    const translatedStripDataUrl = submitResponse.image_data_url
      || (submitResponse.image_b64 ? `data:image/png;base64,${submitResponse.image_b64}` : '');
    if (!translatedStripDataUrl) {
      throw new Error(`Backend job ${submitResponse.job_id || 'unknown'} completed without a rendered image`);
    }

    // 6. Uncombine: load the overlaid strip and cut it back into per-page
    //    images. The backend may return the strip at a different scale than
    //    we sent it (colorize/upscale paths resize), so derive a scale factor
    //    from the returned dimensions and map the seam coordinates through it
    //    instead of assuming a 1:1 pixel match.
    const translatedStrip = await new Promise((resolve, reject) => {
      const el = new Image();
      el.onload = () => resolve(el);
      el.onerror = () => reject(new Error("translated strip load failed"));
      el.src = translatedStripDataUrl;
    });

    const scaleX = translatedStrip.naturalWidth / maxW;
    const scaleY = translatedStrip.naturalHeight / totalH;

    const sliceCanvas = document.createElement('canvas');
    const sliceCtx = sliceCanvas.getContext('2d');

    for (const s of seams) {
      const sx = Math.round(s.x * scaleX);
      const sy = Math.round(s.y * scaleY);
      const sw = Math.max(1, Math.round(s.w * scaleX));
      const sh = Math.max(1, Math.round(s.h * scaleY));

      sliceCanvas.width = sw;
      sliceCanvas.height = sh;
      sliceCtx.clearRect(0, 0, sw, sh);
      sliceCtx.drawImage(translatedStrip, sx, sy, sw, sh, 0, 0, sw, sh);

      await applyTranslatedSrc(s.img, sliceCanvas.toDataURL('image/png'));
    }
  }

  function loadTranslatedImage(newSrc) {
    return new Promise((resolve, reject) => {
      if (typeof newSrc !== 'string' || !newSrc.startsWith('data:image/')) {
        reject(new Error('Backend returned an invalid translated image data URL'));
        return;
      }
      const probe = new Image();
      probe.onload = () => {
        if (!probe.naturalWidth || !probe.naturalHeight) {
          reject(new Error('Translated image decoded with zero dimensions'));
          return;
        }
        resolve({ width: probe.naturalWidth, height: probe.naturalHeight });
      };
      probe.onerror = () => reject(new Error('Translated image could not be decoded by the page'));
      probe.src = newSrc;
    });
  }

  // Points an <img> (and any <picture>/srcset siblings that would otherwise
  // win the resolution race) at a translated result, and marks it so the
  // finder skips it and the reader notices the update.
  async function applyTranslatedSrc(img, newSrc) {
    const decoded = await loadTranslatedImage(newSrc);
    if (!img || !img.isConnected) {
      throw new Error('The target page image was removed before translation finished');
    }
    if (!img.dataset.mtOriginalSrc) img.dataset.mtOriginalSrc = img.currentSrc || img.src;

    const picture = img.closest('picture');
    if (picture) {
      picture.querySelectorAll('source').forEach(source => {
        source.removeAttribute('srcset');
        source.removeAttribute('data-srcset');
      });
    }
    img.removeAttribute('srcset');
    img.removeAttribute('data-srcset');
    img.srcset = '';
    for (const attr of ['data-src', 'data-original', 'data-lazy-src', 'data-url', 'data-image']) {
      if (img.hasAttribute(attr)) img.removeAttribute(attr);
    }
    img.src = newSrc;
    img.setAttribute('src', newSrc);
    img.dataset.mtTargetSrc = newSrc;
    img.setAttribute('data-mt-translated', 'true');

    await new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve)));
    const liveSrc = img.currentSrc || img.src;
    const replaced = liveSrc === newSrc || img.src === newSrc || liveSrc.startsWith('data:image/');
    if (!replaced) {
      throw new Error(`Site restored the original image after replacement (live src: ${liveSrc.slice(0, 120)})`);
    }

    console.log(
      `[MangaTranslator] Replaced page image with translated PNG ${decoded.width}x${decoded.height} ` +
      `(${Math.round(newSrc.length / 1024)} KiB data URL).`
    );
    try {
      document.dispatchEvent(new CustomEvent('mt-image-translated', { detail: { img } }));
    } catch (e) {}
  }

  // ========================================================================
  // SPINNER
  // ========================================================================
  function createSpinner(img) {
    const wrap = document.createElement('div');
    const rect = img.getBoundingClientRect();
    wrap.style.cssText = `
      position: absolute;
      display: flex; align-items: center; justify-content: center;
      background: rgba(0,0,0,0.7); z-index: 9999; pointer-events: none;
      border-radius: 4px;
      width: ${img.clientWidth}px; height: ${img.clientHeight}px;
      left: ${rect.left + window.scrollX}px; top: ${rect.top + window.scrollY}px;
    `;
    const spinner = document.createElement('div');
    spinner.style.cssText = `
      width: 40px; height: 40px; border: 5px solid #444;
      border-top: 5px solid #0066cc; border-radius: 50%;
      animation: mt-spin 1s linear infinite;
    `;
    wrap.appendChild(spinner);
    document.body.appendChild(wrap);
    return wrap;
  }

  // ========================================================================
  // IMAGE FINDER
  // ========================================================================
  // The size/pixel/aspect/non-data: gates below all answer one question: "is
  // this <img> a manga page worth sending to the backend?" For a page we have
  // already translated that question is settled, so callers that just want to
  // DISPLAY pages (the reader) pass includeTranslated and skip the re-decision.
  // Overlaid results fail several of those gates on purpose: they are inline
  // data: URLs, and the backend returns them smaller than the source, so the
  // 200px box floor and the 700k-pixel floor both drop them.
  function findAllTranslatableImages(opts = {}) {
    const includeTranslated = opts.includeTranslated === true;
    const allImages  = Array.from(document.querySelectorAll('img'));
    const validImages = [];

    for (const img of allImages) {
      // The reader renders its own <img> copies of every page inside its
      // overlay. Translating those is worthless — renderMode() destroys and
      // rebuilds them, so the result is thrown away — and a copy can shadow
      // the real page image in the dedupe pass below. Only ever hand back
      // images that belong to the host page. (Off-screen proxies live in
      // #mt-reader-proxies, outside the overlay, so they stay eligible.)
      if (img.closest('#mt-reader-overlay')) continue;

      const isTranslated = img.hasAttribute('data-mt-translated');
      // Translate flow (the default): skip translated pages so nothing gets
      // sent through the backend, and paid for, twice.
      if (isTranslated && !includeTranslated) continue;

      if (isTranslated) {
        if (!isElementVisible(img)) continue;
        // Trust the source recorded at overlay time instead of re-deriving it.
        // getBestImageUrl would actively pick the wrong image here: it prefers
        // data-src / data-original, which still hold the ORIGINAL untranslated
        // URL, and its srcset parser splits on ',' which truncates a base64
        // data: URL into an unusable "data:image/png;base64" stub.
        if (!img.dataset.mtTargetSrc) img.dataset.mtTargetSrc = img.currentSrc || img.src;
        validImages.push(img);
        continue;
      }

      const bestSrc = getBestImageUrl(img);
      if (!bestSrc || bestSrc.startsWith('data:') || bestSrc.startsWith('chrome://')) continue;
      if (!isElementVisible(img)) continue;
      if (!img.complete || img.naturalWidth === 0) continue;

      // Reject small rendered previews/thumbnails even when the underlying
      // source image is high-res (e.g. an 80x100 <img> pointing at a full
      // manga page). naturalWidth/naturalHeight reflect the SOURCE image; we
      // also check the on-page rendered box (clientWidth/clientHeight) and
      // exclude anything drawn smaller than 200x200 — those are previews,
      // not the actual reader page.
      if (img.clientWidth < 200 || img.clientHeight < 200) continue;

      const pixelCount  = img.naturalWidth * img.naturalHeight;
      if (pixelCount < 700000) continue;

      const aspectRatio = img.naturalWidth / img.naturalHeight;
      if (aspectRatio > 4.0 || aspectRatio < 0.2) continue;

      img.dataset.mtTargetSrc = bestSrc;
      validImages.push(img);
    }

    const seen = new Set();
    return validImages.filter(img => {
      if (seen.has(img.dataset.mtTargetSrc)) return false;
      seen.add(img.dataset.mtTargetSrc);
      return true;
    });
  }

  function getBestImageUrl(img) {
    if (img.srcset) {
      let bestUrl = img.src, bestW = 0;
      for (const entry of img.srcset.split(',').map(s => s.trim())) {
        const [url, descriptor] = entry.split(' ');
        const w = descriptor ? parseInt(descriptor.replace('w', '')) : 0;
        if (w > bestW) { bestW = w; bestUrl = url; }
      }
      if (bestUrl) return bestUrl;
    }
    for (const attr of ['data-src', 'data-original', 'data-lazy-src', 'data-url', 'data-image']) {
      const val = img.getAttribute(attr);
      if (val && val.startsWith('http')) return val;
    }
    const picture = img.closest('picture');
    if (picture) {
      for (const source of picture.querySelectorAll('source')) {
        if (source.srcset) return source.srcset.split(',')[0].trim().split(' ')[0];
      }
    }
    return img.src;
  }

  function isElementVisible(img) {
    if (img.clientWidth === 0 || img.clientHeight === 0) return false;
    const s = window.getComputedStyle(img);
    if (s.display === 'none' || s.visibility === 'hidden' || parseFloat(s.opacity) <= 0.1) return false;
    let parent = img.parentElement;
    while (parent && parent !== document.body) {
      const ps = window.getComputedStyle(parent);
      if (ps.display === 'none' || ps.visibility === 'hidden') return false;
      parent = parent.parentElement;
    }
    return true;
  }

  async function blobToDataUrl(blob) {
    return await new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onloadend = () => resolve(reader.result);
      reader.onerror = () => reject(new Error('Could not encode the rendered image'));
      reader.readAsDataURL(blob);
    });
  }

  async function finishBackendTranslation(serverUrl, response) {
    if (!response?.success) throw new Error(response?.error || 'API submission failed');
    if (!response.pending) return response;
    if (!response.job_id) throw new Error('Backend accepted translation without returning a job ID');

    const jobId = response.job_id;
    const deadline = Date.now() + 15 * 60 * 1000;
    console.log(`[MangaTranslator] Polling backend job ${jobId} from the page.`);
    try {
      while (Date.now() < deadline) {
        const statusResponse = await fetch(`${serverUrl}/v1/translate/${jobId}`);
        const status = await statusResponse.json().catch(() => ({}));
        if (!statusResponse.ok) {
          throw new Error(`Job ${jobId} poll failed: HTTP ${statusResponse.status}: ${status.detail || JSON.stringify(status)}`);
        }
        if (status.status === 'cancelled' || status.status === 'failed') {
          throw new Error(status.error || `Backend job ${jobId} ${status.status}`);
        }
        if (status.status === 'completed') {
          console.log(`[MangaTranslator] Job ${jobId} completed; fetching rendered PNG.`);
          const imageResponse = await fetch(`${serverUrl}/v1/translate/${jobId}/image`, { method: 'POST' });
          if (!imageResponse.ok) {
            const detail = await imageResponse.text().catch(() => '');
            throw new Error(`Rendered image fetch failed for job ${jobId}: HTTP ${imageResponse.status}${detail ? `: ${detail.slice(0, 300)}` : ''}`);
          }
          const blob = await imageResponse.blob();
          if (!blob.size) throw new Error(`Backend returned an empty rendered image for job ${jobId}`);
          const imageDataUrl = await blobToDataUrl(blob);
          await chrome.runtime.sendMessage({ type: 'stopTranslationHealth', jobId, active: true });
          console.log(`[MangaTranslator] Rendered PNG received for job ${jobId}: ${blob.size} bytes.`);
          return {
            success: true,
            job_id: jobId,
            image_data_url: imageDataUrl,
            image_b64: imageDataUrl.split(',', 2)[1] || '',
            image_bytes: blob.size,
          };
        }
        await new Promise(resolve => setTimeout(resolve, 5000));
      }
      throw new Error(`Backend job ${jobId} timed out after 15 minutes`);
    } catch (error) {
      await chrome.runtime.sendMessage({ type: 'stopTranslationHealth', jobId });
      throw error;
    }
  }

  async function sendRuntimeMessage(request, stage, timeoutMs = 960000) {
    return await Promise.race([
      chrome.runtime.sendMessage(request),
      new Promise((_, reject) => setTimeout(() => reject(new Error(`${stage} timed out after ${Math.round(timeoutMs / 1000)} seconds`)), timeoutMs)),
    ]);
  }

  // ========================================================================
  // PROCESS A SINGLE IMAGE
  // ========================================================================
  async function processImage(img, serverUrl, colorize, targetLang, ocrLang, opts = {}) {
    const targetSrc = img.dataset.mtTargetSrc;
    const initialSubmitResponse = await sendRuntimeMessage({
      type: "translateImageUrl",
      imageUrl: targetSrc,
      pageUrl: window.location.href,
      tabId: null,
      serverUrl: serverUrl,
      colorize: colorize,
      targetLang: targetLang,
      ocrLang: ocrLang,
      contextMode: opts.contextOn ? 'on' : 'off',
      contextLevel: opts.contextLevel || 'low',
      styleAware: opts.styleAware === true,
      styleFonts: opts.styleFonts || null,
    }, `Backend submission timed out: ${targetSrc}`, 60000);
    const submitResponse = await finishBackendTranslation(serverUrl, initialSubmitResponse);
    const translatedDataUrl = submitResponse.image_data_url
      || (submitResponse.image_b64 ? `data:image/png;base64,${submitResponse.image_b64}` : '');
    if (!translatedDataUrl) {
      throw new Error(`Backend job ${submitResponse.job_id || 'unknown'} completed without a rendered image`);
    }
    console.log(
      `[MangaTranslator] Received rendered image for job ${submitResponse.job_id || 'unknown'}: ` +
      `${submitResponse.image_bytes || 'unknown'} bytes.`
    );

    await applyTranslatedSrc(img, translatedDataUrl);
  }

  // ========================================================================
  // PROGRESS OVERLAY
  // ========================================================================
  function createProgressOverlay(total, ocrLang, colorize, targetLang) {
    const overlay = document.createElement('div');
    overlay.style.cssText = `
      position: fixed; top: 15px; right: 15px; z-index: 2147483647;
      padding: 15px 20px; background: rgba(30,30,46,0.95); color: #ffffff;
      border-radius: 8px; font-family: Arial, sans-serif; font-size: 14px;
      box-shadow: 0 4px 10px rgba(0,0,0,0.5); min-width: 260px; border: 1px solid #444;
    `;
    overlay.innerText = `Starting ${total} images… [OCR Lang: ${ocrLang}, Lang: ${targetLang}, Color: ${colorize ? 'On' : 'Off'}]`;
    document.body.appendChild(overlay);
    return overlay;
  }

  function updateOverlay(overlay, current, total, currentSrc) {
    const pct = total > 0 ? (current / total) * 100 : 0;
    if (currentSrc) {
      const short = currentSrc.length > 40 ? currentSrc.substring(0, 37) + '...' : currentSrc;
      overlay.innerHTML = `
        <div style="margin-bottom: 8px; font-weight: bold;">Translating ${current + 1} / ${total}</div>
        <div style="font-size: 11px; color: #aaa; word-break: break-all;">${short}</div>
        <div style="margin-top: 10px; height: 5px; background: #444; border-radius: 2px; overflow: hidden;">
          <div style="width: ${pct}%; height: 100%; background: #0066cc; transition: width 0.3s;"></div>
        </div>
      `;
    } else {
      overlay.innerHTML = `
        <div style="margin-bottom: 8px; font-weight: bold;">Processed ${current} / ${total}</div>
        <div style="margin-top: 10px; height: 5px; background: #444; border-radius: 2px; overflow: hidden;">
          <div style="width: ${pct}%; height: 100%; background: #28a745; transition: width 0.3s;"></div>
        </div>
      `;
    }
  }

  // Group variant: shows "Group X/Y (n pages)" instead of a per-image counter.
  function updateOverlayGroup(overlay, groupIdx, groupTotal, pagesInGroup, current, total) {
    const pct = total > 0 ? (current / total) * 100 : 0;
    overlay.innerHTML = `
      <div style="margin-bottom: 8px; font-weight: bold;">Group ${groupIdx} / ${groupTotal} — ${pagesInGroup} pages stitched</div>
      <div style="font-size: 11px; color: #aaa;">Processed ${current} / ${total} pages</div>
      <div style="margin-top: 10px; height: 5px; background: #444; border-radius: 2px; overflow: hidden;">
        <div style="width: ${pct}%; height: 100%; background: #0066cc; transition: width 0.3s;"></div>
      </div>
    `;
  }

})();