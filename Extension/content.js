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
    const container = document.getElementById('mtFontWeightPicker');
    if (!container || container.dataset.built === '1') return;
    container.dataset.built = '1';
    container.innerHTML = '';

    const { serverUrl, fontWeight } = await chrome.storage.local.get(['serverUrl', 'fontWeight']);
    const selected = fontWeight !== undefined ? parseInt(fontWeight, 10) : 2;
    document.getElementById('mtFontWeightHidden').value = selected;

    let fontFamily = 'sans-serif';
    if (serverUrl) {
      try {
        const face = await loadServerFontFace(serverUrl);
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
      const cssW = 40, cssH = 32;
      canvas.style.width = cssW + 'px';
      canvas.style.height = cssH + 'px';
      canvas.width = cssW * dpr;
      canvas.height = cssH * dpr;
      canvas.dataset.level = level;
      canvas.style.cssText += `border-radius:4px; border:2px solid ${level === selected ? '#28a745' : '#444'}; display:block;`;
      drawFontWeightSwatch(canvas, level, fontFamily);

      const lbl = document.createElement('div');
      lbl.innerText = level + 1;
      lbl.style.cssText = 'font-size:10px; color:#aaa; margin-top:2px;';

      wrap.appendChild(canvas);
      wrap.appendChild(lbl);
      wrap.onclick = async () => {
        document.getElementById('mtFontWeightHidden').value = level;
        chrome.storage.local.set({ fontWeight: String(level) });
        container.querySelectorAll('canvas').forEach(c => {
          c.style.border = `2px solid ${parseInt(c.dataset.level, 10) === level ? '#28a745' : '#444'}`;
        });
        // ★ 1:1 with popup.js — push font weight to server immediately on click
        const { serverUrl: url } = await chrome.storage.local.get(['serverUrl']);
        if (url) {
          try {
            await fetch(`${url}/SetFont`, {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify({ stroke_width: level })
            });
          } catch (e) {
            console.warn('[MangaTranslator] Failed to push font weight to server:', e);
          }
        }
      };
      container.appendChild(wrap);
    }
  }

  function refreshFontWeightSelection() {
    chrome.storage.local.get(['fontWeight'], (data) => {
      const selected = data.fontWeight !== undefined ? parseInt(data.fontWeight, 10) : 2;
      const hidden = document.getElementById('mtFontWeightHidden');
      if (hidden) hidden.value = selected;
      const container = document.getElementById('mtFontWeightPicker');
      if (container) {
        container.querySelectorAll('canvas').forEach(c => {
          c.style.border = `2px solid ${parseInt(c.dataset.level, 10) === selected ? '#28a745' : '#444'}`;
        });
      }
    });
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
      const { fontWeight } = await chrome.storage.local.get(['fontWeight']);
      const strokeWidth = fontWeight !== undefined ? parseInt(fontWeight, 10) : 2;
      const res = await fetch(`${serverUrl}/SetFont`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ font_name: filename, stroke_width: strokeWidth })
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
      startTranslationProcess(message.ocrLang, message.targetLang);
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
        statusEl.innerText = data.ocr_mode === 'lens' ? 'Google Lens active' : (data.ocr_mode === 'glm' ? 'GLM active' : 'Hayai active');
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
        if (statusEl) statusEl.innerText = data.ocr_mode === 'lens' ? 'Google Lens active' : (data.ocr_mode === 'glm' ? 'GLM active' : 'Hayai active');
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
  // Helper: load all cached settings into the floating popup — 1:1 with popup.js
  // ========================================================================
  async function loadCachedSettingsIntoPopup() {
    const sel = document.getElementById('mtTargetLangSelect');
    const ocrLangSel = document.getElementById('mtOcrLangSelect');

    const data = await chrome.storage.local.get(
      ['serverUrl', 'ocrLang', 'colorize', 'targetLang', 'fontWeight', 'modelType',
       'openrouterModel', 'openrouterApiKey', 'inpaintMode', 'ocrMode', 'cloudMode']
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
        <button id="mtSettingsBtn" title="Settings" aria-label="Settings" style="width:34px; height:34px; padding:0; flex:0 0 auto; background:#2a2a3c; border:1px solid #444; border-radius:8px; color:#fff; cursor:pointer; display:flex; align-items:center; justify-content:center;">
          <svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="3"/><path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 1 1-2.83 2.83l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-4 0v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 1 1-2.83-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1 0-4h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 1 1 2.83-2.83l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 4 0v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 1 1 2.83 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 0 4h-.09a1.65 1.65 0 0 0-1.51 1z"/></svg>
        </button>
      </div>

      <button id="mtStartBtn" style="
        width:100%; box-sizing:border-box; padding:15px; background:#22a552; color:#fff; border:none;
        border-radius:8px; cursor:pointer; font-weight:600; font-size:16px; margin-bottom:12px;
        box-shadow:0 3px 10px rgba(34,165,82,0.35); display:flex; align-items:center; justify-content:center;
      ">${ICON.arrow}<span>Translate All</span></button>

      <label style="${labCss}">${ICON.globe}Translate to</label>
      <select id="mtTargetLangSelect" style="${selCss}"><!-- populated by JS --></select>

      <div style="margin:0 0 12px 0; padding:10px; background:#1b2430; border-radius:8px; border:1px solid #3a6ea5; display:flex; align-items:center;">
        <input type="checkbox" id="mtCloudMode" style="width:auto; margin:0;">
        <label for="mtCloudMode" style="display:flex; align-items:center; font-weight:500; margin:0 0 0 8px; cursor:pointer; color:#d5d5e2;">${ICON.cloud}Cloud Mode — minimal PC load</label>
      </div>

      <div id="mtStatus" style="font-size:12px; color:#46c877; text-align:center; min-height:15px; margin-bottom:4px;"></div>

      <div id="mtSettingsPanel" style="display:none; border-top:1px solid #2c2c3a; margin-top:4px; padding-top:14px;">
        <div style="${secCss}">${ICON.scan}OCR</div>
        <label style="${labCss}">OCR Engine</label>
        <select id="mtOcrModeSelect" style="${selCss}">
          <option value="hayai">Hayai (Local, Japanese)</option>
          <option value="glm">GLM (Local, Korean)</option>
          <option value="lens">Google Lens (Cloud, All)</option>
        </select>
        <div id="mtOcrModeStatus" style="font-size:12px; color:#46c877; text-align:center; min-height:15px; margin-bottom:8px;"></div>
        <label style="${labCss}">Source Language</label>
        <select id="mtOcrLangSelect" style="${selCss}"><!-- populated by JS --></select>

        <div style="${secCss}">${ICON.type}Typesetting</div>
        <label style="${labCss}">Font Family</label>
        <div id="mtFontFamilyScroll" style="display:flex; gap:6px; overflow-x:auto; overflow-y:hidden; white-space:nowrap; padding-bottom:4px;">
          <div style="font-size:11px; color:#888;">Loading fonts…</div>
        </div>
        <div id="mtFontFamilyStatus" style="font-size:11px; color:#46c877; margin:4px 0 12px; min-height:14px;"></div>

        <label style="${labCss}">Font Boldness</label>
        <div id="mtFontWeightPicker" style="display:flex; gap:6px; margin-bottom:12px;">
          <div style="font-size:11px; color:#888;">Loading preview…</div>
        </div>
        <input type="hidden" id="mtFontWeightHidden" value="2">

        <div style="${secCss}">${ICON.inpaint}Inpainting</div>
        <label style="${labCss}">Inpainting Mode</label>
        <select id="mtInpaintMode" style="${selCss}">
          <option value="low">Low (faster, default)</option>
          <option value="high">High (better quality, bigger model)</option>
          <option value="none">None (fill with bg color, no model)</option>
        </select>
        <div style="${hintCss}">High mode auto-downloads a larger LaMa model. 'None' fills text regions with the detected background color instead of inpainting.</div>
        <div id="mtInpaintModeStatus" style="font-size:12px; color:#46c877; text-align:center; min-height:15px; margin-bottom:8px;"></div>

        <div style="${secCss}">${ICON.brain}Translation Model</div>
        <select id="mtModelTypeSelect" style="${selCss}">
          <option value="local">Local (GGUF)</option>
          <option value="openrouter">OpenRouter</option>
        </select>
        <div id="mtOpenrouterRow" style="display:none; padding:10px; background:#1c1c26; border-radius:8px; border:1px solid #3a3a4c; margin-bottom:12px;">
          <input type="text" id="mtOpenrouterModel" placeholder="e.g. openai/gpt-4o-mini" style="${inCss}">
          <input type="password" id="mtOpenrouterKey" placeholder="OpenRouter API Key" style="${inCss}">
          <button id="mtSetModelBtn" style="width:100%; box-sizing:border-box; padding:9px; background:#2a2a3c; color:#fff; border:1px solid #4a4a5e; border-radius:8px; cursor:pointer; font-weight:600; font-size:13px;">Set Model</button>
          <div id="mtModelStatus" style="margin-top:6px; font-size:11px; color:#46c877;"></div>
        </div>

        <div style="margin:0 0 12px 0; padding:10px; background:#22222e; border-radius:8px; border:1px solid #3a3a4c; display:flex; align-items:center;">
          <input type="checkbox" id="mtColorize" style="width:auto; margin:0;">
          <label for="mtColorize" style="display:flex; align-items:center; font-weight:500; margin:0 0 0 8px; cursor:pointer; color:#d5d5e2;">${ICON.palette}Enable Colorization</label>
        </div>

        <button id="mtAdvancedBtn" style="width:100%; box-sizing:border-box; padding:11px; background:transparent; color:#b9b9cc; border:1px solid #3a3a4c; border-radius:8px; cursor:pointer; font-weight:500; font-size:14px; display:flex; align-items:center; justify-content:center;">${ICON.wrench}Advanced Settings</button>
      </div>
    `;
    document.body.appendChild(floatPopup);

    // ── Wire the focal Translate button FIRST ──
    wireTranslateButton();

    try {
      loadCachedSettingsIntoPopup();
      initFontWeightPicker();
      chrome.storage.local.get(['serverUrl'], (d) => initFontFamilyPicker(d.serverUrl || ''));
    } catch (e) {
      console.warn('[MangaTranslator] Popup init (pickers/cache) failed:', e);
    }

    // ★ Model type change — 1:1 with popup.js
    document.getElementById('mtModelTypeSelect').onchange = (e) => {
      const isOpenRouter = e.target.value === 'openrouter';
      document.getElementById('mtOpenrouterRow').style.display = isOpenRouter ? 'block' : 'none';
      chrome.storage.local.set({ modelType: e.target.value });

      if (!isOpenRouter) {
        chrome.storage.local.get(['serverUrl'], async (d) => {
          if (d.serverUrl) {
            try {
              await fetch(`${d.serverUrl}/SetModelType`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ model_type: 'local' })
              });
            } catch (e) {
              console.warn('[MangaTranslator] Failed to switch to local model:', e);
            }
          }
        });
      }
    };

    // ★ Inpaint mode change — 1:1 with popup.js (real-time push to server)
    document.getElementById('mtInpaintMode').onchange = (e) => {
      chrome.storage.local.set({ inpaintMode: e.target.value });
      chrome.storage.local.get(['serverUrl'], (d) => {
        pushInpaintMode(d.serverUrl, e.target.value);
      });
    };

    // ★ OCR mode change — 1:1 with popup.js (real-time push to server)
    document.getElementById('mtOcrModeSelect').onchange = (e) => {
      chrome.storage.local.set({ ocrMode: e.target.value });
      chrome.storage.local.get(['serverUrl'], (d) => {
        pushOcrMode(d.serverUrl, e.target.value);
      });
    };

    // ★ Colorize change — cache immediately — 1:1 with popup.js
    document.getElementById('mtColorize').onchange = (e) => {
      chrome.storage.local.set({ colorize: e.target.checked });
    };

    // ★ Set OpenRouter model — 1:1 with popup.js
    document.getElementById('mtSetModelBtn').onclick = async () => {
      const statusEl = document.getElementById('mtModelStatus');
      const { serverUrl } = await chrome.storage.local.get(['serverUrl']);
      const model = document.getElementById('mtOpenrouterModel').value.trim();
      const apiKey = document.getElementById('mtOpenrouterKey').value.trim();

      if (!serverUrl) { alert("Please set your FastAPI Server URL in Advanced Settings first!"); return; }
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
          chrome.storage.local.set({ modelType: 'openrouter', openrouterModel: model, openrouterApiKey: apiKey });
          statusEl.innerText = `Active: ${data.openrouter_model}`;
        } else {
          statusEl.innerHTML = `<span style="color:#ff4d4d;">Error: ${data.detail}</span>`;
        }
      } catch (e) {
        statusEl.innerHTML = `<span style="color:#ff4d4d;">Error: ${e}</span>`;
      }
    };

    // ★ Translate All button — 1:1 with popup.js (caches ALL settings + pushes to server)
    function wireTranslateButton() {
      document.getElementById('mtStartBtn').onclick = async () => {
        const cloudMode      = document.getElementById('mtCloudMode').checked;
        const selectedOcrMode   = document.getElementById('mtOcrModeSelect').value;
        const selectedOcrLang   = document.getElementById('mtOcrLangSelect').value;
        const selectedLang      = document.getElementById('mtTargetLangSelect').value;
        const selectedWeight    = document.getElementById('mtFontWeightHidden').value;
        const selectedModelType = document.getElementById('mtModelTypeSelect').value;
        const selectedInpaint   = document.getElementById('mtInpaintMode').value;
        const colorize          = document.getElementById('mtColorize').checked;

        const orModelEl = document.getElementById('mtOpenrouterModel');
        const orKeyEl = document.getElementById('mtOpenrouterKey');

        // ★ Cache ALL settings — 1:1 with popup.js
        const settingsToCache = {
          targetLang: selectedLang,
          fontWeight: selectedWeight,
          modelType: selectedModelType,
          ocrMode: selectedOcrMode,
          ocrLang: selectedOcrLang,
          inpaintMode: selectedInpaint,
          colorize: colorize,
          cloudMode: cloudMode,
        };
        if (orModelEl && orModelEl.value.trim()) settingsToCache.openrouterModel = orModelEl.value.trim();
        if (orKeyEl && orKeyEl.value.trim()) settingsToCache.openrouterApiKey = orKeyEl.value.trim();

        // Include the active font family (the chip with the green border), if any.
        const activeFamilyChip = Array.from(document.querySelectorAll('#mtFontFamilyScroll button[data-filename]'))
          .find(c => c.style.borderColor === 'rgb(40, 167, 69)' || c.style.borderColor === '#28a745');
        if (activeFamilyChip) settingsToCache.fontFamily = activeFamilyChip.dataset.filename;

        const { serverUrl } = await chrome.storage.local.get(['serverUrl']);
        settingsToCache.serverUrl = serverUrl || '';
        chrome.storage.local.set(settingsToCache);

        // ★ In cloud mode, push cloud settings to server first — 1:1 with popup.js
        if (cloudMode && serverUrl) {
          await pushCloudModeFromPopup(serverUrl);
        } else if (serverUrl) {
          // Push font weight
          try {
            await fetch(`${serverUrl}/SetFont`, {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify({ stroke_width: parseInt(selectedWeight, 10) })
            });
          } catch (e) {
            console.warn('[MangaTranslator] Failed to push font weight to server:', e);
          }

          // Push inpaint mode
          await pushInpaintMode(serverUrl, selectedInpaint);

          // Push OCR mode
          await pushOcrMode(serverUrl, selectedOcrMode);

          // If Local is selected, ensure server is switched back to local model
          if (selectedModelType === 'local') {
            try {
              await fetch(`${serverUrl}/SetModelType`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ model_type: 'local' })
              });
            } catch (e) {
              console.warn('[MangaTranslator] Failed to switch server to local model:', e);
            }
          }
        }

        floatPopup.style.display = 'none';
        startTranslationProcess(selectedOcrLang, selectedLang);
      };
    }

    // Remaining wiring is best-effort
    try {
      // Gear button toggles settings panel
      document.getElementById('mtSettingsBtn').onclick = () => {
        const panel = document.getElementById('mtSettingsPanel');
        const open = panel.style.display === 'none';
        panel.style.display = open ? 'block' : 'none';
        chrome.storage.local.set({ settingsPanelOpen: open });
      };
      chrome.storage.local.get(['settingsPanelOpen'], (d) => {
        document.getElementById('mtSettingsPanel').style.display = d.settingsPanelOpen === true ? 'block' : 'none';
      });

      // Advanced Settings button
      document.getElementById('mtAdvancedBtn').onclick = () => {
        floatPopup.style.display = 'none';
        openSettingsModal();
      };

      // ★ Cloud Mode toggle — 1:1 with popup.js
      const cloudEl = document.getElementById('mtCloudMode');
      cloudEl.onchange = async (e) => {
        const on = e.target.checked;
        applyCloudModeToPopup(on);
        const { serverUrl } = await chrome.storage.local.get(['serverUrl']);

        if (on) {
          chrome.storage.local.set({
            cloudMode: true,
            ocrMode: 'lens',
            modelType: 'openrouter',
            inpaintMode: 'none',
            colorize: false,
          });
          await pushCloudModeFromPopup(serverUrl);
        } else {
          chrome.storage.local.set({ cloudMode: false });
          await disableCloudModeOnServer(serverUrl);
        }
      };
    } catch (e) {
      console.warn('[MangaTranslator] Popup secondary wiring failed:', e);
    }
  }

  function toggleFloatPopup() {
    if (floatPopup.style.display === 'block') {
      floatPopup.style.display = 'none';
    } else {
      // Re-sync dropdowns + API key with cache every time popup opens
      loadCachedSettingsIntoPopup();
      refreshFontWeightSelection();
      chrome.storage.local.get(['serverUrl'], (d) => initFontFamilyPicker(d.serverUrl || ''));
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
              <th style="padding: 8px; border: 1px solid #444; text-align: left; background: #1e1e2e; color: #fff;">Size (MB)</th>
              <th style="padding: 8px; border: 1px solid #444; text-align: left; background: #1e1e2e; color: #fff;">Action</th>
            </tr>
          </thead>
          <tbody>
            <tr><td colspan="4" style="text-align:center; padding: 8px; border: 1px solid #444; color: #aaa;">Click "Refresh List" to load models...</td></tr>
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
      tableBody.innerHTML = '<tr><td colspan="4" style="text-align:center; padding: 8px; border: 1px solid #444; color: #aaa;">Loading...</td></tr>';
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
                } else {
                  modal.querySelector('#mtModelInstallStatus').innerHTML = `<span style="color:#ff4d4d;">Error: ${data.detail}</span>`;
                }
              } catch (err) {
                modal.querySelector('#mtModelInstallStatus').innerHTML = `<span style="color:#ff4d4d;">Error: ${err}</span>`;
              }
            });
          });
        } else {
          tableBody.innerHTML = '<tr><td colspan="4" style="text-align:center; padding: 8px; border: 1px solid #444; color: #aaa;">No models found.</td></tr>';
        }
      } catch (e) {
        tableBody.innerHTML = `<tr><td colspan="4" style="text-align:center; padding: 8px; border: 1px solid #444; color:#ff4d4d;">Error: ${e}</td></tr>`;
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

  // ========================================================================
  // MAIN TRANSLATION PROCESS (unchanged)
  // ========================================================================
  async function startTranslationProcess(selectedOcrLang, selectedTargetLang) {
    if (isTranslating) {
      console.warn("[MangaTranslator] Already translating, ignoring request.");
      return;
    }
    isTranslating = true;
    floatPopup.style.display = 'none';

    const stored = await chrome.storage.local.get(['serverUrl', 'ocrLang', 'colorize', 'targetLang']);

    if (!stored.serverUrl) {
      alert("Please set your FastAPI Server URL in the extension popup or Advanced Settings!");
      isTranslating = false;
      return;
    }

    const serverUrl      = stored.serverUrl;
    const targetOcr      = selectedOcrLang   || stored.ocrLang   || 'ja';
    const targetLanguage = selectedTargetLang || stored.targetLang || 'en';
    const colorize       = stored.colorize !== false;

    console.log(`[MangaTranslator] Starting — OCR Lang: ${targetOcr}, Lang: ${targetLanguage}, Colorize: ${colorize}, Server: ${serverUrl}`);

    let images = findAllTranslatableImages();
    if (images.length === 0) {
      alert("No suitable manga images found on this page. (Images must be at least 700k pixels and visible)");
      isTranslating = false;
      return;
    }

    images.sort((a, b) => a.getBoundingClientRect().top - b.getBoundingClientRect().top);

    console.log(`[MangaTranslator] Found ${images.length} images to translate.`);

    const spinners = images.map(img => createSpinner(img));
    const overlay  = createProgressOverlay(images.length, targetOcr, colorize, targetLanguage);

    let processedCount = 0;
    for (const img of images) {
      updateOverlay(overlay, processedCount, images.length, img.src);
      img.style.outline = '4px solid yellow';
      img.style.outlineOffset = '-4px';

      try {
        await processImage(img, serverUrl, colorize, targetLanguage, targetOcr);
        console.log(`[MangaTranslator] ✅ Done: ${img.dataset.mtTargetSrc}`);
      } catch (e) {
        console.error(`[MangaTranslator] ❌ Failed: ${img.src}`, e);
      }

      img.style.outline = '';
      const spinner = spinners.shift();
      if (spinner) spinner.remove();

      processedCount++;
      updateOverlay(overlay, processedCount, images.length);
    }

    overlay.innerText = `✅ Done! (OCR Lang: ${targetOcr}, Lang: ${targetLanguage}, Colorize: ${colorize ? 'On' : 'Off'})`;
    setTimeout(() => overlay.remove(), 4000);
    isTranslating = false;
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
  function findAllTranslatableImages() {
    const allImages  = Array.from(document.querySelectorAll('img'));
    const validImages = [];

    for (const img of allImages) {
      if (img.hasAttribute('data-mt-translated')) continue;

      const bestSrc = getBestImageUrl(img);
      if (!bestSrc || bestSrc.startsWith('data:') || bestSrc.startsWith('chrome://')) continue;
      if (!isElementVisible(img)) continue;
      if (!img.complete || img.naturalWidth === 0) continue;

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

  // ========================================================================
  // PROCESS A SINGLE IMAGE
  // ========================================================================
  async function processImage(img, serverUrl, colorize, targetLang, ocrLang) {
    const targetSrc = img.dataset.mtTargetSrc;

    const fetchResponse = await chrome.runtime.sendMessage({ type: "fetchImage", url: targetSrc });
    if (!fetchResponse.success) throw new Error(`fetchImage failed: ${fetchResponse.error}`);

    const submitResponse = await chrome.runtime.sendMessage({
      type: "submitImage",
      serverUrl:   serverUrl,
      base64Data:  fetchResponse.base64,
      colorize:    colorize,
      targetLang:  targetLang,
      ocrLang:     ocrLang,
    });

    if (!submitResponse.success) throw new Error(submitResponse.error || "API submission failed");

    const newSrc = `data:image/png;base64,${submitResponse.image_b64}`;
    if (!img.dataset.mtOriginalSrc) img.dataset.mtOriginalSrc = img.src;
    img.src = newSrc;
    img.setAttribute('data-mt-translated', 'true');
    if (img.srcset) img.srcset = newSrc;

    const picture = img.closest('picture');
    if (picture) picture.querySelectorAll('source').forEach(s => { s.srcset = newSrc; });
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

})();