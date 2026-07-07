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
      wrap.onclick = () => {
        document.getElementById('mtFontWeightHidden').value = level;
        chrome.storage.local.set({ fontWeight: String(level) });
        container.querySelectorAll('canvas').forEach(c => {
          c.style.border = `2px solid ${parseInt(c.dataset.level, 10) === level ? '#28a745' : '#444'}`;
        });
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
  // Helper: load all cached settings into the floating popup dropdowns
  // ========================================================================
  function loadCachedSettingsIntoPopup() {
    chrome.storage.local.get(
      ['targetLang', 'fontWeight', 'modelType', 'openrouterModel', 'openrouterApiKey', 'ocrMode', 'ocrLang'],
      (data) => {
        const sel = document.getElementById('mtTargetLangSelect');
        if (sel && data.targetLang) sel.value = data.targetLang;

        const modelTypeSel = document.getElementById('mtModelTypeSelect');
        const openrouterRow = document.getElementById('mtOpenrouterRow');
        if (modelTypeSel) {
          modelTypeSel.value = data.modelType || 'local';
          openrouterRow.style.display = modelTypeSel.value === 'openrouter' ? 'block' : 'none';
        }
        if (data.openrouterModel) {
          const orModelInput = document.getElementById('mtOpenrouterModel');
          if (orModelInput) orModelInput.value = data.openrouterModel;
        }
        // Load cached API key (displays as •••• because input is type="password")
        if (data.openrouterApiKey) {
          const orKeyInput = document.getElementById('mtOpenrouterKey');
          if (orKeyInput) orKeyInput.value = data.openrouterApiKey;
        }
        const ocrModeSel = document.getElementById('mtOcrModeSelect');
        if (ocrModeSel) ocrModeSel.value = data.ocrMode || 'hayai';
        const ocrLangSel = document.getElementById('mtOcrLangSelect');
        if (ocrLangSel) ocrLangSel.value = data.ocrLang || 'ja';
      }
    );
  }

  function injectUI() {
    // 1. Floating Button
    floatBtn = document.createElement('button');
    floatBtn.innerText = '⚙️';
    floatBtn.style.cssText = `
      position: fixed; top: 50%; left: 15px; transform: translateY(-50%);
      z-index: 2147483647; width: 50px; height: 50px;
      background: #0052a3; color: white; border: 1px solid #007bff; border-radius: 50%;
      cursor: pointer; font-size: 24px; box-shadow: 0 4px 6px rgba(0,0,0,0.5);
      display: flex; align-items: center; justify-content: center;
    `;
    floatBtn.onmouseover = () => floatBtn.style.background = '#0066cc';
    floatBtn.onmouseout  = () => floatBtn.style.background = '#0052a3';
    floatBtn.onclick = (e) => { e.stopPropagation(); toggleFloatPopup(); };
    document.body.appendChild(floatBtn);

    // 2. Popup Menu — includes Target Language, Font Boldness, and Model selectors
    floatPopup = document.createElement('div');
    floatPopup.style.cssText = `
      position: fixed; top: 50%; left: 75px; transform: translateY(-50%);
      z-index: 2147483647; padding: 15px; background: #1e1e2e;
      border-radius: 8px; box-shadow: 0 4px 10px rgba(0,0,0,0.5);
      font-family: Arial, sans-serif; display: none; width: 260px;
      color: #e0e0e0; border: 1px solid #444;
      max-height: 85vh; overflow-y: auto;
    `;
    floatPopup.innerHTML = `
      <div style="margin-bottom: 14px;">
        <div style="font-weight: bold; color: #ffffff; margin-bottom: 6px;">OCR Engine</div>
        <select id="mtOcrModeSelect" style="
          width: 100%; padding: 7px 8px; border-radius: 4px;
          border: 1px solid #555; background: #2a2a3c; color: #e0e0e0;
          font-size: 13px; cursor: pointer; margin-bottom: 10px;
        ">
          <option value="hayai">Hayai (Local, Japanese)</option>
          <option value="glm">GLM (Local, Korean)</option>
          <option value="lens">Google Lens (Cloud, All)</option>
        </select>
        <div style="font-weight: bold; color: #ffffff; margin-bottom: 6px;">OCR Language</div>
        <select id="mtOcrLangSelect" style="
          width: 100%; padding: 7px 8px; border-radius: 4px;
          border: 1px solid #555; background: #2a2a3c; color: #e0e0e0;
          font-size: 13px; cursor: pointer;
        ">
          <option value="ja">Japanese</option>
          <option value="ko">Korean</option>
          <option value="en">English</option>
          <option value="zh">Chinese</option>
          <option value="ru">Russian</option>
        </select>
      </div>

      <div style="margin-bottom: 14px;">
        <div style="font-weight: bold; color: #ffffff; margin-bottom: 6px;">Target Language</div>
        <select id="mtTargetLangSelect" style="
          width: 100%; padding: 7px 8px; border-radius: 4px;
          border: 1px solid #555; background: #2a2a3c; color: #e0e0e0;
          font-size: 13px; cursor: pointer;
        ">
          <option value="en">English</option>
          <option value="es">Spanish</option>
          <option value="ru">Russian</option>
          <option value="id">Indonesian</option>
          <option value="ko">Korean</option>
          <option value="ja">Japanese</option>
          <option value="zh">Chinese</option>
        </select>
      </div>

      <div style="margin-bottom: 14px;">
        <div style="font-weight: bold; color: #ffffff; margin-bottom: 6px;">Font Boldness</div>
        <div id="mtFontWeightPicker" style="display:flex; gap:6px;">
          <div style="font-size:11px; color:#888;">Loading preview…</div>
        </div>
        <input type="hidden" id="mtFontWeightHidden" value="2">
      </div>

      <div style="margin-bottom: 14px;">
        <div style="font-weight: bold; color: #ffffff; margin-bottom: 6px;">Translation Model</div>
        <select id="mtModelTypeSelect" style="
          width: 100%; padding: 7px 8px; border-radius: 4px;
          border: 1px solid #555; background: #2a2a3c; color: #e0e0e0;
          font-size: 13px; cursor: pointer; margin-bottom: 8px;
        ">
          <option value="local">Local (GGUF)</option>
          <option value="openrouter">OpenRouter</option>
        </select>
        <div id="mtOpenrouterRow" style="display:none; padding: 8px; background: #16161f; border-radius: 4px; border: 1px solid #333;">
          <input type="text" id="mtOpenrouterModel" placeholder="e.g. openai/gpt-4o-mini" style="
            width: 100%; padding: 7px 8px; margin-bottom: 6px; box-sizing: border-box;
            border: 1px solid #555; border-radius: 4px; background: #2a2a3c; color: #e0e0e0; font-size: 12px;">
          <input type="password" id="mtOpenrouterKey" placeholder="OpenRouter API Key" style="
            width: 100%; padding: 7px 8px; margin-bottom: 6px; box-sizing: border-box;
            border: 1px solid #555; border-radius: 4px; background: #2a2a3c; color: #e0e0e0; font-size: 12px;">
          <button id="mtSetModelBtn" style="
            width: 100%; padding: 8px; background: #3a3f4b; color: #fff;
            border: 1px solid #555; border-radius: 4px; cursor: pointer;
            font-weight: bold; font-size: 12px;
          ">Set Model</button>
          <div id="mtModelStatus" style="margin-top: 6px; font-size: 11px; color: #28a745;"></div>
        </div>
      </div>

      <button id="mtStartBtn" style="
        width: 100%; padding: 10px; background: #28a745; color: white;
        border: none; border-radius: 4px; cursor: pointer;
        font-weight: bold; margin-bottom: 10px; font-size: 14px;
      ">Translate All</button>
      <button id="mtSettingsBtn" style="
        width: 100%; padding: 10px; background: #3a3f4b; color: #fff;
        border: 1px solid #555; border-radius: 4px; cursor: pointer;
        font-weight: bold; font-size: 14px;
      ">⚙️ Advanced Settings</button>
    `;
    document.body.appendChild(floatPopup);

    // Restore saved settings into the dropdowns (autoload from cache)
    loadCachedSettingsIntoPopup();

    initFontWeightPicker();

    // Toggle OpenRouter fields when model type changes
    document.getElementById('mtModelTypeSelect').onchange = (e) => {
      document.getElementById('mtOpenrouterRow').style.display = e.target.value === 'openrouter' ? 'block' : 'none';
      chrome.storage.local.set({ modelType: e.target.value });
    };

    // Set OpenRouter model on the server — also caches the API key
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
          // Cache model + API key so they persist across popup reopens
          chrome.storage.local.set({ modelType: 'openrouter', openrouterModel: model, openrouterApiKey: apiKey });
          statusEl.innerText = `Active: ${data.openrouter_model}`;
        } else {
          statusEl.innerHTML = `<span style="color:#ff4d4d;">Error: ${data.detail}</span>`;
        }
      } catch (e) {
        statusEl.innerHTML = `<span style="color:#ff4d4d;">Error: ${e}</span>`;
      }
    };

    // Translate All button — reads OCR mode/lang, target language, font weight, and model type
    document.getElementById('mtStartBtn').onclick = async () => {
      const selectedOcrMode   = document.getElementById('mtOcrModeSelect').value;
      const selectedOcrLang   = document.getElementById('mtOcrLangSelect').value;
      const selectedLang      = document.getElementById('mtTargetLangSelect').value;
      const selectedWeight    = document.getElementById('mtFontWeightHidden').value;
      const selectedModelType = document.getElementById('mtModelTypeSelect').value;

      // Persist choices so the extension popup stays in sync
      chrome.storage.local.set({
        targetLang: selectedLang,
        fontWeight: selectedWeight,
        modelType: selectedModelType,
        ocrMode: selectedOcrMode,
        ocrLang: selectedOcrLang
      });
      floatPopup.style.display = 'none';

      const { serverUrl } = await chrome.storage.local.get(['serverUrl']);
      if (serverUrl) {
        // Push font boldness to server
        try {
          await fetch(`${serverUrl}/SetFont`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ stroke_width: parseInt(selectedWeight, 10) })
          });
        } catch (e) {
          console.warn('[MangaTranslator] Failed to push font weight to server:', e);
        }

        // Push OCR Engine Mode to server
        try {
          await fetch(`${serverUrl}/SetOcrMode`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ mode: selectedOcrMode })
          });
        } catch (e) {
          console.warn('[MangaTranslator] Failed to push OCR mode to server:', e);
        }

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

      startTranslationProcess(selectedOcrLang, selectedLang);
    };

    document.getElementById('mtSettingsBtn').onclick = () => {
      floatPopup.style.display = 'none';
      openSettingsModal();
    };
  }

  function toggleFloatPopup() {
    if (floatPopup.style.display === 'block') {
      floatPopup.style.display = 'none';
    } else {
      // Re-sync dropdowns + API key with cache every time popup opens
      loadCachedSettingsIntoPopup();
      refreshFontWeightSelection();
      floatPopup.style.display = 'block';
    }
  }

  // ========================================================================
  // SETTINGS MODAL
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
  // MAIN TRANSLATION PROCESS
  // ========================================================================
  async function startTranslationProcess(selectedOcrLang, selectedTargetLang) {
    if (isTranslating) {
      console.warn("[MangaTranslator] Already translating, ignoring request.");
      return;
    }
    isTranslating = true;
    floatPopup.style.display = 'none';

    // Load all settings from storage
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

    // Sort top to bottom
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

    // Deduplicate by src
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