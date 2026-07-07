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
  const family = `MTFontOptions_${filename.replace(/[^a-zA-Z0-9]/g, '_')}`;
  const face = new FontFace(family, buf);
  await face.load();
  document.fonts.add(face);
  _mtFontByteCache.set(cacheKey, face);
  return { face, filename, strokeWidth: info.stroke_width };
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

async function initFontWeightPicker(serverUrl) {
  const container = document.getElementById('optFontWeightPicker');
  if (!container) return;
  container.innerHTML = '';

  const { fontWeight } = await chrome.storage.local.get(['fontWeight']);
  const selected = fontWeight !== undefined ? parseInt(fontWeight, 10) : 2;
  document.getElementById('optFontWeightHidden').value = selected;

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
    wrap.onclick = async () => {
      document.getElementById('optFontWeightHidden').value = level;
      chrome.storage.local.set({ fontWeight: String(level) });
      container.querySelectorAll('canvas').forEach(c => {
        c.style.border = `2px solid ${parseInt(c.dataset.level, 10) === level ? '#28a745' : '#555'}`;
      });

      const url = document.getElementById('optServerUrl').value.trim().replace(/\/$/, '');
      const statusEl = document.getElementById('optFontStatus');
      if (!url) return;
      statusEl.innerText = 'Saving...';
      try {
        await fetch(`${url}/SetFont`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ stroke_width: level })
        });
        statusEl.innerText = `Stroke width set to ${level}`;
      } catch (e) {
        statusEl.innerHTML = `<span class="error">Error: ${e}</span>`;
      }
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
  });
}

async function selectFontFamily(serverUrl, filename, container) {
  const statusEl = document.getElementById('optFontFamilyStatus');
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
      statusEl.innerHTML = `<span class="error">Error: ${data.detail}</span>`;
      return;
    }

    statusEl.innerText = `Active: ${filename}`;
    chrome.storage.local.set({ fontFamily: filename });
    container.querySelectorAll('.font-chip').forEach(c => {
      c.classList.toggle('active', c.dataset.filename === filename);
    });

    await initFontWeightPicker(serverUrl);
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
  } catch (e) {
    console.warn('[MangaTranslator] Could not fetch model type from server:', e);
  }
}

async function syncInpaintModeFromServer(serverUrl) {
  const statusEl = document.getElementById('optInpaintStatus');
  if (!serverUrl) return;
  try {
    const res = await fetch(`${serverUrl}/GetInpaintMode`);
    const data = await res.json();
    document.getElementById('optInpaintMode').value = data.inpaint_mode || 'low';
    chrome.storage.local.set({ inpaintMode: data.inpaint_mode || 'low' });
    statusEl.innerText = data.inpaint_mode === 'high'
      ? (data.high_model_downloaded ? `High model ready (${data.high_model_size_mb} MB)` : 'High model will download on first use')
      : '';
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
      statusEl.innerText = data.inpaint_mode === 'high'
        ? `High model ready (${data.high_model_size_mb} MB)`
        : 'Low mode active';
      chrome.storage.local.set({ inpaintMode: data.inpaint_mode });
    } else {
      statusEl.innerHTML = `<span class="error">Error: ${data.detail}</span>`;
    }
  } catch (e) {
    statusEl.innerHTML = `<span class="error">Error: ${e}</span>`;
  }
}

async function syncOcrModeFromServer(serverUrl) {
  const statusEl = document.getElementById('optOcrStatus');
  if (!serverUrl) return;
  try {
    const res = await fetch(`${serverUrl}/GetOcrMode`);
    const data = await res.json();
    document.getElementById('optOcrMode').value = data.ocr_mode || 'hayai';
    chrome.storage.local.set({ ocrMode: data.ocr_mode || 'hayai' });
    statusEl.innerText = data.ocr_mode === 'lens' ? 'Google Lens active' : (data.ocr_mode === 'glm' ? 'GLM active' : 'Hayai active');
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
      statusEl.innerText = data.ocr_mode === 'lens' ? 'Google Lens active' : (data.ocr_mode === 'glm' ? 'GLM active' : 'Hayai active');
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
    ['serverUrl', 'modelType', 'openrouterModel', 'openrouterApiKey', 'inpaintMode', 'ocrMode', 'fontWeight'],
    (data) => {
      const url = data.serverUrl || 'http://localhost:7860';
      document.getElementById('optServerUrl').value = url;

      // ★ Autoload cached dropdown selections
      if (data.ocrMode) document.getElementById('optOcrMode').value = data.ocrMode;
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

      initFontFamilyPicker(url);
      initFontWeightPicker(url);
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
  initFontWeightPicker(url);
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
  const fontWeight = document.getElementById('optFontWeightHidden').value;

  chrome.storage.local.set({
    serverUrl: url,
    ocrMode: ocrMode,
    modelType: modelType,
    openrouterModel: openrouterModel,
    openrouterApiKey: openrouterApiKey,
    inpaintMode: inpaintMode,
    fontWeight: fontWeight
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
  chrome.storage.local.set({ ocrMode: e.target.value });
  pushOcrMode(serverUrl, e.target.value);
});

// ============================================================================
// GGUF MODEL LIST / SWITCH / INSTALL
// ============================================================================
document.getElementById('refreshModelsBtn').addEventListener('click', async () => {
  const serverUrl = document.getElementById('optServerUrl').value.trim().replace(/\/$/, '');
  const tableBody = document.querySelector('#modelsTable tbody');
  tableBody.innerHTML = '<tr><td colspan="4" style="text-align:center;">Loading...</td></tr>';

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
            } else {
              document.getElementById('modelStatus').innerHTML = `<span class="error">Error: ${data.detail}</span>`;
            }
          } catch (err) {
            document.getElementById('modelStatus').innerHTML = `<span class="error">Error: ${err}</span>`;
          }
        });
      });
    } else {
      tableBody.innerHTML = '<tr><td colspan="4" style="text-align:center;">No models found in API server.</td></tr>';
    }
  } catch (e) {
    tableBody.innerHTML = `<tr><td colspan="4" class="error" style="text-align:center;">Error fetching models: ${e}</td></tr>`;
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
      document.getElementById('refreshModelsBtn').click();
    } else {
      document.getElementById('modelStatus').innerHTML = `<span class="error">Error: ${data.detail}</span>`;
    }
  } catch (err) {
    document.getElementById('modelStatus').innerHTML = `<span class="error">Error: ${err}</span>`;
  }
});