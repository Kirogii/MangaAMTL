document.addEventListener('DOMContentLoaded', () => {
  // Load all settings including fontWeight and modelType
  chrome.storage.local.get(
    ['serverUrl', 'ocrModel', 'colorize', 'targetLang', 'fontWeight', 'modelType', 'openrouterModel'],
    (data) => {
      document.getElementById('serverUrl').value = data.serverUrl || 'http://localhost:7860';
      const radio = document.querySelector(`input[name="ocrModel"][value="${data.ocrModel || 'ja'}"]`);
      if (radio) radio.checked = true;
      document.getElementById('colorize').checked = data.colorize !== false;
      document.getElementById('targetLang').value = data.targetLang || 'en';
      document.getElementById('fontWeight').value = data.fontWeight !== undefined ? data.fontWeight : '2';

      const modelType = data.modelType || 'local';
      document.getElementById('modelType').value = modelType;
      document.getElementById('openrouterBox').style.display = modelType === 'openrouter' ? 'block' : 'none';
      if (data.openrouterModel) {
        document.getElementById('openrouterModel').value = data.openrouterModel;
      }
    }
  );
});

// Toggle OpenRouter fields when model type changes
document.getElementById('modelType').addEventListener('change', (e) => {
  const isOpenRouter = e.target.value === 'openrouter';
  document.getElementById('openrouterBox').style.display = isOpenRouter ? 'block' : 'none';
  chrome.storage.local.set({ modelType: e.target.value });
});

// Set OpenRouter model / api key on the server
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
      chrome.storage.local.set({ modelType: 'openrouter', openrouterModel: model });
      statusEl.innerText = `Active: ${data.openrouter_model}`;
    } else {
      statusEl.innerHTML = `<span style="color:#ff4d4d;">Error: ${data.detail}</span>`;
    }
  } catch (err) {
    statusEl.innerHTML = `<span style="color:#ff4d4d;">Error: ${err}</span>`;
  }
});

document.getElementById('saveBtn').addEventListener('click', async () => {
  const url = document.getElementById('serverUrl').value.trim().replace(/\/$/, '');
  const ocrModel = document.querySelector('input[name="ocrModel"]:checked').value;
  const colorize = document.getElementById('colorize').checked;
  const targetLang = document.getElementById('targetLang').value;
  const fontWeight = document.getElementById('fontWeight').value;
  const modelType = document.getElementById('modelType').value;

  chrome.storage.local.set({
    serverUrl: url,
    ocrModel: ocrModel,
    colorize: colorize,
    targetLang: targetLang,
    fontWeight: fontWeight,
    modelType: modelType
  }, () => {
    const status = document.getElementById('status');
    status.innerText = 'Settings saved!';
    setTimeout(() => status.innerText = '', 2000);
  });

  // Push font boldness to the server right away
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

    // If Local is selected, make sure the server switches back to local model
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

document.getElementById('translateBtn').addEventListener('click', () => {
  chrome.storage.local.get(['ocrModel', 'targetLang'], (data) => {
    chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
      chrome.tabs.sendMessage(tabs[0].id, {
        action: "translateAllImages",
        ocrModel: data.ocrModel || 'ja',
        targetLang: data.targetLang || 'en'
      });
      window.close();
    });
  });
});

document.getElementById('optionsBtn').addEventListener('click', () => {
  chrome.runtime.openOptionsPage();
});
