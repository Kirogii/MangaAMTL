document.addEventListener('DOMContentLoaded', () => {
  // ① Load all 4 settings including targetLang
  chrome.storage.local.get(['serverUrl', 'ocrModel', 'colorize', 'targetLang'], (data) => {
    document.getElementById('serverUrl').value = data.serverUrl || 'http://localhost:7860';
    const radio = document.querySelector(`input[name="ocrModel"][value="${data.ocrModel || 'ja'}"]`);
    if (radio) radio.checked = true;
    document.getElementById('colorize').checked = data.colorize !== false;
    document.getElementById('targetLang').value = data.targetLang || 'en'; // ← ADD
  });
});

document.getElementById('saveBtn').addEventListener('click', () => {
  const url = document.getElementById('serverUrl').value.trim().replace(/\/$/, '');
  const ocrModel = document.querySelector('input[name="ocrModel"]:checked').value;
  const colorize = document.getElementById('colorize').checked;
  const targetLang = document.getElementById('targetLang').value; // ← ADD

  // ② Save all 4 settings
  chrome.storage.local.set({ serverUrl: url, ocrModel: ocrModel, colorize: colorize, targetLang: targetLang }, () => {
    const status = document.getElementById('status');
    status.innerText = 'Settings saved!';
    setTimeout(() => status.innerText = '', 2000);
  });
});

document.getElementById('translateBtn').addEventListener('click', () => {
  // ③ Read from storage so the value is always current, then send targetLang
  chrome.storage.local.get(['ocrModel', 'targetLang'], (data) => {
    chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
      chrome.tabs.sendMessage(tabs[0].id, {
        action: "translateAllImages",
        ocrModel: data.ocrModel || 'ja',
        targetLang: data.targetLang || 'en'  // ← ADD
      });
      window.close();
    });
  });
});

document.getElementById('optionsBtn').addEventListener('click', () => {
  chrome.runtime.openOptionsPage();
});