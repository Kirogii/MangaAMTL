// Load saved URL + OCR model + Colorize
document.addEventListener('DOMContentLoaded', () => {
  chrome.storage.local.get(['serverUrl', 'ocrModel', 'colorize'], (data) => {
    document.getElementById('serverUrl').value = data.serverUrl || 'http://localhost:7860';
    const model = data.ocrModel || 'ja';
    const radio = document.querySelector(`input[name="ocrModel"][value="${model}"]`);
    if (radio) radio.checked = true;
    
    // Default to true if not set
    document.getElementById('colorize').checked = data.colorize !== false;
  });
});

// Save URL + OCR model + Colorize
document.getElementById('saveBtn').addEventListener('click', () => {
  const url = document.getElementById('serverUrl').value.trim().replace(/\/$/, '');
  const ocrModel = document.querySelector('input[name="ocrModel"]:checked').value;
  const colorize = document.getElementById('colorize').checked;
  
  chrome.storage.local.set({ serverUrl: url, ocrModel: ocrModel, colorize: colorize }, () => {
    const status = document.getElementById('status');
    status.innerText = 'Settings saved!';
    setTimeout(() => status.innerText = '', 2000);
  });
});

// Trigger translation on current page
document.getElementById('translateBtn').addEventListener('click', () => {
  const ocrModel = document.querySelector('input[name="ocrModel"]:checked').value;
  chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
    chrome.tabs.sendMessage(tabs[0].id, {
      action: "translateAllImages",
      ocrModel: ocrModel
    });
    window.close();
  });
});

// Open Advanced Settings page in a new tab
document.getElementById('optionsBtn').addEventListener('click', () => {
  chrome.runtime.openOptionsPage();
});