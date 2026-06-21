// Load saved URL + OCR model
document.addEventListener('DOMContentLoaded', () => {
  chrome.storage.local.get(['serverUrl', 'ocrModel'], (data) => {
    document.getElementById('serverUrl').value = data.serverUrl || 'http://localhost:7860';
    const model = data.ocrModel || 'ja';
    const radio = document.querySelector(`input[name="ocrModel"][value="${model}"]`);
    if (radio) radio.checked = true;
  });
});

// Save URL + OCR model
document.getElementById('saveBtn').addEventListener('click', () => {
  const url = document.getElementById('serverUrl').value.trim().replace(/\/$/, '');
  const ocrModel = document.querySelector('input[name="ocrModel"]:checked').value;
  chrome.storage.local.set({ serverUrl: url, ocrModel: ocrModel }, () => {
    const status = document.getElementById('status');
    status.innerText = 'Settings saved!';
    setTimeout(() => status.innerText = '', 2000);
  });
});

// Trigger translation on current page (pass chosen OCR model)
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