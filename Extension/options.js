document.addEventListener('DOMContentLoaded', () => {
  // Load saved URL
  chrome.storage.local.get(['serverUrl'], (data) => {
    document.getElementById('optServerUrl').value = data.serverUrl || 'http://localhost:7860';
  });

  // Save URL
  document.getElementById('saveUrlBtn').addEventListener('click', () => {
    const url = document.getElementById('optServerUrl').value.trim().replace(/\/$/, '');
    chrome.storage.local.set({ serverUrl: url }, () => {
      const status = document.getElementById('urlStatus');
      status.innerText = 'URL Saved!';
      setTimeout(() => status.innerText = '', 2000);
    });
  });

  // Switch OCR Model
  document.getElementById('setOcrBtn').addEventListener('click', async () => {
    const serverUrl = document.getElementById('optServerUrl').value.trim().replace(/\/$/, '');
    const model = document.getElementById('ocrSelect').value;
    try {
      const res = await fetch(`${serverUrl}/setmodel?model=${model}`, { method: "POST" });
      const data = await res.json();
      document.getElementById('ocrStatus').innerText = `OCR model set to: ${data.current_model}`;
    } catch (e) {
      document.getElementById('ocrStatus').innerHTML = `<span class="error">Error: ${e}</span>`;
    }
  });

  // List Models
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
        
        // Attach listeners to switch buttons
        document.querySelectorAll('#modelsTable button.success').forEach(btn => {
          btn.addEventListener('click', async (e) => {
            const repo = e.target.dataset.repo;
            const file = e.target.dataset.file;
            document.getElementById('modelStatus').innerText = `Switching to ${repo}/${file}...`;
            try {
              const res = await fetch(`${serverUrl}/v1/changemodel`, {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({repo_id: repo, filename: file})
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

  // Install Custom Model
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
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({repo_id: repo, filename: file || null})
      });
      const data = await res.json();
      if (res.ok) {
        document.getElementById('modelStatus').innerText = `Success! Active model: ${data.repo_id}/${data.filename}`;
        document.getElementById('refreshModelsBtn').click(); // Refresh list
      } else {
        document.getElementById('modelStatus').innerHTML = `<span class="error">Error: ${data.detail}</span>`;
      }
    } catch (err) {
      document.getElementById('modelStatus').innerHTML = `<span class="error">Error: ${err}</span>`;
    }
  });
});