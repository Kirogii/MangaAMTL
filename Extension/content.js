(function() {
  let isTranslating = false;
  let floatBtn, floatPopup;

  // Listen for messages from popup
  chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
    if (message.action === "translateAllImages") {
      startTranslationProcess(message.ocrModel);
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

  function injectUI() {
    // 1. Floating Button (Middle-left of screen, always visible)
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
    floatBtn.onmouseout = () => floatBtn.style.background = '#0052a3';
    floatBtn.onclick = (e) => { e.stopPropagation(); toggleFloatPopup(); };
    document.body.appendChild(floatBtn);

    // 2. Popup Menu (Beside the floating button)
    floatPopup = document.createElement('div');
    floatPopup.style.cssText = `
      position: fixed; top: 50%; left: 75px; transform: translateY(-50%);
      z-index: 2147483647; padding: 15px; background: #1e1e2e;
      border-radius: 8px; box-shadow: 0 4px 10px rgba(0,0,0,0.5);
      font-family: Arial, sans-serif; display: none; width: 220px;
      color: #e0e0e0; border: 1px solid #444;
    `;
    floatPopup.innerHTML = `
      <div style="margin-bottom: 10px; font-weight: bold; color: #ffffff;">OCR Language</div>
      <div style="margin-bottom: 15px; color: #ccc;">
        <input type="radio" id="mtOcrJa" name="mtOcrModel" value="ja" checked>
        <label for="mtOcrJa" style="font-size: 14px;">Japanese</label><br>
        <input type="radio" id="mtOcrKo" name="mtOcrModel" value="ko" style="margin-top: 5px;">
        <label for="mtOcrKo" style="font-size: 14px;">Korean</label>
      </div>
      <button id="mtStartBtn" style="width: 100%; padding: 10px; background: #28a745; color: white; border: none; border-radius: 4px; cursor: pointer; font-weight: bold; margin-bottom: 10px;">Translate All</button>
      <button id="mtSettingsBtn" style="width: 100%; padding: 10px; background: #3a3f4b; color: #fff; border: 1px solid #555; border-radius: 4px; cursor: pointer; font-weight: bold;">⚙️ Advanced Settings</button>
    `;
    document.body.appendChild(floatPopup);

    document.getElementById('mtStartBtn').onclick = () => {
      const selectedModel = document.querySelector('input[name="mtOcrModel"]:checked').value;
      floatPopup.style.display = 'none';
      startTranslationProcess(selectedModel);
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
      floatPopup.style.display = 'block';
    }
  }

  // ========================================================================
  // SETTINGS MODAL (Embedded in page - Dark Mode)
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

    // Inject options.html content (Dark Mode Styles Inlined)
    modal.insertAdjacentHTML('beforeend', `
      <h2 style="margin-top: 0; color: #ffffff;">Advanced Manga Translator Settings</h2>
      
      <div style="background: #2a2a3c; padding: 15px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.3); margin-bottom: 20px; border: 1px solid #333;">
        <h3 style="margin-top: 0; color: #ffffff;">API Server</h3>
        <label style="font-weight: bold; color: #aaaaaa; display: block; margin-bottom: 5px;" for="mtOptServerUrl">FastAPI Server URL:</label>
        <input type="text" id="mtOptServerUrl" placeholder="http://localhost:7860" style="width: 100%; padding: 8px; margin-bottom: 10px; box-sizing: border-box; border: 1px solid #555; border-radius: 4px; background: #1e1e2e; color: #fff;">
        <button id="mtSaveUrlBtn" style="padding: 10px 15px; background: #0066cc; color: white; border: none; border-radius: 4px; cursor: pointer; font-weight: bold;">Save URL</button>
        <div id="mtUrlStatus" style="margin-top: 10px; font-size: 14px; color: #28a745;"></div>
      </div>

      <div style="background: #2a2a3c; padding: 15px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.3); margin-bottom: 20px; border: 1px solid #333;">
        <h3 style="margin-top: 0; color: #ffffff;">OCR Model (Japanese/Korean)</h3>
        <div style="display: flex; gap: 10px; align-items: center; margin-bottom: 10px; flex-wrap: wrap;">
          <select id="mtOcrSelect" style="padding: 8px; border: 1px solid #555; border-radius: 4px; background: #1e1e2e; color: #fff;">
            <option value="ja">Japanese (Hayai+YOLO)</option>
            <option value="ko">Korean (PaddleOCR)</option>
          </select>
          <button id="mtSetOcrBtn" style="padding: 10px 15px; background: #0066cc; color: white; border: none; border-radius: 4px; cursor: pointer; font-weight: bold;">Switch OCR Model</button>
        </div>
        <div id="mtOcrStatus" style="margin-top: 10px; font-size: 14px; color: #28a745;"></div>
      </div>

      <div style="background: #2a2a3c; padding: 15px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.3); margin-bottom: 20px; border: 1px solid #333;">
        <h3 style="margin-top: 0; color: #ffffff;">Translation GGUF Model</h3>
        <div style="display: flex; gap: 10px; align-items: center; margin-bottom: 10px; flex-wrap: wrap;">
          <button id="mtRefreshModelsBtn" style="padding: 10px 15px; background: #3a3f4b; color: white; border: 1px solid #555; border-radius: 4px; cursor: pointer; font-weight: bold;">Refresh List</button>
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
        <label style="font-weight: bold; color: #aaaaaa; display: block; margin-bottom: 5px;">Repo ID (e.g. hugging-quants/Llama-3.2-1B-Instruct-GGUF):</label>
        <input type="text" id="mtCustomRepo" placeholder="repo_id" style="width: 100%; padding: 8px; margin-bottom: 10px; box-sizing: border-box; border: 1px solid #555; border-radius: 4px; background: #1e1e2e; color: #fff;">
        <label style="font-weight: bold; color: #aaaaaa; display: block; margin-bottom: 5px;">Filename (leave blank to auto-find):</label>
        <input type="text" id="mtCustomFile" placeholder="filename.gguf" style="width: 100%; padding: 8px; margin-bottom: 10px; box-sizing: border-box; border: 1px solid #555; border-radius: 4px; background: #1e1e2e; color: #fff;">
        <button id="mtInstallModelBtn" style="padding: 10px 15px; background: #28a745; color: white; border: none; border-radius: 4px; cursor: pointer; font-weight: bold;">Download & Switch</button>
        <div id="mtModelStatus" style="margin-top: 10px; font-size: 14px; color: #28a745;"></div>
      </div>
    `);

    overlay.appendChild(modal);
    document.body.appendChild(overlay);

    // Close if clicking outside the modal
    overlay.addEventListener('click', (e) => {
      if (e.target === overlay) overlay.remove();
    });

    initSettingsModalLogic(modal);
  }

  function initSettingsModalLogic(modal) {
    // Load saved URL
    chrome.storage.local.get(['serverUrl'], (data) => {
      modal.querySelector('#mtOptServerUrl').value = data.serverUrl || 'http://localhost:7860';
    });

    // Save URL
    modal.querySelector('#mtSaveUrlBtn').addEventListener('click', () => {
      const url = modal.querySelector('#mtOptServerUrl').value.trim().replace(/\/$/, '');
      chrome.storage.local.set({ serverUrl: url }, () => {
        const status = modal.querySelector('#mtUrlStatus');
        status.innerText = 'URL Saved!';
        setTimeout(() => status.innerText = '', 2000);
      });
    });

    // Switch OCR
    modal.querySelector('#mtSetOcrBtn').addEventListener('click', async () => {
      const serverUrl = modal.querySelector('#mtOptServerUrl').value.trim().replace(/\/$/, '');
      const model = modal.querySelector('#mtOcrSelect').value;
      try {
        const res = await fetch(`${serverUrl}/setmodel?model=${model}`, { method: "POST" });
        const data = await res.json();
        modal.querySelector('#mtOcrStatus').innerText = `OCR model set to: ${data.current_model}`;
      } catch (e) {
        modal.querySelector('#mtOcrStatus').innerHTML = `<span style="color:#ff4d4d;">Error: ${e}</span>`;
      }
    });

    // Refresh Models
    modal.querySelector('#mtRefreshModelsBtn').addEventListener('click', async () => {
      const serverUrl = modal.querySelector('#mtOptServerUrl').value.trim().replace(/\/$/, '');
      const tableBody = modal.querySelector('#mtModelsTable tbody');
      tableBody.innerHTML = '<tr><td colspan="4" style="text-align:center; padding: 8px; border: 1px solid #444; color: #aaa;">Loading...</td></tr>';
      
      try {
        const res = await fetch(`${serverUrl}/v1/listmodels`);
        const data = await res.json();
        tableBody.innerHTML = '';
        
        if (data.models && data.models.length > 0) {
          data.models.forEach(m => {
            const tr = document.createElement('tr');
            tr.innerHTML = `
              <td style="padding: 8px; border: 1px solid #444; color: #ccc;">${m.repo_id}</td>
              <td style="padding: 8px; border: 1px solid #444; color: #ccc;">${m.filename}</td>
              <td style="padding: 8px; border: 1px solid #444; color: #ccc;">${m.size_mb}</td>
              <td style="padding: 8px; border: 1px solid #444;"><button class="mt-success-btn" data-repo="${m.repo_id}" data-file="${m.filename}" style="padding: 5px 10px; background: #28a745; color: white; border: none; border-radius: 4px; cursor: pointer;">Switch</button></td>
            `;
            tableBody.appendChild(tr);
          });
          
          // Attach listeners to switch buttons
          tableBody.querySelectorAll('.mt-success-btn').forEach(btn => {
            btn.addEventListener('click', async (e) => {
              const repo = e.target.dataset.repo;
              const file = e.target.dataset.file;
              modal.querySelector('#mtModelStatus').innerText = `Switching to ${repo}/${file}...`;
              try {
                const res = await fetch(`${serverUrl}/v1/changemodel`, {
                  method: 'POST',
                  headers: {'Content-Type': 'application/json'},
                  body: JSON.stringify({repo_id: repo, filename: file})
                });
                const data = await res.json();
                if (res.ok) {
                  modal.querySelector('#mtModelStatus').innerText = `Active: ${data.repo_id}/${data.filename}`;
                } else {
                  modal.querySelector('#mtModelStatus').innerHTML = `<span style="color:#ff4d4d;">Error: ${data.detail}</span>`;
                }
              } catch (err) {
                modal.querySelector('#mtModelStatus').innerHTML = `<span style="color:#ff4d4d;">Error: ${err}</span>`;
              }
            });
          });
        } else {
          tableBody.innerHTML = '<tr><td colspan="4" style="text-align:center; padding: 8px; border: 1px solid #444; color: #aaa;">No models found in API server.</td></tr>';
        }
      } catch (e) {
        tableBody.innerHTML = `<tr><td colspan="4" style="text-align:center; padding: 8px; border: 1px solid #444; color:#ff4d4d;">Error fetching models: ${e}</td></tr>`;
      }
    });

    // Install Custom
    modal.querySelector('#mtInstallModelBtn').addEventListener('click', async () => {
      const serverUrl = modal.querySelector('#mtOptServerUrl').value.trim().replace(/\/$/, '');
      const repo = modal.querySelector('#mtCustomRepo').value.trim();
      const file = modal.querySelector('#mtCustomFile').value.trim();
      
      if (!repo) {
        alert("Please enter a Repo ID.");
        return;
      }
      
      modal.querySelector('#mtModelStatus').innerText = `Downloading & switching to ${repo}/${file || 'auto'}... (This may take a while)`;
      try {
        const res = await fetch(`${serverUrl}/v1/changemodel`, {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({repo_id: repo, filename: file || null})
        });
        const data = await res.json();
        if (res.ok) {
          modal.querySelector('#mtModelStatus').innerText = `Success! Active model: ${data.repo_id}/${data.filename}`;
          modal.querySelector('#mtRefreshModelsBtn').click(); // Refresh list
        } else {
          modal.querySelector('#mtModelStatus').innerHTML = `<span style="color:#ff4d4d;">Error: ${data.detail}</span>`;
        }
      } catch (err) {
        modal.querySelector('#mtModelStatus').innerHTML = `<span style="color:#ff4d4d;">Error: ${err}</span>`;
      }
    });
  }

  injectUI();

  async function startTranslationProcess(selectedOcrModel) {
    if (isTranslating) return;
    isTranslating = true;
    floatPopup.style.display = 'none';

    // Grab URL, OCR model, and Colorize settings
    const { serverUrl, ocrModel, colorize } = await chrome.storage.local.get(['serverUrl', 'ocrModel', 'colorize']);
    if (!serverUrl) {
      alert("Please set your FastAPI Server URL in the extension popup!");
      isTranslating = false;
      return;
    }

    const targetOcr = selectedOcrModel || ocrModel || 'ja';
    if (!['ja', 'ko'].includes(targetOcr)) {
      alert(`Invalid OCR model: ${targetOcr}`);
      isTranslating = false;
      return;
    }

    // Switch server model before doing any work
    try {
      const resp = await fetch(`${serverUrl}/setmodel?model=${targetOcr}`, { method: "POST" });
      if (!resp.ok) console.warn(`Server returned ${resp.status} when switching OCR model`);
    } catch (e) {
      console.warn("Failed to switch OCR model on server:", e);
    }

    let images = findAllTranslatableImages();
    if (images.length === 0) {
      alert("No suitable manga images found on this page. (Images must be at least 700k pixels and visible)");
      isTranslating = false;
      return;
    }

    // SORT IMAGES TOP TO BOTTOM
    images.sort((a, b) => {
      const rectA = a.getBoundingClientRect();
      const rectB = b.getBoundingClientRect();
      return rectA.top - rectB.top;
    });

    // Add spinners to all images initially
    const spinners = [];
    images.forEach(img => {
      spinners.push(createSpinner(img));
    });

    const overlay = createProgressOverlay(images.length, targetOcr, colorize);

    let processedCount = 0;
    for (const img of images) {
      updateOverlay(overlay, processedCount, images.length, img.src);
      
      // Add yellow border to currently working image
      img.style.outline = '4px solid yellow';
      img.style.outlineOffset = '-4px';

      try {
        await processImage(img, serverUrl, colorize);
      } catch (e) {
        console.error(`Failed to translate ${img.src}:`, e);
      }

      // Remove yellow border and spinner when done
      img.style.outline = '';
      const spinner = spinners.shift();
      if (spinner) spinner.remove();

      processedCount++;
      updateOverlay(overlay, processedCount, images.length);
    }

    overlay.innerText = `✅ Translation Complete! (OCR: ${targetOcr === 'ja' ? 'Japanese' : 'Korean'}, Colorize: ${colorize ? 'On' : 'Off'})`;
    setTimeout(() => overlay.remove(), 3000);
    isTranslating = false;
  }

  function createSpinner(img) {
    const overlay = document.createElement('div');
    overlay.style.cssText = `
      position: absolute; 
      display: flex; align-items: center; justify-content: center;
      background: rgba(0,0,0,0.7); z-index: 9999; pointer-events: none;
      border-radius: 4px;
    `;
    
    // Calculate position to perfectly overlap the image
    const rect = img.getBoundingClientRect();
    overlay.style.width = `${img.clientWidth}px`;
    overlay.style.height = `${img.clientHeight}px`;
    overlay.style.left = `${rect.left + window.scrollX}px`;
    overlay.style.top = `${rect.top + window.scrollY}px`;

    const spinner = document.createElement('div');
    spinner.style.cssText = `
      width: 40px; height: 40px; border: 5px solid #444;
      border-top: 5px solid #0066cc; border-radius: 50%;
      animation: mt-spin 1s linear infinite;
    `;
    overlay.appendChild(spinner);
    document.body.appendChild(overlay);
    return overlay;
  }

  // ========================================================================
  // ADVANCED IMAGE FINDER ALGORITHM
  // ========================================================================
  function findAllTranslatableImages() {
    const allImages = Array.from(document.querySelectorAll('img'));
    const validImages = [];

    for (const img of allImages) {
      if (img.hasAttribute('data-mt-translated')) continue;

      const bestSrc = getBestImageUrl(img);
      if (!bestSrc || bestSrc.startsWith('data:') || bestSrc.startsWith('chrome://')) continue;

      if (!isElementVisible(img)) continue;

      if (!img.complete || img.naturalWidth === 0) {
        continue;
      }

      const width = img.naturalWidth;
      const height = img.naturalHeight;
      const pixelCount = width * height;

      if (pixelCount < 700000) continue;

      const aspectRatio = width / height;
      if (aspectRatio > 4.0 || aspectRatio < 0.2) continue;

      img.dataset.mtTargetSrc = bestSrc;
      validImages.push(img);
    }

    const uniqueImages = [];
    const seenSrcs = new Set();
    for (const img of validImages) {
      if (!seenSrcs.has(img.dataset.mtTargetSrc)) {
        seenSrcs.add(img.dataset.mtTargetSrc);
        uniqueImages.push(img);
      }
    }

    return uniqueImages;
  }

  function getBestImageUrl(img) {
    if (img.srcset) {
      const srcsetEntries = img.srcset.split(',').map(s => s.trim());
      let bestUrl = img.src;
      let bestW = 0;
      for (const entry of srcsetEntries) {
        const [url, descriptor] = entry.split(' ');
        const w = descriptor ? parseInt(descriptor.replace('w', '')) : 0;
        if (w > bestW) {
          bestW = w;
          bestUrl = url;
        }
      }
      if (bestUrl) return bestUrl;
    }

    const lazyAttrs = ['data-src', 'data-original', 'data-lazy-src', 'data-url', 'data-image'];
    for (const attr of lazyAttrs) {
      const val = img.getAttribute(attr);
      if (val && val.startsWith('http')) return val;
    }

    const picture = img.closest('picture');
    if (picture) {
      const sources = picture.querySelectorAll('source');
      for (const source of sources) {
        if (source.srcset) {
          return source.srcset.split(',')[0].trim().split(' ')[0];
        }
      }
    }

    return img.src;
  }

  function isElementVisible(img) {
    if (img.clientWidth === 0 || img.clientHeight === 0) return false;
    
    const style = window.getComputedStyle(img);
    if (style.display === 'none' || style.visibility === 'hidden' || parseFloat(style.opacity) <= 0.1) {
      return false;
    }

    let parent = img.parentElement;
    while (parent && parent !== document.body) {
      const pStyle = window.getComputedStyle(parent);
      if (pStyle.display === 'none' || pStyle.visibility === 'hidden') {
        return false;
      }
      parent = parent.parentElement;
    }

    return true;
  }

  // ========================================================================
  // PROCESSING & REPLACEMENT LOGIC
  // ========================================================================
  async function processImage(img, serverUrl, colorize) {
    const targetSrc = img.dataset.mtTargetSrc;

    const fetchResponse = await chrome.runtime.sendMessage({ type: "fetchImage", url: targetSrc });
    if (!fetchResponse.success) throw new Error("Failed to fetch image");

    const submitResponse = await chrome.runtime.sendMessage({
      type: "submitImage",
      serverUrl: serverUrl,
      base64Data: fetchResponse.base64,
      colorize: colorize // Pass the colorize setting to background.js
    });

    if (submitResponse.success && submitResponse.image_b64) {
      const newSrc = `data:image/png;base64,${submitResponse.image_b64}`;
      
      if (!img.dataset.mtOriginalSrc) {
        img.dataset.mtOriginalSrc = img.src;
      }

      img.src = newSrc;
      img.setAttribute('data-mt-translated', 'true');

      if (img.srcset) img.srcset = newSrc;
      
      const picture = img.closest('picture');
      if (picture) {
        picture.querySelectorAll('source').forEach(source => {
          source.srcset = newSrc;
        });
      }
    } else {
      throw new Error(submitResponse.error || "API submission failed");
    }
  }

  // ========================================================================
  // UI OVERLAY
  // ========================================================================
  function createProgressOverlay(total, ocrModel, colorize) {
    const overlay = document.createElement('div');
    overlay.style.cssText = `
      position: fixed;
      top: 15px;
      right: 15px;
      z-index: 2147483647;
      padding: 15px 20px;
      background: rgba(30, 30, 46, 0.95);
      color: #ffffff;
      border-radius: 8px;
      font-family: Arial, sans-serif;
      font-size: 14px;
      box-shadow: 0 4px 10px rgba(0,0,0,0.5);
      min-width: 250px;
      border: 1px solid #444;
    `;
    const label = ocrModel === 'ko' ? 'Korean (PaddleOCR)' : 'Japanese (Hayai+YOLO)';
    const colLabel = colorize ? 'On' : 'Off';
    overlay.innerText = `Starting translation of ${total} images... [OCR: ${label}, Color: ${colLabel}]`;
    document.body.appendChild(overlay);
    return overlay;
  }

  function updateOverlay(overlay, current, total, currentSrc) {
    if (currentSrc) {
      const shortSrc = currentSrc.length > 40 ? currentSrc.substring(0, 37) + '...' : currentSrc;
      overlay.innerHTML = `
        <div style="margin-bottom: 10px; font-weight: bold;">Translating ${current + 1} / ${total}</div>
        <div style="font-size: 11px; color: #aaa; word-break: break-all;">${shortSrc}</div>
        <div style="margin-top: 10px; height: 5px; background: #444; border-radius: 2px; overflow: hidden;">
          <div style="width: ${(current/total)*100}%; height: 100%; background: #0066cc; transition: width 0.3s;"></div>
        </div>
      `;
    } else {
      overlay.innerHTML = `
        <div style="margin-bottom: 10px; font-weight: bold;">Processed ${current} / ${total}</div>
        <div style="margin-top: 10px; height: 5px; background: #444; border-radius: 2px; overflow: hidden;">
          <div style="width: ${(current/total)*100}%; height: 100%; background: #28a745; transition: width 0.3s;"></div>
        </div>
      `;
    }
  }
})();