(function() {
  let isTranslating = false;
  let topBtn, floatBtn, floatPopup;

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
    // 1. Top Button (Visible at top of page)
    topBtn = document.createElement('button');
    topBtn.innerText = '🌐 Translate Manga';
    topBtn.style.cssText = `
      position: fixed; top: 15px; left: 15px; z-index: 2147483647;
      padding: 10px 15px; background: #0066cc; color: white;
      border: none; border-radius: 5px; cursor: pointer;
      font-family: Arial, sans-serif; font-weight: bold;
      box-shadow: 0 4px 6px rgba(0,0,0,0.3); transition: opacity 0.3s;
    `;
    topBtn.onmouseover = () => topBtn.style.background = '#0052a3';
    topBtn.onmouseout = () => topBtn.style.background = '#0066cc';
    topBtn.onclick = () => toggleFloatPopup();
    document.body.appendChild(topBtn);

    // 2. Floating Button (Middle of screen, appears on scroll)
    floatBtn = document.createElement('button');
    floatBtn.innerText = '⚙️';
    floatBtn.style.cssText = `
      position: fixed; top: 50%; left: 15px; transform: translateY(-50%);
      z-index: 2147483647; width: 50px; height: 50px;
      background: #0066cc; color: white; border: none; border-radius: 50%;
      cursor: pointer; font-size: 24px; box-shadow: 0 4px 6px rgba(0,0,0,0.3);
      display: none; /* hidden initially */
    `;
    floatBtn.onmouseover = () => floatBtn.style.background = '#0052a3';
    floatBtn.onmouseout = () => floatBtn.style.background = '#0066cc';
    floatBtn.onclick = (e) => { e.stopPropagation(); toggleFloatPopup(); };
    document.body.appendChild(floatBtn);

    // 3. Popup Menu (Beside the floating button)
    floatPopup = document.createElement('div');
    floatPopup.style.cssText = `
      position: fixed; top: 50%; left: 75px; transform: translateY(-50%);
      z-index: 2147483647; padding: 15px; background: white;
      border-radius: 8px; box-shadow: 0 4px 10px rgba(0,0,0,0.3);
      font-family: Arial, sans-serif; display: none; width: 200px;
    `;
    floatPopup.innerHTML = `
      <div style="margin-bottom: 10px; font-weight: bold; color: #333;">OCR Language</div>
      <div style="margin-bottom: 15px;">
        <input type="radio" id="mtOcrJa" name="mtOcrModel" value="ja" checked>
        <label for="mtOcrJa" style="font-size: 14px;">Japanese</label><br>
        <input type="radio" id="mtOcrKo" name="mtOcrModel" value="ko" style="margin-top: 5px;">
        <label for="mtOcrKo" style="font-size: 14px;">Korean</label>
      </div>
      <button id="mtStartBtn" style="width: 100%; padding: 10px; background: #28a745; color: white; border: none; border-radius: 4px; cursor: pointer; font-weight: bold;">Translate All</button>
    `;
    document.body.appendChild(floatPopup);

    document.getElementById('mtStartBtn').onclick = () => {
      const selectedModel = document.querySelector('input[name="mtOcrModel"]:checked').value;
      floatPopup.style.display = 'none';
      startTranslationProcess(selectedModel);
    };

    // Scroll listener to swap top button for floating button
    window.addEventListener('scroll', () => {
      if (window.scrollY > 50) {
        topBtn.style.opacity = '0';
        topBtn.style.pointerEvents = 'none';
        floatBtn.style.display = 'block';
      } else {
        topBtn.style.opacity = '1';
        topBtn.style.pointerEvents = 'auto';
        floatBtn.style.display = 'none';
        floatPopup.style.display = 'none';
      }
    });
  }

  function toggleFloatPopup() {
    if (floatPopup.style.display === 'block') {
      floatPopup.style.display = 'none';
    } else {
      floatPopup.style.display = 'block';
    }
  }

  injectUI();

  async function startTranslationProcess(selectedOcrModel) {
    if (isTranslating) return;
    isTranslating = true;
    floatPopup.style.display = 'none';

    const { serverUrl, ocrModel } = await chrome.storage.local.get(['serverUrl', 'ocrModel']);
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

    const overlay = createProgressOverlay(images.length, targetOcr);

    let processedCount = 0;
    for (const img of images) {
      updateOverlay(overlay, processedCount, images.length, img.src);
      
      // Add yellow border to currently working image
      img.style.outline = '4px solid yellow';
      img.style.outlineOffset = '-4px';

      try {
        await processImage(img, serverUrl);
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

    overlay.innerText = `✅ Translation Complete! (OCR: ${targetOcr === 'ja' ? 'Japanese' : 'Korean'})`;
    setTimeout(() => overlay.remove(), 3000);
    isTranslating = false;
  }

  function createSpinner(img) {
    const overlay = document.createElement('div');
    overlay.style.cssText = `
      position: absolute; 
      display: flex; align-items: center; justify-content: center;
      background: rgba(0,0,0,0.5); z-index: 9999; pointer-events: none;
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
      width: 40px; height: 40px; border: 5px solid #f3f3f3;
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
  async function processImage(img, serverUrl) {
    const targetSrc = img.dataset.mtTargetSrc;

    const fetchResponse = await chrome.runtime.sendMessage({ type: "fetchImage", url: targetSrc });
    if (!fetchResponse.success) throw new Error("Failed to fetch image");

    const submitResponse = await chrome.runtime.sendMessage({
      type: "submitImage",
      serverUrl: serverUrl,
      base64Data: fetchResponse.base64
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
  function createProgressOverlay(total, ocrModel) {
    const overlay = document.createElement('div');
    overlay.style.cssText = `
      position: fixed;
      top: 15px;
      right: 15px;
      z-index: 2147483647;
      padding: 15px 20px;
      background: rgba(0, 0, 0, 0.85);
      color: white;
      border-radius: 8px;
      font-family: Arial, sans-serif;
      font-size: 14px;
      box-shadow: 0 4px 10px rgba(0,0,0,0.5);
      min-width: 250px;
    `;
    const label = ocrModel === 'ko' ? 'Korean (PaddleOCR)' : 'Japanese (Hayai+YOLO)';
    overlay.innerText = `Starting translation of ${total} images... [OCR: ${label}]`;
    document.body.appendChild(overlay);
    return overlay;
  }

  function updateOverlay(overlay, current, total, currentSrc) {
    if (currentSrc) {
      const shortSrc = currentSrc.length > 40 ? currentSrc.substring(0, 37) + '...' : currentSrc;
      overlay.innerHTML = `
        <div style="margin-bottom: 10px; font-weight: bold;">Translating ${current + 1} / ${total}</div>
        <div style="font-size: 11px; color: #ccc; word-break: break-all;">${shortSrc}</div>
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