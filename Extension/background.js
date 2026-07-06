let ruleCounter = 1;

chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
  if (request.type === "fetchImage") {
    const url = request.url;
    const pageUrl = sender.tab.url;
    const tabId = sender.tab.id;

    fetchImagePowerful(url, pageUrl, tabId)
      .then(base64 => sendResponse({ success: true, base64: base64 }))
      .catch(err => sendResponse({ success: false, error: err.toString() }));
      
    return true; // Keep channel open for async
  }

  if (request.type === "submitImage") {
    const { serverUrl, base64Data, colorize, targetLang, ocrLang } = request;

    // Convert Base64 back to Blob for FormData
    const byteString = atob(base64Data.split(',')[1]);
    const arrayBuffer = new ArrayBuffer(byteString.length);
    const uint8Array = new Uint8Array(arrayBuffer);
    for (let i = 0; i < byteString.length; i++) {
      uint8Array[i] = byteString.charCodeAt(i);
    }
    
    const mimeMatch = base64Data.match(/data:(.*?);base64,/);
    const mimeString = mimeMatch ? mimeMatch[1] : "image/png";
    const blob = new Blob([uint8Array], { type: mimeString });

    const formData = new FormData();
    formData.append("image", blob, "manga_page.png");
    formData.append("target_lang", targetLang || "en");
    formData.append("ocr_lang", ocrLang || "ja");
    formData.append("colorize", colorize ? "true" : "false");

    // Step 1: Create translation job
    fetch(`${serverUrl}/v1/translate`, {
      method: "POST",
      body: formData
    })
    .then(res => res.json())
    .then(data => {
      if (data.job_id) {
        // Step 2: Poll job status
        pollTranslation(serverUrl, data.job_id, sendResponse);
      } else {
        sendResponse({ success: false, error: "No job ID returned" });
      }
    })
    .catch(err => sendResponse({ success: false, error: err.toString() }));

    return true; // Keep channel open for async
  }
});

async function fetchImagePowerful(url, pageUrl, tabId) {
  // --- METHOD 1: Canvas Extraction (Zero network requests, bypasses all network security) ---
  try {
    const canvasResult = await chrome.scripting.executeScript({
      target: { tabId },
      func: (imgUrl) => {
        return new Promise((resolve) => {
          // Look for the image on the page using various attributes
          const img = document.querySelector(`img[src="${imgUrl}"], img[data-original="${imgUrl}"], img[data-src="${imgUrl}"], img[o_src="${imgUrl}"]`);
          if (!img || !img.complete || img.naturalWidth === 0) return resolve(null);
          
          try {
            const canvas = document.createElement('canvas');
            canvas.width = img.naturalWidth;
            canvas.height = img.naturalHeight;
            const ctx = canvas.getContext('2d');
            ctx.drawImage(img, 0, 0);
            resolve(canvas.toDataURL('image/png'));
          } catch (e) {
            resolve(null); // Canvas is tainted (CORS), fallback to Method 2
          }
        });
      },
      args: [url]
    });

    if (canvasResult && canvasResult[0] && canvasResult[0].result) {
      console.log("[BG] Image grabbed via Canvas (No network request needed)");
      return canvasResult[0].result;
    }
  } catch (e) {
    console.log("[BG] Canvas method failed, trying network spoofing...");
  }

  // --- METHOD 2: Network Fetch + Header Spoofing (Bypasses Hotlink Protection) ---
  const ruleId = ruleCounter++;
  const escapedUrl = url.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
  
  // Modify headers at the browser network level to bypass security
  const rule = {
    id: ruleId,
    priority: 1,
    action: {
      type: "modifyHeaders",
      requestHeaders: [
        { header: "Referer", operation: "set", value: pageUrl }, // Pretend we are the webpage
        { header: "Origin", operation: "remove" }                // Strip extension origin
      ]
    },
    condition: {
      regexFilter: "^" + escapedUrl + "$",
      resourceTypes: ["xmlhttprequest"]
    }
  };

  try {
    // Apply the header spoofing rule
    await chrome.declarativeNetRequest.updateDynamicRules({
      addRules: [rule],
      removeRuleIds: [ruleId]
    });
  } catch (e) {
    console.error("[BG] Failed to set spoofing rule:", e);
  }

  try {
    const res = await fetch(url);
    if (!res.ok) {
      const text = await res.text();
      throw new Error(`HTTP ${res.status}: ${text.substring(0, 100)}`);
    }
    
    const contentType = res.headers.get('content-type') || '';
    if (!contentType.startsWith('image/')) {
      const text = await res.text();
      throw new Error(`Expected image but got ${contentType}. Site blocked download. Body: ${text.substring(0, 100)}`);
    }

    const blob = await res.blob();
    
    return await new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onloadend = () => resolve(reader.result);
      reader.onerror = () => reject(new Error("FileReader error"));
      reader.readAsDataURL(blob);
    });
  } finally {
    // Clean up rule to prevent memory leaks
    await chrome.declarativeNetRequest.updateDynamicRules({
      removeRuleIds: [ruleId]
    });
  }
}

function pollTranslation(serverUrl, jobId, sendResponse) {
  let attempts = 0;
  const maxAttempts = 60; // Timeout after ~2 minutes

  const poll = () => {
    fetch(`${serverUrl}/v1/translate/${jobId}`)
      .then(res => res.json())
      .then(data => {
        if (data.status === "completed") {
          // Step 3: Fetch the rendered image once the job is done
          fetchFinalImage(serverUrl, jobId, sendResponse);
        } else if (data.status === "failed") {
          sendResponse({ success: false, error: data.error || "Server error" });
        } else {
          attempts++;
          if (attempts < maxAttempts) {
            setTimeout(poll, 2000); // Wait 2s before polling again
          } else {
            sendResponse({ success: false, error: "Polling timeout" });
          }
        }
      })
      .catch(err => {
        attempts++;
        if (attempts < maxAttempts) {
          setTimeout(poll, 2000);
        } else {
          sendResponse({ success: false, error: err.toString() });
        }
      });
  };
  poll();
}

function fetchFinalImage(serverUrl, jobId, sendResponse) {
  fetch(`${serverUrl}/v1/translate/${jobId}/image`, { method: "POST" })
    .then(res => {
      if (!res.ok) throw new Error(`Image fetch failed: HTTP ${res.status}`);
      return res.blob();
    })
    .then(blob => {
      const reader = new FileReader();
      reader.onloadend = () => {
        const base64 = reader.result.split(',')[1];
        sendResponse({ success: true, image_b64: base64 });
      };
      reader.onerror = () => sendResponse({ success: false, error: "FileReader error" });
      reader.readAsDataURL(blob);
    })
    .catch(err => sendResponse({ success: false, error: err.toString() }));
}