let ruleCounter = 1;
const IMAGE_FETCH_TIMEOUT_MS = 30000;
const JOB_HEALTH_ALARM = "mangaTranslatorJobHealth";
const JOB_HEALTH_INTERVAL_MINUTES = 0.5;
const JOB_HEALTH_STORAGE_KEY = "mangaTranslatorActiveJobs";

async function getActiveJobs() {
  const stored = await chrome.storage.session.get(JOB_HEALTH_STORAGE_KEY);
  return stored[JOB_HEALTH_STORAGE_KEY] || {};
}

async function saveActiveJobs(jobs) {
  await chrome.storage.session.set({ [JOB_HEALTH_STORAGE_KEY]: jobs });
}

async function reportJobHealth(job, active) {
  try {
    const response = await fetch(`${job.serverUrl}/v1/health`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ job_id: job.jobId, active }),
    });
    if (!response.ok) {
      console.warn(`[BG] Health report for ${job.jobId} failed: HTTP ${response.status}`);
    }
  } catch (error) {
    console.warn(`[BG] Health report for ${job.jobId} failed:`, error);
  }
}

async function registerJobHealth(jobId, serverUrl, tabId) {
  const jobs = await getActiveJobs();
  jobs[jobId] = { jobId, serverUrl: String(serverUrl).replace(/\/$/, ""), tabId };
  await saveActiveJobs(jobs);
  await chrome.alarms.create(JOB_HEALTH_ALARM, { periodInMinutes: JOB_HEALTH_INTERVAL_MINUTES });
  await reportJobHealth(jobs[jobId], true);
}

async function stopJobHealth(jobId, active = false) {
  const jobs = await getActiveJobs();
  const job = jobs[jobId];
  if (!job) return;
  delete jobs[jobId];
  await saveActiveJobs(jobs);
  if (!active) await reportJobHealth(job, false);
  if (!Object.keys(jobs).length) await chrome.alarms.clear(JOB_HEALTH_ALARM);
}

async function stopJobsForTab(tabId) {
  const jobs = await getActiveJobs();
  const stopped = Object.values(jobs).filter(job => job.tabId === tabId);
  if (!stopped.length) return;
  for (const job of stopped) delete jobs[job.jobId];
  await saveActiveJobs(jobs);
  await Promise.all(stopped.map(job => reportJobHealth(job, false)));
  if (!Object.keys(jobs).length) await chrome.alarms.clear(JOB_HEALTH_ALARM);
}

chrome.alarms.onAlarm.addListener(async alarm => {
  if (alarm.name !== JOB_HEALTH_ALARM) return;
  for (const job of Object.values(await getActiveJobs())) {
    await reportJobHealth(job, true);
  }
});

chrome.tabs.onRemoved.addListener(tabId => {
  stopJobsForTab(tabId);
});

chrome.tabs.onUpdated.addListener((tabId, changeInfo) => {
  if (changeInfo.url) stopJobsForTab(tabId);
});

chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
  if (request.type === "fetchImage") {
    const url = request.url;
    const pageUrl = sender.tab?.url;
    const tabId = sender.tab?.id;
    if (!url || !pageUrl || tabId === undefined) {
      sendResponse({ success: false, error: "Missing image URL or sender tab" });
      return;
    }

    console.log("[BG] Fetching image for translation:", url);
    fetchImagePowerful(url, pageUrl, tabId)
      .then(base64 => {
        console.log("[BG] Image ready for backend submission:", url);
        sendResponse({ success: true, base64: base64 });
      })
      .catch(err => {
        console.error("[BG] Image fetch failed:", url, err);
        sendResponse({ success: false, error: err.toString() });
      });

    return true; // Keep channel open for async
  }
  if (request.type === "submitImage") {
    submitTranslation(request, sendResponse, sender.tab?.id);
    return true;
  }

  if (request.type === "stopTranslationHealth") {
    stopJobHealth(request.jobId, request.active === true)
      .then(() => sendResponse({ success: true }));
    return true;
  }

  if (request.type === "translateImageUrl") {
    const imageUrl = request.imageUrl;
    const pageUrl = sender.tab?.url || request.pageUrl;
    const tabId = sender.tab?.id;
    if (!imageUrl || !pageUrl || tabId === undefined) {
      sendResponse({ success: false, error: "Missing image URL or source tab for translation" });
      return;
    }
    console.log("[BG] Fetching and submitting image in one request path:", imageUrl);
    fetchImagePowerful(imageUrl, pageUrl, tabId)
      .then(base64Data => submitTranslation({ ...request, base64Data }, sendResponse, tabId))
      .catch(err => {
        console.error("[BG] Image fetch failed before /v1/translate:", imageUrl, err);
        sendResponse({ success: false, error: err.toString() });
      });
    return true;
  }
});

function submitTranslation(request, sendResponse, senderTabId) {
  const { serverUrl, base64Data, colorize, targetLang, ocrLang, combineAmount,
          contextMode, contextLevel, styleAware, styleFonts } = request;
  const endpoint = `${String(serverUrl || '').replace(/\/$/, '')}/v1/translate`;
    console.log("[BG] submitImage received; preparing request:", endpoint, {
      hasBase64: typeof base64Data === 'string' && base64Data.length > 0,
      base64Length: typeof base64Data === 'string' ? base64Data.length : 0,
    });

    if (!serverUrl || typeof base64Data !== 'string' || !base64Data.includes(',')) {
      sendResponse({ success: false, error: "submitImage missing serverUrl or valid base64 image data" });
      return;
    }

    try {
      // Convert Base64 back to Blob for FormData
      const encodedImage = base64Data.split(',')[1];
      const byteString = atob(encodedImage);
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
      const requestedOcrLang = ocrLang || "ja";
      formData.append(
        "ocr_lang",
        requestedOcrLang === (targetLang || "en") ? "auto" : requestedOcrLang
      );
      formData.append("colorize", colorize ? "true" : "false");
      if (contextMode === 'on') {
        formData.append("skip_sfx", "true");
        formData.append("context_aware", "true");
        formData.append("context_level", contextLevel === 'high' ? 'high' : 'low');
      }
      if (combineAmount && combineAmount > 1) formData.append("combine_amount", String(combineAmount));
      if (styleAware) formData.append("style_aware", "true");
      formData.append("style_fonts", JSON.stringify(styleFonts || {}));

      console.log("[BG] Sending POST /v1/translate:", endpoint);
      fetch(endpoint, { method: "POST", body: formData })
        .then(async res => {
          const data = await res.json().catch(() => ({}));
          if (!res.ok) throw new Error(`HTTP ${res.status}: ${data.detail || JSON.stringify(data)}`);
          return data;
        })
        .then(async data => {
          console.log("[BG] /v1/translate response:", data);
          if (data.job_id) {
            if (senderTabId !== undefined) {
              await registerJobHealth(data.job_id, serverUrl, senderTabId);
            }
            // Return immediately. The page content script owns the long poll;
            // an MV3 service worker may be suspended during local GGUF work.
            sendResponse({ success: true, pending: true, job_id: data.job_id });
          } else sendResponse({ success: false, error: "No job ID returned" });
        })
        .catch(err => {
          console.error("[BG] /v1/translate failed:", err);
          sendResponse({ success: false, error: err.toString() });
        });
    } catch (err) {
      console.error("[BG] submitImage preparation failed before /v1/translate:", err);
      sendResponse({ success: false, error: err.toString() });
    }

    return true;
  }

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
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), IMAGE_FETCH_TIMEOUT_MS);
    let res;
    try {
      res = await fetch(url, {
        credentials: "include",
        referrer: pageUrl,
        signal: controller.signal,
      });
    } finally {
      clearTimeout(timeoutId);
    }
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
  const maxAttempts = 450; // Local vision OCR + GGUF translation can take up to 15 minutes.

  const poll = () => {
    fetch(`${serverUrl}/v1/translate/${jobId}`)
      .then(async res => {
        const data = await res.json().catch(() => ({}));
        if (!res.ok) throw new Error(`Job poll failed: HTTP ${res.status}: ${data.detail || JSON.stringify(data)}`);
        return data;
      })
      .then(data => {
        if (data.status === "completed") {
          console.log(`[BG] Job ${jobId} completed; requesting rendered image.`);
          fetchFinalImage(serverUrl, jobId, sendResponse);
        } else if (data.status === "failed") {
          console.error(`[BG] Job ${jobId} failed:`, data.error);
          sendResponse({ success: false, error: data.error || "Server error", job_id: jobId });
        } else {
          attempts++;
          if (attempts < maxAttempts) {
            setTimeout(poll, 2000);
          } else {
            sendResponse({
              success: false,
              error: `Polling timeout after 15 minutes for job ${jobId}`,
              job_id: jobId,
            });
          }
        }
      })
      .catch(err => {
        attempts++;
        if (attempts < maxAttempts) {
          setTimeout(poll, 2000);
        } else {
          sendResponse({ success: false, error: err.toString(), job_id: jobId });
        }
      });
  };
  poll();
}

function fetchFinalImage(serverUrl, jobId, sendResponse) {
  fetch(`${serverUrl}/v1/translate/${jobId}/image`, { method: "POST" })
    .then(async res => {
      if (!res.ok) {
        const detail = await res.text().catch(() => "");
        throw new Error(`Image fetch failed: HTTP ${res.status}${detail ? `: ${detail.slice(0, 300)}` : ""}`);
      }
      return res.blob();
    })
    .then(blob => {
      if (!blob.size) throw new Error(`Rendered image for job ${jobId} was empty`);
      console.log(`[BG] Rendered image received for job ${jobId}: ${blob.size} bytes (${blob.type || "unknown type"}).`);
      const reader = new FileReader();
      reader.onloadend = () => {
        const dataUrl = typeof reader.result === "string" ? reader.result : "";
        const comma = dataUrl.indexOf(',');
        const base64 = comma >= 0 ? dataUrl.slice(comma + 1) : "";
        if (!base64) {
          sendResponse({ success: false, error: `Rendered image encoding failed for job ${jobId}`, job_id: jobId });
          return;
        }
        console.log(`[BG] Sending rendered image for job ${jobId} to the page (${base64.length} base64 chars).`);
        sendResponse({
          success: true,
          image_b64: base64,
          image_data_url: dataUrl,
          image_bytes: blob.size,
          job_id: jobId,
        });
      };
      reader.onerror = () => sendResponse({ success: false, error: "FileReader error", job_id: jobId });
      reader.readAsDataURL(blob);
    })
    .catch(err => {
      console.error(`[BG] Rendered image handoff failed for job ${jobId}:`, err);
      sendResponse({ success: false, error: err.toString(), job_id: jobId });
    });
}