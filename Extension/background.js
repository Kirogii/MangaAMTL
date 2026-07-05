// Fetch image as Base64 to bypass CORS
chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
  if (request.type === "fetchImage") {
    fetch(request.url)
      .then(res => res.blob())
      .then(blob => {
        const reader = new FileReader();
        reader.onloadend = () => sendResponse({ success: true, base64: reader.result });
        reader.onerror = () => sendResponse({ success: false, error: "FileReader error" });
        reader.readAsDataURL(blob);
      })
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
    const blob = new Blob([uint8Array], { type: "image/png" });

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
        // reader.result is a full data URL: "data:image/png;base64,...."
        const base64 = reader.result.split(',')[1];
        sendResponse({ success: true, image_b64: base64 });
      };
      reader.onerror = () => sendResponse({ success: false, error: "FileReader error" });
      reader.readAsDataURL(blob);
    })
    .catch(err => sendResponse({ success: false, error: err.toString() }));
}
