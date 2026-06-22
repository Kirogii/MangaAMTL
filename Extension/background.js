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
    const { serverUrl, base64Data, colorize, targetLang } = request;
    
    // Convert Base64 back to Blob for FormData
    const byteString = atob(base64Data.split(',')[1]);
    const arrayBuffer = new ArrayBuffer(byteString.length);
    const uint8Array = new Uint8Array(arrayBuffer);
    for (let i = 0; i < byteString.length; i++) {
      uint8Array[i] = byteString.charCodeAt(i);
    }
    const blob = new Blob([uint8Array], { type: "image/png" });

    const formData = new FormData();
    formData.append("file", blob, "manga_page.png");
    formData.append("use_lama", "true");
    formData.append("lang", targetLang || "en"); // Send target language to backend
    formData.append("colorize", colorize ? "true" : "false"); // Pass colorize flag

    // Step 1: Upload
    fetch(`${serverUrl}/v1/translate/upload`, {
      method: "POST",
      body: formData
    })
    .then(res => res.json())
    .then(data => {
      if (data.id) {
        // Step 2: Poll for result
        pollTranslation(serverUrl, data.id, sendResponse);
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
    fetch(`${serverUrl}/v1/translate/${jobId}?wait=10`)
      .then(res => res.json())
      .then(data => {
        if (data.status === "done") {
          sendResponse({ success: true, image_b64: data.image_b64 });
        } else if (data.status === "error") {
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