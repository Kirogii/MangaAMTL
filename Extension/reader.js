// =============================================================================
// MangaAMTL Reader — fullscreen manga reader injected by content.js
// =============================================================================
// Modes:
//   - webtoon:  vertical scroll, all pages stacked. Default.
//   - two-page: book spread with CSS-3D curvature, page-flip animation,
//     crinkle sound, browser-native high-quality upscaling, and a soft
//     center crease that blends into the spine.
//
// Input:
//   - Touch: swipe down from top = exit; tap center = settings;
//     swipe left/right = next/prev (two-page); drag a page to flip it.
//   - Mouse (PC): ‹ › buttons, ☰ settings, drag a page to flip, ← → arrows,
//     Esc to exit. Click outside the settings panel to close it.
//
// Upscaling: two-page mode renders each page through a canvas with
// imageSmoothingQuality='high' so low-res sources look cleaner at fullscreen.
// Aspect ratio is always preserved; the spread fits the viewport with no
// scrolling.
// =============================================================================
(function () {
  'use strict';

  if (window.__mtReaderInit) return;
  window.__mtReaderInit = true;

  const IS_TOUCH = ('ontouchstart' in window) || navigator.maxTouchPoints > 0;
  const REDUCED_MOTION = window.matchMedia &&
    window.matchMedia('(prefers-reduced-motion: reduce)').matches;

  let overlay = null;
  let stage = null;
  let progressBar = null;
  let progressLabel = null;
  let controlsBar = null;
  let pages = [];          // [{ src, imgEl, upscaledDataUrl }]
  let mode = 'webtoon';    // 'webtoon' | 'two-page' — restored from storage on open
  let currentSpread = 0;   // index of the RIGHT page of the current spread
  let controlsVisible = false;
  let controlsHideTimer = null;
  let lastScrollTop = 0;   // tracks scroll position for direction-based auto-hide
  let centerSettingsBtn = null;
  // Set when the webtoon scroll-down gesture hid the gear, so later visibility
  // recalcs don't bring it back until the user scrolls up again.
  let gearScrollHidden = false;
  let soundOn = false;      // restored from storage on open
  let autoTranslate = false; // per-book auto-translate toggle
  let chapterStartIndex = 0; // page index at the start of the current chapter (for per-chapter progress)

  // Per-page upscale cache so we only pay the canvas cost once per page.
  const upscaleCache = new Map(); // src -> dataUrl

  // ── Audio (page flip sound from sounds/pageflip.mp3) ──────────────────
  // Cache the URL once — chrome.runtime.getURL is cheap but calling it on
  // every flip is unnecessary.
  let _pageFlipUrl = null;
  function getPageFlipUrl() {
    if (!_pageFlipUrl) {
      try { _pageFlipUrl = chrome.runtime.getURL('sounds/pageflip.mp3'); }
      catch (e) { _pageFlipUrl = null; }
    }
    return _pageFlipUrl;
  }
  function playCrinkle() {
    if (!soundOn || REDUCED_MOTION) return;
    const url = getPageFlipUrl();
    if (!url) { console.warn('[Reader] pageflip.mp3 URL unavailable'); return; }
    // Create a fresh Audio element per flip. Cloning can lose the src in some
    // content-script contexts; a fresh Audio(src) is reliable.
    try {
      const audio = new Audio(url);
      audio.volume = 0.6;
      const playPromise = audio.play();
      if (playPromise && typeof playPromise.catch === 'function') {
        playPromise.catch((e) => {
          console.warn('[Reader] pageflip.mp3 play failed:', e.name, e.message);
        });
      }
      // Free the element once playback ends so we don't leak elements.
      audio.addEventListener('ended', () => { audio.src = ''; }, { once: true });
      audio.addEventListener('error', () => { /* already warned if play() rejects */ }, { once: true });
    } catch (e) {
      console.warn('[Reader] pageflip.mp3 could not be created:', e);
    }
  }

  // ── Browser-native upscaling (two-page only) ──────────────────────────
  // Draw the source image onto a canvas at 2x its natural size with
  // imageSmoothingQuality='high', then read back as a data URL. The result
  // is cached per-src. For sources that are already large, this is a no-op.
  function upscaleImage(src) {
    return new Promise((resolve) => {
      if (upscaleCache.has(src)) return resolve(upscaleCache.get(src));
      const img = new Image();
      img.crossOrigin = 'anonymous';
      img.onload = () => {
        const w = img.naturalWidth, h = img.naturalHeight;
        // Only upscale if the image is small enough that fullscreen would look soft.
        // Target: at least 2000px on the long edge — each page box is up to
        // 50vw × 100vh on high-DPI displays, so we want plenty of headroom.
        const targetLong = 2000;
        const longEdge = Math.max(w, h);
        if (longEdge >= targetLong) {
          // Already high-res — just use the original.
          upscaleCache.set(src, src);
          return resolve(src);
        }
        const scale = targetLong / longEdge;
        const cw = Math.round(w * scale);
        const ch = Math.round(h * scale);
        const canvas = document.createElement('canvas');
        canvas.width = cw; canvas.height = ch;
        const cctx = canvas.getContext('2d');
        cctx.imageSmoothingEnabled = true;
        cctx.imageSmoothingQuality = 'high';
        cctx.drawImage(img, 0, 0, cw, ch);
        try {
          const dataUrl = canvas.toDataURL('image/jpeg', 0.92);
          upscaleCache.set(src, dataUrl);
          resolve(dataUrl);
        } catch (e) {
          // CORS-tainted canvas — fall back to the original src.
          upscaleCache.set(src, src);
          resolve(src);
        }
      };
      img.onerror = () => { upscaleCache.set(src, src); resolve(src); };
      img.src = src;
    });
  }

  // Track MutationObservers so we can disconnect them on exit.
  let srcObservers = [];

  function disconnectObservers() {
    srcObservers.forEach(o => o.disconnect());
    srcObservers = [];
  }

  // ── Open ──────────────────────────────────────────────────────────────
  window.__mtOpenReader = async function () {
    if (overlay) return;
    // Reset infinite-scroll chaining state so a fresh reader session doesn't
    // inherit stale next-chapter URLs from a previous visit.
    infiniteScrollLoading = false;
    infiniteScrollExhausted = false;
    _lastNextChapterUrl = null;
    _discoveredNextUrl = null;
    _chapterBoundaries = [];
    lastScrollTop = 0;
    if (!window.__mtFindTranslatableImages) {
      alert('Reader failed: image finder not available. Reload the page.');
      return;
    }
    // includeTranslated: already-translated pages must still be listed here or
    // they vanish from the reader on re-entry. The finder's size/pixel gates
    // exist to choose translation candidates and would otherwise reject the
    // overlaid results, which come back as smaller inline data: URLs.
    let imgs = window.__mtFindTranslatableImages({ includeTranslated: true });
    if (!imgs || imgs.length === 0) {
      // Nothing yet — the site may still be booting its reader (MangaDex shows
      // its logo for a while) or drip-feeding pages. Poll instead of failing.
      const waitUi = showWaitingOverlay();
      imgs = await waitForPageImages({ ui: waitUi });
      waitUi.remove();
      if (imgs === null) return;  // user cancelled
      if (!imgs || imgs.length === 0) {
        alert('No manga images detected on this page. The site may still be loading — try again in a moment.');
        return;
      }
    }

    // Show a loading screen while we wait for every image to finish loading.
    // The reader can't size pages correctly without knowing their natural
    // dimensions, and we don't want partially-loaded spreads flashing.
    pages = imgs.map(img => ({
      src: img.dataset.mtTargetSrc || img.src,
      el: null,
      domImg: img,  // keep a ref to the live DOM <img> so we can detect translation updates
    }));

    const loadingOverlay = showLoadingOverlay(imgs.length);

    // Wait for every image's source to finish loading (or fail).
    await Promise.all(imgs.map(img => waitForImageLoad(img)));

    // Restore saved mode + sound preference.
    const stored = await chrome.storage.local.get(['mtReaderMode', 'mtReaderSound', 'mtAutoTranslate_' + window.location.pathname]);
    if (stored.mtReaderMode === 'two-page' || stored.mtReaderMode === 'webtoon') {
      mode = stored.mtReaderMode;
    }
    soundOn = stored.mtReaderSound === true;
    autoTranslate = stored['mtAutoTranslate_' + window.location.pathname] === true;

    loadingOverlay.remove();
    buildReaderDom();
    renderMode();

    // On mobile, request real fullscreen so the browser's address bar + bottom
    // nav are hidden and the manga gets the entire screen. This is a no-op on
    // desktop (where the overlay already covers the viewport).
    requestFullscreenOnMobile();

    // Watch each original DOM image for src changes (translation updates the
    // src + sets data-mt-translated). When that happens, refresh the reader's
    // copy of that page so the translated image is displayed.
    watchForTranslationUpdates();

    // Show the tap-region guide on first open (mobile only). Dismissed by
    // tapping anywhere. "Show Guide" button in settings re-opens it.
    showTapGuideIfFirstTime();

    sessionStorage.setItem('mtReaderActive', '1');
    console.log(`[Reader] Opened with ${pages.length} pages, mode=${mode}, sound=${soundOn}, autoTranslate=${autoTranslate}`);

    // If auto-translate is on for this book, trigger translation after a
    // short delay so images have settled.
    if (autoTranslate) {
      console.log('[Reader] Auto-translate is ON — triggering translation.');
      setTimeout(() => triggerTranslate(), 500);
    }
  };

  // ── Mobile fullscreen ─────────────────────────────────────────────────
  // Uses the Fullscreen API to hide the browser chrome (address bar, bottom
  // nav bar) so the manga overlay gets the true full screen on phones. On
  // desktop this is skipped — the fixed overlay already covers everything.
  // Must be called from a user gesture (the book-icon click) which is why it
  // lives in the open path, not a setTimeout.
  let didEnterFullscreen = false;
  function requestFullscreenOnMobile() {
    if (!IS_TOUCH) return; // desktop: the overlay is enough
    // Request fullscreen on document.documentElement (the page root) instead
    // of the overlay — mobile browsers are more permissive with the root
    // element than with extension-injected content-script elements.
    const el = document.documentElement;
    if (!el) return;
    _doFullscreenRequest(el);
    document.addEventListener('fullscreenchange', onFullscreenChange);
    document.addEventListener('webkitfullscreenchange', onFullscreenChange);
  }
  function _doFullscreenRequest(el) {
    const fn = el.requestFullscreen || el.webkitRequestFullscreen || el.webkitRequestFullScreen;
    if (fn) {
      try {
        const result = fn.call(el);
        if (result && typeof result.then === 'function') {
          result.then(() => { didEnterFullscreen = true; }).catch(() => { didEnterFullscreen = false; });
        } else {
          didEnterFullscreen = true; // legacy sync API
        }
      } catch (e) {
        didEnterFullscreen = false;
        console.warn('[Reader] Fullscreen request failed:', e);
      }
    }
    // Fallback for iOS Safari (no Fullscreen API on element): use the
    // viewport meta + scroll-to-zero trick to at least hide the address bar.
    if (!didEnterFullscreen) {
      try { window.scrollTo(0, 0); } catch (e) {}
    }
  }
  function onFullscreenChange() {
    const isFs = document.fullscreenElement || document.webkitFullscreenElement;
    if (!isFs && overlay) {
      // Fullscreen was lost while the reader is still open — re-request it
      // so the reader forces fullscreen until the user exits via Exit.
      // Small delay to avoid fighting the browser's own transition.
      setTimeout(() => {
        if (overlay) _doFullscreenRequest(overlay);
      }, 300);
    }
  }
  function exitFullscreenIfActive() {
    if (!didEnterFullscreen) return;
    // Remove the re-request watcher before exiting so we don't re-grab
    // fullscreen while the reader is tearing down.
    document.removeEventListener('fullscreenchange', onFullscreenChange);
    document.removeEventListener('webkitfullscreenchange', onFullscreenChange);
    const fn = document.exitFullscreen || document.webkitExitFullscreen || document.webkitCancelFullScreen;
    if (fn) {
      try { fn.call(document); } catch (e) {}
    }
    didEnterFullscreen = false;
  }

  // ── Wait for an <img> to finish loading its current src ───────────────
  function waitForImageLoad(img) {
    return new Promise((resolve) => {
      // Already complete (cached)?
      if (img.complete && img.naturalWidth > 0) return resolve();
      if (img.complete && img.naturalWidth === 0) {
        // Failed to load — resolve anyway so we don't block forever.
        return resolve();
      }
      let done = false;
      const finish = () => { if (!done) { done = true; resolve(); } };
      img.addEventListener('load', finish, { once: true });
      img.addEventListener('error', finish, { once: true });
      // Safety timeout: don't wait more than 15s per image.
      setTimeout(finish, 15000);
    });
  }

  // ── Client-side image caching ──────────────────────────────────────────
  // Stores fetched page image data URLs in chrome.storage.local so repeat
  // visits are instant and work offline. Keyed by source URL. Keeps only the
  // last 50 cached images to avoid hitting the storage quota.
  const IMG_CACHE_PREFIX = 'mtImgCache_';
  const IMG_CACHE_MAX = 50;

  async function getCachedImage(src) {
    return new Promise((resolve) => {
      chrome.storage.local.get(IMG_CACHE_PREFIX + src, (data) => {
        resolve(data[IMG_CACHE_PREFIX + src] || null);
      });
    });
  }
  async function setCachedImage(src, dataUrl) {
    if (!src || src.startsWith('data:')) return;
    const key = IMG_CACHE_PREFIX + src;
    chrome.storage.local.set({ [key]: dataUrl });
    // Trim cache to IMG_CACHE_MAX entries.
    chrome.storage.local.get(null, (all) => {
      const cacheKeys = Object.keys(all).filter(k => k.startsWith(IMG_CACHE_PREFIX));
      if (cacheKeys.length > IMG_CACHE_MAX) {
        const toRemove = cacheKeys.slice(0, cacheKeys.length - IMG_CACHE_MAX);
        chrome.storage.local.remove(toRemove);
      }
    });
  }

  // ── Loading overlay ───────────────────────────────────────────────────
  function showLoadingOverlay(count) {
    const lo = document.createElement('div');
    lo.style.cssText = `
      position: fixed; inset: 0; z-index: 2147483647;
      background: #000; color: #fff;
      display: flex; flex-direction: column; align-items: center; justify-content: center;
      font-family: Arial, sans-serif;
    `;
    lo.innerHTML = `
      <div style="margin-bottom: 16px;">
        <svg viewBox="0 0 24 24" width="40" height="40" fill="none" stroke="#22a552" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="animation: mt-r-spin 1s linear infinite;">
          <path d="M21 12a9 9 0 1 1-6.219-8.56"/>
        </svg>
      </div>
      <div style="font-size: 14px; color: #ccc;">Loading ${count} pages…</div>
    `;
    // Inject the spin keyframes if not already present.
    if (!document.getElementById('mt-reader-spin-style')) {
      const s = document.createElement('style');
      s.id = 'mt-reader-spin-style';
      s.textContent = `@keyframes mt-r-spin { to { transform: rotate(360deg); } }`;
      document.head.appendChild(s);
    }
    document.body.appendChild(lo);
    return lo;
  }

  // ── Waiting for slow / protected sites ────────────────────────────────
  // Sites like MangaDex render an app shell first (the spinning logo) and only
  // fetch the chapter pages afterwards, sometimes minutes later behind rate
  // limiting. Bailing on the first empty finder call gives a false "no images"
  // error, so instead we poll until the page count stops growing.
  const SITE_WAIT_TIMEOUT_MS = 180000;
  const SITE_WAIT_POLL_MS = 500;
  const SITE_WAIT_SETTLE_MS = 2000;

  // Standalone waiting UI: toast() lives inside the reader overlay, which does
  // not exist yet at this point.
  function showWaitingOverlay() {
    const wo = document.createElement('div');
    wo.style.cssText = `
      position: fixed; inset: 0; z-index: 2147483647;
      background: #000; color: #fff;
      display: flex; flex-direction: column; align-items: center; justify-content: center;
      font-family: Arial, sans-serif;
    `;
    wo.innerHTML = `
      <div style="margin-bottom: 16px;">
        <svg viewBox="0 0 24 24" width="40" height="40" fill="none" stroke="#22a552" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="animation: mt-r-spin 1s linear infinite;">
          <path d="M21 12a9 9 0 1 1-6.219-8.56"/>
        </svg>
      </div>
      <div id="mt-wait-msg" style="font-size: 14px; color: #ccc;">Waiting for the chapter to load…</div>
      <div id="mt-wait-sub" style="font-size: 12px; color: #777; margin-top: 6px;"></div>
      <button id="mt-wait-cancel" style="
        margin-top: 22px; padding: 8px 20px; cursor: pointer;
        background: #1a1a1a; color: #ccc; border: 1px solid #333; border-radius: 6px;
        font-family: Arial, sans-serif; font-size: 13px;">Cancel</button>
    `;
    if (!document.getElementById('mt-reader-spin-style')) {
      const s = document.createElement('style');
      s.id = 'mt-reader-spin-style';
      s.textContent = `@keyframes mt-r-spin { to { transform: rotate(360deg); } }`;
      document.head.appendChild(s);
    }
    document.body.appendChild(wo);
    const msg = wo.querySelector('#mt-wait-msg');
    const sub = wo.querySelector('#mt-wait-sub');
    const api = {
      el: wo,
      cancelled: false,
      setText(main, secondary) {
        if (main != null) msg.textContent = main;
        if (secondary != null) sub.textContent = secondary;
      },
      remove() { wo.remove(); },
    };
    wo.querySelector('#mt-wait-cancel').addEventListener('click', () => { api.cancelled = true; });
    return api;
  }

  function waitForDocumentReady(deadline) {
    return new Promise((resolve) => {
      const check = () => {
        if (document.readyState === 'complete' || Date.now() >= deadline) return resolve();
        setTimeout(check, 200);
      };
      check();
    });
  }

  // Polls the finder until the detected page count holds steady for settleMs
  // (lazy-loaded chapters trickle in) or the deadline passes. Returns the best
  // set seen, which may be empty if the site never produced pages.
  async function waitForPageImages(opts = {}) {
    const timeoutMs = opts.timeoutMs != null ? opts.timeoutMs : SITE_WAIT_TIMEOUT_MS;
    const settleMs = opts.settleMs != null ? opts.settleMs : SITE_WAIT_SETTLE_MS;
    const ui = opts.ui || null;
    const started = Date.now();
    const deadline = started + timeoutMs;

    if (ui) ui.setText('Waiting for the site to load…', null);
    await waitForDocumentReady(deadline);

    let best = [];
    let stableSince = Date.now();

    while (Date.now() < deadline) {
      if (ui && ui.cancelled) return null;
      const found = window.__mtFindTranslatableImages
        ? (window.__mtFindTranslatableImages({ includeTranslated: true }) || [])
        : [];

      if (found.length > best.length) {
        best = found;
        stableSince = Date.now();
      } else if (best.length > 0 && Date.now() - stableSince >= settleMs) {
        return best;
      }

      if (ui) {
        const secs = Math.round((Date.now() - started) / 1000);
        ui.setText(
          best.length > 0 ? 'Loading chapter pages…' : 'Waiting for the chapter to load…',
          `${best.length} page${best.length === 1 ? '' : 's'} found — ${secs}s`
        );
      }
      await _sleep(SITE_WAIT_POLL_MS);
    }
    return best;
  }

  // ── Watch for translation updates ─────────────────────────────────────
  // When translation finishes, content.js repoints img.src / mtTargetSrc and
  // sets data-mt-translated="true" on the original DOM images. We watch for
  // those changes and swap the corresponding reader page to the translated
  // result so it appears without requiring the user to reopen the reader.
  function watchForTranslationUpdates() {
    disconnectObservers();
    installTranslationListeners();
    pages.forEach((p, idx) => {
      if (!p.domImg) return;
      const observer = new MutationObserver(() => {
        // Prefer the live src: with combine groups the translated slice is a
        // data: URL written straight to src, and mtTargetSrc is updated to
        // match. Either way this differs from the pre-translation source.
        const newSrc = p.domImg.currentSrc || p.domImg.src || p.domImg.dataset.mtTargetSrc;
        if (!newSrc || newSrc === p.src) return;

        console.log(`[Reader] Page ${idx + 1} updated (translation complete) — refreshing.`);
        // Drop the stale upscale entry for the OLD src; it is keyed by source
        // URL and the page no longer points at it.
        upscaleCache.delete(p.src);
        p.src = newSrc;
        refreshPage(p, idx);
      });
      observer.observe(p.domImg, { attributes: true, attributeFilter: ['src', 'srcset', 'data-mt-translated'] });
      srcObservers.push(observer);
    });
  }

  // A MutationObserver only ever fires for the nodes it was attached to. A
  // translation started from INSIDE the reader (auto-translate, or the
  // Translate button) can therefore complete on images no observer is watching
  // — which is why translations used to appear only after exiting and
  // re-entering. content.js announces every applied result on the document, so
  // we reconcile the whole page list from those events instead of trusting
  // observer coverage.
  let translationListenersInstalled = false;
  function installTranslationListeners() {
    if (translationListenersInstalled) return;
    translationListenersInstalled = true;
    document.addEventListener('mt-image-translated', scheduleResync);
    document.addEventListener('mt-translation-complete', scheduleResync);
  }

  let resyncTimer = null;
  function scheduleResync() {
    if (!overlay) return;
    // Results land one page at a time; coalesce a burst into a single pass so
    // two-page mode re-renders once rather than per translated page.
    clearTimeout(resyncTimer);
    resyncTimer = setTimeout(resyncPages, 60);
  }

  function resyncPages() {
    if (!overlay) return;
    let changed = false;
    let spreadChanged = false;

    pages.forEach((p, idx) => {
      if (!p.domImg) return;
      const newSrc = p.domImg.dataset.mtTargetSrc || p.domImg.currentSrc || p.domImg.src;
      if (!newSrc || newSrc === p.src) return;

      upscaleCache.delete(p.src);
      p.src = newSrc;
      changed = true;
      if (mode === 'webtoon') {
        if (p.el && p.el.isConnected) p.el.src = newSrc;
      } else if (idx === currentSpread || idx === currentSpread + 1) {
        spreadChanged = true;
      }
    });

    if (!changed) return;
    console.log('[Reader] Resynced pages after translation update.');
    // Segments go green as translations land.
    refreshSeek(scrubbing ? scrubIdx : undefined);
    // renderTwoPage() rather than renderMode(): the latter resets the
    // infinite-scroll flags, which would let an exhausted chapter re-fetch.
    if (spreadChanged) renderTwoPage();
  }

  // Infinite-scroll pages are appended straight into the reader from the next
  // chapter's HTML, so they have no <img> on the host page and nothing for
  // content.js to find or translate. Give each one an off-screen stand-in that
  // lives OUTSIDE the reader overlay: the finder skips overlay-owned images,
  // but a proxy is a normal page image as far as it is concerned.
  //
  // Off-screen rather than hidden on purpose — isElementVisible() rejects
  // display:none / visibility:hidden / opacity, but not negative positioning.
  // The explicit size clears the finder's 200x200 rendered-box floor.
  function ensureProxyHost() {
    let host = document.getElementById('mt-reader-proxies');
    if (!host) {
      host = document.createElement('div');
      host.id = 'mt-reader-proxies';
      host.style.cssText = 'position:absolute; left:-100000px; top:0; width:1200px; pointer-events:none;';
      document.body.appendChild(host);
    }
    return host;
  }

  function ensureProxyImg(src) {
    const host = ensureProxyHost();
    const img = document.createElement('img');
    img.src = src;
    img.dataset.mtTargetSrc = src;
    img.style.cssText = 'display:block; width:1000px; height:auto; min-height:400px;';
    host.appendChild(img);
    return img;
  }

  function removeProxyHost() {
    const host = document.getElementById('mt-reader-proxies');
    if (host) host.remove();
  }

  // Swap a single page's rendered image in place instead of re-rendering the
  // whole view. A full renderMode() would reset webtoon scroll position and
  // interrupt an in-progress two-page flip, which is jarring while pages
  // finish translating one by one.
  function refreshPage(p, idx) {
    if (!overlay) return;

    if (mode === 'webtoon') {
      if (p.el && p.el.isConnected) p.el.src = p.src;
      return;
    }

    // Two-page: only the visible spread needs redrawing. Pages outside it
    // pick up the new src when the user flips to them. renderTwoPage() instead
    // of renderMode() so the infinite-scroll state isn't reset mid-chapter.
    if (idx === currentSpread || idx === currentSpread + 1) renderTwoPage();
  }

  // ── Tap-region guide overlay ──────────────────────────────────────────
  // Shows a translucent overlay highlighting the left/center/right tap zones.
  // On mobile it auto-shows the first time the reader opens; dismissed by
  // tapping anywhere. The "Guide" button in the settings panel re-opens it.
  function showTapGuideIfFirstTime() {
    const seen = localStorage.getItem('mtReaderGuideSeen');
    if (!seen && IS_TOUCH) {
      showTapGuide(false);
      localStorage.setItem('mtReaderGuideSeen', '1');
    }
  }
  function showTapGuide(forceShow) {
    if (!overlay) return;
    // Remove any existing guide.
    const existing = document.getElementById('mt-tap-guide');
    if (existing) existing.remove();
    const guide = document.createElement('div');
    guide.id = 'mt-tap-guide';
    guide.style.cssText = `
      position: absolute; inset: 0; z-index: 20;
      background: rgba(0,0,0,0.75); color: #fff;
      display: flex; pointer-events: auto;
      cursor: pointer; user-select: none;
    `;
    const zoneCss = `
      flex: 1; display: flex; flex-direction: column; align-items: center;
      justify-content: center; padding: 20px; text-align: center;
    `;
    const arrow = (svg) => `<div style="margin-bottom: 12px;">${svg}</div>`;
    guide.innerHTML = `
      <div style="${zoneCss}; border-right: 1px solid rgba(255,255,255,0.1);">
        ${arrow('<svg viewBox="0 0 24 24" width="48" height="48" fill="none" stroke="#22a552" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="15 18 9 12 15 6"/></svg>')}
        <div style="font-size: 16px; font-weight: 700; margin-bottom: 4px;">Tap Left</div>
        <div style="font-size: 12px; color: #aaa;">Previous page</div>
      </div>
      <div style="${zoneCss}; border-right: 1px solid rgba(255,255,255,0.1);">
        ${arrow('<svg viewBox="0 0 24 24" width="48" height="48" fill="none" stroke="#3a6ea5" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="3"/><path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 1 1-2.83 2.83l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-4 0v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 1 1-2.83-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1 0-4h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 1 1 2.83-2.83l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 4 0v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 1 1 2.83 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 0 4h-.09a1.65 1.65 0 0 0-1.51 1z"/></svg>')}
        <div style="font-size: 16px; font-weight: 700; margin-bottom: 4px;">Tap Center</div>
        <div style="font-size: 12px; color: #aaa;">Open settings</div>
      </div>
      <div style="${zoneCss}">
        ${arrow('<svg viewBox="0 0 24 24" width="48" height="48" fill="none" stroke="#22a552" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="9 18 15 12 9 6"/></svg>')}
        <div style="font-size: 16px; font-weight: 700; margin-bottom: 4px;">Tap Right</div>
        <div style="font-size: 12px; color: #aaa;">Next page</div>
      </div>
      <div style="position: absolute; bottom: 12px; left: 50%; transform: translateX(-50%); display: flex; flex-direction: column; align-items: center; gap: 4px; pointer-events: none; color: #666; font-size: 12px;">
        Tap anywhere to dismiss
      </div>
    `;
    // Tapping anywhere on the guide dismisses it.
    guide.onclick = (e) => { e.stopPropagation(); guide.remove(); };
    overlay.appendChild(guide);
  }

  // ── Build DOM (one-time) ──────────────────────────────────────────────
  function buildReaderDom() {
    overlay = document.createElement('div');
    overlay.id = 'mt-reader-overlay';
    overlay.style.cssText = `
      position: fixed; inset: 0; z-index: 2147483647;
      background: #000; color: #fff;
      display: flex; flex-direction: column;
      overscroll-behavior: contain;
      user-select: none; -webkit-user-select: none;
    `;

    stage = document.createElement('div');
    stage.id = 'mt-reader-stage';
    stage.style.cssText = `
      flex: 1; position: relative; overflow: hidden;
      width: 100%; height: 100%;
      perspective: 1800px;
    `;
    overlay.appendChild(stage);

    // Click outside settings → close settings.
    overlay.addEventListener('click', (e) => {
      if (controlsVisible && !controlsBar.contains(e.target)) {
        hideControls();
      }
    });

    // Progress bar — this IS the seek bar. At rest it's a plain 4px rail; on
    // hover/tap the same rail grows and sections off into per-page segments,
    // so the scrubber blends out of the progress bar instead of sitting above
    // it. buildSeekBar() fills in the bubble + segment track.
    seekWrap = document.createElement('div');
    seekWrap.id = 'mt-r-seek';
    seekWrap.style.cssText = `
      position: absolute; bottom: 0; left: 0; right: 0;
      height: 18px; background: rgba(0,0,0,0.6);
      display: flex; align-items: flex-end; padding: 0 8px 7px;
      gap: 8px; z-index: 8;
      transition: height 220ms cubic-bezier(0.22, 0.61, 0.36, 1),
                  padding-bottom 220ms cubic-bezier(0.22, 0.61, 0.36, 1),
                  background 220ms ease;
    `;
    // The rail is a shared surface: progressFill paints it while idle, the
    // segment track takes it over while seeking.
    progressBar = document.createElement('div');
    progressBar.style.cssText = `flex: 1; height: 4px; background: #333; border-radius: 2px; position: relative;`;
    const progressFill = document.createElement('div');
    progressFill.id = 'mt-reader-progress-fill';
    progressFill.style.cssText = `height: 100%; width: 0%; background: #22a552; border-radius: 2px; transition: width 0.2s, opacity 160ms ease;`;
    progressBar.appendChild(progressFill);
    progressLabel = document.createElement('div');
    progressLabel.style.cssText = `font-size: 11px; color: #ccc; min-width: 40px; text-align: right; line-height: 1;`;
    seekWrap.appendChild(progressBar);
    seekWrap.appendChild(progressLabel);
    overlay.appendChild(seekWrap);

    // Controls
    controlsBar = document.createElement('div');
    controlsBar.id = 'mt-reader-controls';
    controlsBar.style.cssText = `
      position: absolute; bottom: 26px; left: 50%; transform: translateX(-50%);
      background: rgba(20,20,31,0.95); border: 1px solid #3a3a4c; border-radius: 12px;
      padding: 10px 14px; display: none; gap: 8px; z-index: 7;
      backdrop-filter: blur(6px);
      max-width: 95vw; flex-wrap: wrap; justify-content: center;
    `;
    const btnCss = `padding: 10px 14px; background: #2a2a3c; color: #fff; border: 1px solid #4a4a5e; border-radius: 8px; cursor: pointer; font-size: 13px; font-weight: 600; display: flex; align-items: center; justify-content: center; gap: 6px;`;
    // White SVG icons for sound on/off (not emoji) — speaker with/without X.
    const SOUND_ON_ICON = `<svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="#fff" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polygon points="11 5 6 9 2 9 2 15 6 15 11 19 11 5"/><path d="M19.07 4.93a10 10 0 0 1 0 14.14M15.54 8.46a5 5 0 0 1 0 7.07"/></svg>`;
    const SOUND_OFF_ICON = `<svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="#fff" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polygon points="11 5 6 9 2 9 2 15 6 15 11 19 11 5"/><line x1="23" y1="9" x2="17" y2="15"/><line x1="17" y1="9" x2="23" y2="15"/></svg>`;
    // Auto-translate icon: green sparkle when ON, grey when OFF.
    const AUTO_ON_ICON = `<svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="#22a552" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="m12 3-1.9 5.8a2 2 0 0 1-1.3 1.3L3 12l5.8 1.9a2 2 0 0 1 1.3 1.3L12 21l1.9-5.8a2 2 0 0 1 1.3-1.3L21 12l-5.8-1.9a2 2 0 0 1-1.3-1.3z"/></svg><span style="color:#22a552;">Auto</span>`;
    const AUTO_OFF_ICON = `<svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="#888" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="m12 3-1.9 5.8a2 2 0 0 1-1.3 1.3L3 12l5.8 1.9a2 2 0 0 1 1.3 1.3L12 21l1.9-5.8a2 2 0 0 1 1.3-1.3L21 12l-5.8-1.9a2 2 0 0 1-1.3-1.3z"/></svg><span>Auto</span>`;
    controlsBar.innerHTML = `
      <button id="mt-r-translate" style="${btnCss}">Translate All</button>
      <button id="mt-r-webtoon" style="${btnCss}">Webtoon</button>
      <button id="mt-r-twopage" style="${btnCss}">Two-Page</button>
      <button id="mt-r-sound" style="${btnCss}">${soundOn ? SOUND_ON_ICON : SOUND_OFF_ICON}</button>
      <button id="mt-r-autotrans" style="${btnCss}">${autoTranslate ? AUTO_ON_ICON : AUTO_OFF_ICON}</button>
      <button id="mt-r-guide" style="${btnCss}">Guide</button>
      <button id="mt-r-exit" style="${btnCss}">Exit</button>
    `;
    overlay.appendChild(controlsBar);
    controlsBar.querySelector('#mt-r-translate').onclick = (e) => { e.stopPropagation(); triggerTranslate(); };
    controlsBar.querySelector('#mt-r-webtoon').onclick = (e) => {
      e.stopPropagation();
      switchMode('webtoon');
    };
    controlsBar.querySelector('#mt-r-twopage').onclick = (e) => {
      e.stopPropagation();
      switchMode('two-page');
    };
    controlsBar.querySelector('#mt-r-sound').onclick = (e) => {
      e.stopPropagation();
      soundOn = !soundOn;
      chrome.storage.local.set({ mtReaderSound: soundOn });
      // Swap the icon between speaker-on and speaker-off SVGs.
      e.currentTarget.innerHTML = soundOn ? SOUND_ON_ICON : SOUND_OFF_ICON;
    };
    controlsBar.querySelector('#mt-r-exit').onclick = (e) => { e.stopPropagation(); exitReader(); };
    controlsBar.querySelector('#mt-r-guide').onclick = (e) => { e.stopPropagation(); showTapGuide(true); };
    controlsBar.querySelector('#mt-r-autotrans').onclick = (e) => {
      e.stopPropagation();
      autoTranslate = !autoTranslate;
      e.currentTarget.innerHTML = autoTranslate ? AUTO_ON_ICON : AUTO_OFF_ICON;
      const bookKey = 'mtAutoTranslate_' + window.location.pathname;
      if (autoTranslate) {
        chrome.storage.local.set({ [bookKey]: true });
        triggerTranslate();
      } else {
        chrome.storage.local.set({ [bookKey]: false });
      }
    };

    // Center settings button (visible on ALL devices — PC and mobile). A
    // small gear icon sits at the bottom center. Tapping it opens the
    // controls bar. Auto-hides when scrolling down in webtoon mode and
    // reappears when scrolling up.
    centerSettingsBtn = document.createElement('button');
    centerSettingsBtn.innerHTML = `<svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="#fff" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="3"/><path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 1 1-2.83 2.83l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-4 0v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 1 1-2.83-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1 0-4h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 1 1 2.83-2.83l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 4 0v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 1 1 2.83 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 0 4h-.09a1.65 1.65 0 0 0-1.51 1z"/></svg>`;
    centerSettingsBtn.style.cssText = `
      position: absolute; bottom: 30px; left: 50%; transform: translateX(-50%);
      width: 40px; height: 40px; background: rgba(20,20,31,0.7); color: #fff;
      border: 1px solid #3a3a4c; border-radius: 8px; cursor: pointer;
      display: flex; align-items: center; justify-content: center; z-index: 6;
    `;
    centerSettingsBtn.onclick = (e) => { e.stopPropagation(); toggleControls(); };
    overlay.appendChild(centerSettingsBtn);
    updateGearVisibility();

    if (!IS_TOUCH) {
      document.addEventListener('keydown', onKeyDown);
    } else {
      wireTouchGestures();
    }

    // ── Tap zones (two-page mode only) ────────────────────────────────
    // Full-height invisible click areas: left 30% = prev, center 40% =
    // settings, right 30% = next. Only shown in two-page mode — in webtoon
    // mode the user needs free vertical scrolling, so these are hidden.
    const sideZoneW = Math.floor(window.innerWidth * 0.30);
    const centerW = Math.floor(window.innerWidth * 0.40);

    const leftZone = document.createElement('div');
    leftZone.id = 'mt-r-leftzone';
    leftZone.style.cssText = `
      position: absolute; left: 0; top: 0; bottom: 0;
      width: ${sideZoneW}px; z-index: 4; cursor: pointer; display: none;
    `;
    leftZone.onclick = (e) => { e.stopPropagation(); goPrev(); };
    overlay.appendChild(leftZone);

    const rightZone = document.createElement('div');
    rightZone.id = 'mt-r-rightzone';
    rightZone.style.cssText = `
      position: absolute; right: 0; top: 0; bottom: 0;
      width: ${sideZoneW}px; z-index: 4; cursor: pointer; display: none;
    `;
    rightZone.onclick = (e) => { e.stopPropagation(); goNext(); };
    overlay.appendChild(rightZone);

    const centerZone = document.createElement('div');
    centerZone.id = 'mt-r-centerzone';
    centerZone.style.cssText = `
      position: absolute; left: 50%; top: 0; bottom: 0;
      transform: translateX(-50%);
      width: ${centerW}px; z-index: 4; cursor: pointer; display: none;
    `;
    centerZone.onclick = (e) => { e.stopPropagation(); toggleControls(); };
    overlay.appendChild(centerZone);

    // Mouse wheel — page flip with cooldown. Attached to `overlay` on ALL
    // devices (touch laptops have mice too). A single wheel notch triggers
    // exactly one flip, then a cooldown blocks subsequent ticks. If the user
    // keeps scrolling aggressively (≥ BURST_THRESHOLD ticks within a burst
    // window), we lift the cooldown so each notch flips a page.
    overlay.addEventListener('wheel', onWheel, { passive: false });

    // Seek bar must be built before wireSeekInput() — the latter binds to the
    // track element.
    buildSeekBar();
    wireSeekInput();

    document.body.appendChild(overlay);
    document.body.style.overflow = 'hidden';

    // Re-flow the spread when the viewport changes so the two pages keep
    // covering half the screen each. Debounced — renderTwoPage is async and
    // we don't want to thrash during a drag-resize.
    let resizeTimer = null;
    window.addEventListener('resize', () => {
      clearTimeout(resizeTimer);
      resizeTimer = setTimeout(() => {
        if (overlay && mode === 'two-page') renderTwoPage();
      }, 150);
    });
  }

  // ── Render current mode ───────────────────────────────────────────────
  function renderMode() {
    stage.innerHTML = '';
    // Reset infinite-scroll state on mode switch so webtoon can re-trigger.
    infiniteScrollLoading = false;
    infiniteScrollExhausted = false;
    if (mode === 'webtoon') renderWebtoon();
    else renderTwoPage();
    // Show/hide tap zones based on mode. In webtoon mode the user needs free
    // vertical scrolling — tap zones would block touch events. In two-page
    // mode tap zones are the primary navigation method.
    const showZones = mode === 'two-page';
    const lz = document.getElementById('mt-r-leftzone');
    const rz = document.getElementById('mt-r-rightzone');
    const cz = document.getElementById('mt-r-centerzone');
    if (lz) lz.style.display = showZones ? 'block' : 'none';
    if (rz) rz.style.display = showZones ? 'block' : 'none';
    if (cz) cz.style.display = showZones ? 'block' : 'none';
    // Set touch-action on stage: none for two-page (we handle gestures), pan-y
    // for webtoon (let the browser scroll).
    stage.style.touchAction = (mode === 'two-page') ? 'manipulation' : 'pan-y';
    // The center tap zone above already opens the controls bar in two-page
    // mode, so the gear is redundant there and would sit on top of the spread.
    gearScrollHidden = false;
    updateGearVisibility();
    updateProgress();
    // renderWebtoon() throws away every p.el and builds fresh <img> elements,
    // so observers captured against the old ones are pointing at detached
    // nodes. Re-attach after every render to keep them 1:1 with `pages`.
    watchForTranslationUpdates();
  }

  // ── Mode switching with position preservation ─────────────────────────
  // Switching modes used to jump back to page 1 because webtoon tracks
  // position by scrollTop and two-page tracks it by `currentSpread` — neither
  // was derived from the other. Convert between them so the reader lands on
  // whatever page the user was actually looking at.
  function getCurrentPageIndex() {
    if (mode === 'two-page') return currentSpread;
    // Webtoon: the topmost page whose start has scrolled past the viewport top
    // is the one being read.
    const st = stage ? stage.scrollTop : 0;
    let idx = 0;
    for (let i = 0; i < pages.length; i++) {
      const el = pages[i].el;
      if (!el || !el.isConnected) continue;
      if (el.offsetTop <= st + 4) idx = i;
      else break;
    }
    return idx;
  }

  function restoreScrollToPage(idx) {
    if (!stage) return;
    const apply = () => {
      const p = pages[idx];
      if (!p || !p.el || !p.el.isConnected) return;
      const prev = stage.style.scrollBehavior;
      stage.style.scrollBehavior = 'auto';  // jump, don't animate a long scroll
      stage.scrollTop = p.el.offsetTop;
      stage.style.scrollBehavior = prev;
    };
    apply();
    // Images above the target finish loading after the first frame and push it
    // down, so re-apply a few times. Bail out if the user has already started
    // scrolling somewhere else — their input wins over our restore.
    let last = stage.scrollTop;
    const reapply = () => {
      if (!overlay || mode !== 'webtoon') return;
      if (Math.abs(stage.scrollTop - last) > 40) return;
      apply();
      last = stage.scrollTop;
    };
    requestAnimationFrame(reapply);
    setTimeout(reapply, 150);
    setTimeout(reapply, 500);
  }

  function switchMode(next) {
    if (!overlay || next === mode) return;
    // Capture the position BEFORE mode flips — getCurrentPageIndex() reads it
    // differently per mode.
    const idx = getCurrentPageIndex();
    mode = next;
    chrome.storage.local.set({ mtReaderMode: next });
    if (next === 'two-page') {
      // currentSpread is the index of the RIGHT page and spreads step by 2, so
      // align to the same parity the flip logic uses.
      currentSpread = Math.max(0, idx - (idx % 2));
      renderMode();
    } else {
      renderMode();
      restoreScrollToPage(idx);
    }
    // Segment geometry is mode-independent, but the highlight anchor is not.
    refreshSeek(idx);
  }

  function renderWebtoon() {
    stage.style.overflowY = 'auto';
    stage.style.touchAction = 'pan-y';
    stage.style.display = 'block';
    stage.style.perspective = 'none';
    stage.style.scrollBehavior = 'smooth';
    stage.style.webkitOverflowScrolling = 'touch';
    const col = document.createElement('div');
    col.id = 'mt-webtoon-column';
    col.style.cssText = `display: flex; flex-direction: column; align-items: center;`;
    pages.forEach((p, i) => {
      const img = document.createElement('img');
      img.src = p.src;
      img.dataset.idx = i;
      img.style.cssText = `max-width: 100%; height: auto; display: block;`;
      img.onload = () => { markPageLoaded(p); updateProgress(); };
      col.appendChild(img);
      p.el = img;
    });
    stage.appendChild(col);
    stage.onscroll = () => {
      updateProgress();
      // Direction-based auto-hide: scrolling down hides the settings button
      // (and controls bar if open); scrolling up reveals the gear again.
      const st = stage.scrollTop;
      const threshold = 8; // ignore tiny scrolls / touch jitter
      if (st > lastScrollTop + threshold) {
        // Scrolling down.
        gearScrollHidden = true;
        updateGearVisibility();
        if (controlsVisible) hideControls();
      } else if (st < lastScrollTop - threshold) {
        // Scrolling up.
        gearScrollHidden = false;
        updateGearVisibility();
      }
      lastScrollTop = st <= 0 ? 0 : st;
      if (infiniteScrollLoading || infiniteScrollExhausted) return;
      const max = stage.scrollHeight - stage.clientHeight;
      if (max - st < 200) {
        loadNextChapterIntoWebtoon();
      }
    };
  }

  // ── Infinite scroll (webtoon mode) ────────────────────────────────────
  // When the user reaches the bottom of the current chapter's pages, fetch
  // the next chapter URL, load its HTML via fetch(), extract manga images
  // using the same finder logic, and append them to the webtoon column. The
  // reader stays open — no page navigation — so reading continues seamlessly.
  let infiniteScrollLoading = false;
  let infiniteScrollExhausted = false;
  let _lastNextChapterUrl = null;  // tracks the current chapter URL for chained fetching
  let _discoveredNextUrl = null;   // next-chapter link found in the last-fetched HTML
  // Chapter boundaries in the webtoon column: [{ url, startIndex }] in load
  // order. Updated as chapters are appended. Used to determine which chapter
  // the user is currently viewing based on scroll position.
  let _chapterBoundaries = [];

  async function loadNextChapterIntoWebtoon() {
    if (infiniteScrollLoading || infiniteScrollExhausted || !overlay) return;
    infiniteScrollLoading = true;
    // Show a small loading indicator at the bottom of the column.
    const col = document.getElementById('mt-webtoon-column');
    if (!col) { infiniteScrollLoading = false; return; }
    const loader = document.createElement('div');
    loader.style.cssText = `padding: 20px; color: #888; font-size: 13px; text-align: center;`;
    loader.textContent = 'Loading next chapter…';
    col.appendChild(loader);

    try {
      // resolveNextChapterUrl probes the candidate chain (DOM link first,
      // then regex-derived sub-chapter → integer rollovers) and returns the
      // first URL that yields at least one fresh manga image. For a DOM link
      // srcs is null and we re-fetch to extract images.
      const existingSrcs = new Set(pages.map(p => p.src));
      const resolved = await resolveNextChapterUrl(existingSrcs);
      if (!resolved) {
        loader.remove();
        console.log('[Reader] Infinite scroll: no next chapter found.');
        infiniteScrollExhausted = true;
        return;
      }
      const next = resolved.url;
      const freshSrcs = resolved.srcs || [];

      loader.remove();

      if (freshSrcs.length === 0) {
        console.log('[Reader] Infinite scroll: no new images found in next chapter.');
        infiniteScrollExhausted = true;
        return;
      }

      console.log(`[Reader] Infinite scroll: appending ${freshSrcs.length} pages from next chapter.`);

      // Parse chapter numbers for the divider page.
      const prevChapterNum = _extractChapterNumber(_lastNextChapterUrl || window.location.href);
      const nextChapterNum = _extractChapterNumber(next);

      // Insert a chapter divider page before the new chapter's images.
      const divider = document.createElement('div');
      divider.style.cssText = `width: 100%; min-height: 200px; display: flex; flex-direction: column; align-items: center; justify-content: center; gap: 10px; background: #1a1a28; color: #ccc; margin: 10px 0; border-radius: 8px;`;
      divider.innerHTML = `
        <div style="font-size: 14px; color: #888;">Previous: Chapter ${prevChapterNum || '?'}</div>
        <div style="font-size: 18px; font-weight: 700; color: #fff;">Next: Chapter ${nextChapterNum || '?'}</div>
      `;
      col.appendChild(divider);

      // Reset per-chapter progress to start at the new chapter's first page.
      chapterStartIndex = pages.length;

      // Record this chapter's boundary { url, startIndex } so we can tell
      // which chapter the user is currently scrolled to on exit.
      _chapterBoundaries.push({ url: next, startIndex: chapterStartIndex });

      // Append each new page to the column.
      for (const src of freshSrcs) {
        const idx = pages.length;
        // domImg is a proxy: these pages come from the next chapter's HTML and
        // have no <img> on this page, so without a stand-in the finder can
        // never see them and they'd stay untranslated forever.
        const pageObj = { src, el: null, domImg: ensureProxyImg(src) };
        pages.push(pageObj);
        const img = document.createElement('img');
        img.src = src;
        img.dataset.idx = idx;
        img.style.cssText = `max-width: 100%; height: auto; display: block;`;
        img.onload = () => { markPageLoaded(pageObj); updateProgress(); };
        col.appendChild(img);
        pageObj.el = img;
      }

      // A new chapter shifts the current range; rebuild the segments for it.
      refreshSeek();

      // The new pages need observers of their own, and the proxies must be in
      // the DOM before triggerTranslate() runs the finder below.
      watchForTranslationUpdates();

      // Remember the next chapter URL for chained fetches.
      _lastNextChapterUrl = next;

      // If auto-translate is on, trigger translation for the new pages.
      if (autoTranslate) {
        console.log('[Reader] Auto-translate: triggering for new chapter pages.');
        triggerTranslate();
      }
    } catch (e) {
      console.warn('[Reader] Infinite scroll fetch failed:', e);
      loader.textContent = 'Failed to load next chapter.';
      infiniteScrollExhausted = true;
    } finally {
      infiniteScrollLoading = false;
    }
  }

  // Extract the chapter number from a URL for divider display.
  function _extractChapterNumber(url) {
    const m = (url || '').match(/\/(?:chapter|ch)[-_\/]?(\d+(?:\.\d+)?)/i) ||
              (url || '').match(/\/(\d+(?:\.\d+)?)(?:[-_]?(?:end|final))?\/?$/);
    return m ? m[1] : null;
  }

  // ── Two-page mode ─────────────────────────────────────────────────────
  // Each half-page:
  //   - Is upscaled once (cached).
  //   - Has rotateY toward the spine for 3D curvature.
  //   - Has a radial gradient overlay darkening toward the center crease.
  // The spread sizes itself to fit the viewport with NO scrolling:
  //   spread_w = min(95vw, 95vh * 2 / pageAspect)
  async function renderTwoPage() {
    stage.style.overflow = 'hidden';
    stage.style.touchAction = 'none';
    stage.style.display = 'flex';
    stage.style.alignItems = 'center';
    stage.style.justifyContent = 'center';
    stage.style.perspective = '1800px';

    // Clamp current spread to a valid right-page index.
    if (currentSpread < 0) currentSpread = 0;
    if (currentSpread > pages.length - 1) currentSpread = pages.length - 1;
    // Snap to an even right-page index so we don't desync over time.
    // (Right page = currentSpread; left page = currentSpread + 1.)

    const rightIdx = currentSpread;
    const leftIdx = currentSpread + 1;

    // Clear any prior spread + lingering flip overlay (fixes the duplication
    // bug where stale page elements persisted across renders).
    stage.innerHTML = '';
    // Remove any orphan flip overlay from a previous drag.
    const oldFlip = document.getElementById('mt-reader-flip');
    if (oldFlip) oldFlip.remove();

    const spread = document.createElement('div');
    spread.id = 'mt-reader-spread';
    spread.style.cssText = `
      display: flex; position: relative;
      transform-style: preserve-3d;
    `;
    stage.appendChild(spread);

    // We need natural dimensions to size the spread correctly. Load both
    // pages' source images (cached by the browser) to measure them.
    const rightPage = pages[rightIdx];
    const leftPage = leftIdx < pages.length ? pages[leftIdx] : null;

    const rightSrc = rightPage ? await upscaleImage(rightPage.src) : null;
    const leftSrc = leftPage ? await upscaleImage(leftPage.src) : null;

    // Determine spread dimensions from the page aspect ratios so the entire
    // spread fits in the viewport with NO scrolling.
    const measure = (src) => new Promise((resolve) => {
      if (!src) return resolve({ w: 1, h: 1 });
      const im = new Image();
      im.onload = () => resolve({ w: im.naturalWidth || 1, h: im.naturalHeight || 1 });
      im.onerror = () => resolve({ w: 1, h: 1 });
      im.src = src;
    });
    const [rDim, lDim] = await Promise.all([measure(rightSrc), measure(leftSrc)]);
    const rAspect = rDim.w / rDim.h;
    const lAspect = lDim.w / lDim.h;

    // ── Page sizing: each page box fills half the viewport ──────────────
    // The user wants the spread to cover the whole screen, not leave big
    // gutters. Each page BOX is exactly 50vw × 100vh (minus progress bar),
    // and the image inside uses object-fit: contain so it preserves its
    // aspect ratio while filling as much of its half-screen box as possible.
    // Browser-native upscaling (cached) smooths low-res sources at this size.
    const PAGE_W = Math.floor(window.innerWidth / 2);
    const PAGE_H = Math.floor(window.innerHeight - 24); // reserve room for progress bar

    // Build each half-page with 3D curvature + crease gradient overlay.
    const makePage = (src, aspect, side) => {
      const wrap = document.createElement('div');
      wrap.className = 'mt-reader-page';
      // Curvature: rotate the outer edge slightly back so the page appears to
      // curve into the spine at the center. Right page rotates +6deg (its
      // right edge tilts away), left page rotates -6deg.
      const rot = side === 'right' ? 6 : -6;
      wrap.style.cssText = `
        width: ${PAGE_W}px; height: ${PAGE_H}px;
        position: relative; overflow: hidden;
        background: #000;
        transform: rotateY(${rot}deg);
        transform-origin: ${side === 'right' ? 'left center' : 'right center'};
        box-shadow: 0 0 40px rgba(0,0,0,0.7);
      `;
      if (src) {
        const img = document.createElement('img');
        img.src = src;
        // object-fit: fill stretches the image to exactly fill the half-screen
        // box — no clipping, no black bars. The entire page is visible; the
        // aspect ratio is slightly adjusted to fit the box so nothing is cut
        // off. object-position anchors toward the spine so the inner edge
        // (center of the spread) stays put.
        const pos = side === 'right' ? 'left center' : 'right center';
        img.style.cssText = `
          width: 100%; height: 100%;
          object-fit: fill; object-position: ${pos};
          display: block;
          image-rendering: high-quality;
          image-rendering: -webkit-optimize-contrast;
        `;
        wrap.appendChild(img);
      }
      // Crease gradient overlay — darker toward the spine (center).
      const crease = document.createElement('div');
      const creaseDir = side === 'right'
        ? 'linear-gradient(to right, rgba(0,0,0,0.55) 0%, rgba(0,0,0,0.15) 8%, rgba(0,0,0,0) 25%)'
        : 'linear-gradient(to left,  rgba(0,0,0,0.55) 0%, rgba(0,0,0,0.15) 8%, rgba(0,0,0,0) 25%)';
      crease.style.cssText = `
        position: absolute; inset: 0; pointer-events: none;
        background: ${creaseDir};
      `;
      wrap.appendChild(crease);
      return wrap;
    };

    // DOM order: left first (so it sits on the left half), then right.
    // Visually for right-to-left manga the right page is the "current" page.
    if (leftPage) spread.appendChild(makePage(leftSrc, lAspect, 'left'));
    else {
      // Empty filler so the right page stays on the right half.
      const filler = document.createElement('div');
      filler.style.cssText = `width: ${PAGE_W}px; height: ${PAGE_H}px; background: #000;`;
      spread.appendChild(filler);
    }
    spread.appendChild(makePage(rightSrc, rAspect, 'right'));

    // Wire drag-to-flip on each page half.
    wireDragFlip(spread);
  }

  // ── Page-flip via click-and-drag (mouse + touch) ──────────────────────
  // Dragging a page rotates it around its spine; releasing past 50% commits
  // the flip, otherwise it snaps back. Works in both modes but is most
  // useful in two-page.
  function wireDragFlip(spread) {
    if (mode !== 'two-page') return;
    const pagesEls = spread.querySelectorAll('.mt-reader-page');
    pagesEls.forEach((pageEl) => {
      const isRight = pageEl.style.transformOrigin === 'left center';
      const spine = isRight ? 'left' : 'right'; // hinge edge
      const dir = isRight ? -1 : 1;             // rotateY direction

      let dragging = false;
      let startX = 0;
      let startRot = 0;
      let baseRot = isRight ? 6 : -6;            // initial curvature

      const onDown = (clientX) => {
        dragging = true;
        startX = clientX;
        startRot = baseRot;
        pageEl.style.transition = 'none';
      };
      const onMove = (clientX) => {
        if (!dragging) return;
        const dx = clientX - startX;
        const w = pageEl.offsetWidth || 1;
        // Map full-width drag → 90deg of rotation.
        const rot = baseRot + dir * (dx / w) * 90;
        // Clamp to [baseRot, baseRot + dir*90]
        const clamped = dir > 0
          ? Math.max(baseRot, Math.min(baseRot + 90, rot))
          : Math.max(baseRot - 90, Math.min(baseRot, rot));
        pageEl.style.transform = `rotateY(${clamped}deg)`;
      };
      const onUp = (clientX) => {
        if (!dragging) return;
        dragging = false;
        const dx = clientX - startX;
        const w = pageEl.offsetWidth || 1;
        const progress = Math.abs(dx) / w; // 0..1
        pageEl.style.transition = 'transform 280ms ease-out';
        if (progress >= 0.5) {
          // Commit the flip.
          playCrinkle();
          pageEl.style.transform = `rotateY(${baseRot + dir * 90}deg)`;
          setTimeout(() => {
            if (isRight) goNextImmediate();
            else goPrevImmediate();
          }, 200);
        } else {
          // Snap back.
          pageEl.style.transform = `rotateY(${baseRot}deg)`;
        }
      };

      // Mouse
      pageEl.addEventListener('mousedown', (e) => {
        if (e.button !== 0) return;
        e.preventDefault();
        onDown(e.clientX);
        const mm = (ev) => onMove(ev.clientX);
        const mu = (ev) => {
          onUp(ev.clientX);
          document.removeEventListener('mousemove', mm);
          document.removeEventListener('mouseup', mu);
        };
        document.addEventListener('mousemove', mm);
        document.addEventListener('mouseup', mu);
      });
      // Touch — coexist with tap zones. We track the drag but do NOT
      // preventDefault or stopPropagation, so a quick tap (small movement,
      // short duration) still fires the click event that the tap zones
      // listen for. Only when the drag exceeds a threshold do we treat it
      // as a real drag and call preventDefault on subsequent move events
      // to suppress the synthetic click.
      const DRAG_THRESHOLD = 10;  // px — below this it's a tap, not a drag
      let dragMoved = false;
      pageEl.addEventListener('touchstart', (e) => {
        const t = e.changedTouches[0];
        dragMoved = false;
        onDown(t.clientX);
      }, { passive: true });
      pageEl.addEventListener('touchmove', (e) => {
        const t = e.changedTouches[0];
        const dx = Math.abs(t.clientX - startX);
        if (!dragMoved && dx > DRAG_THRESHOLD) {
          dragMoved = true; // crossed threshold — this is a real drag now
        }
        if (dragMoved) {
          e.preventDefault(); // suppress the synthetic click on real drags
        }
        onMove(t.clientX);
      }, { passive: false });
      pageEl.addEventListener('touchend', (e) => {
        const t = e.changedTouches[0];
        onUp(t.clientX);
        if (!dragMoved) {
          // Didn't drag — let the click bubble to the tap zone handler.
          return;
        }
        // Suppress the synthetic click that follows a real drag so the tap
        // zone doesn't also fire a page flip.
        e.preventDefault();
      }, { passive: false });
    });
  }

  // ── Programmatic flip (button / arrow / swipe) ────────────────────────
  function flipPage(direction, fast) {
    const duration = REDUCED_MOTION ? 100 : (fast ? 200 : 400);
    const spread = document.getElementById('mt-reader-spread');
    if (!spread) {
      if (direction === 'next') goNextImmediate(); else goPrevImmediate();
      return;
    }
    playCrinkle();
    // For programmatic flips we animate the right (next) or left (prev) page
    // to 90deg, then advance.
    const pagesEls = spread.querySelectorAll('.mt-reader-page');
    const target = direction === 'next' ? pagesEls[pagesEls.length - 1] : pagesEls[0];
    if (!target) return;
    const isRight = target.style.transformOrigin === 'left center';
    const baseRot = isRight ? 6 : -6;
    const dir = isRight ? -1 : 1;
    target.style.transition = `transform ${duration}ms ease-in`;
    target.style.transform = `rotateY(${baseRot + dir * 90}deg)`;
    setTimeout(() => {
      if (direction === 'next') goNextImmediate(); else goPrevImmediate();
    }, duration);
  }

  function goNext() { if (mode === 'two-page') flipPage('next', true); }
  function goPrev() { if (mode === 'two-page') flipPage('prev', true); }
  function goNextImmediate() {
    currentSpread = Math.min(currentSpread + 2, pages.length - 1);
    if (currentSpread >= pages.length - 1) maybeGoNextChapter();
    renderTwoPage();
    updateProgress();
  }
  function goPrevImmediate() {
    currentSpread = Math.max(currentSpread - 2, 0);
    renderTwoPage();
    updateProgress();
  }

  // ── Next-chapter detection ────────────────────────────────────────────
  // Three-tier strategy, most-reliable first:
  //   1. HTML link extraction (primary): fetch the chapter page and parse
  //      its HTML for an explicit "Next Chapter" link/button/arrow. This is
  //      site-canonical and works regardless of URL numbering scheme.
  //   2. Live DOM link (first fetch only): before we have any fetched HTML,
  //      check the current page's DOM for a next-chapter anchor.
  //   3. URL-math candidate sweep (fallback): increment the chapter number
  //      (1.2 → 1.3, 1.4, …, 1.9, 2.1, 2) and probe each with a delay.
  //
  // Each probe returns both page images AND any next-link found in that
  // page's HTML, so after the first successful fetch the resolver can chain
  // via the site's own next-links — no more URL guessing for chapter 3+.

  // Delay between candidate probes to avoid tripping Cloudflare / rate limits.
  const PROBE_DELAY_MS = 350;

  function _sleep(ms) { return new Promise(r => setTimeout(r, ms)); }

  // Parse HTML for a "next chapter" link. Uses CSS selectors first, then
  // regex over the raw HTML for arrow/text patterns common on manga sites.
  // Returns an absolute URL or null. Tries anchors, buttons, and data-attrs
  // so it works on sites that use JS navigation instead of plain <a href>.
  function extractNextLinkFromHtml(html, baseUrl) {
    if (!html) return null;
    let doc = null;
    try { doc = new DOMParser().parseFromString(html, 'text/html'); }
    catch (e) { doc = null; }

    if (doc) {
      const base = doc.createElement('base');
      base.href = baseUrl;
      doc.head.appendChild(base);

      // Collect href/url from an element in priority order.
      const hrefOf = (el) => {
        if (!el) return null;
        return el.href || el.getAttribute('href') ||
               el.dataset.url || el.dataset.href ||
               el.getAttribute('data-url') || el.getAttribute('data-href') ||
               el.getAttribute('data-next') || null;
      };

      // 1) High-confidence CSS selectors.
      const selectors = [
        'a[rel="next"]', 'a.next', '.nav-next a', '#next_chapter',
        'a.next-chapter', 'a[class*="next-chapter" i]',
        'a[class*="next" i]:not([class*="nextpage" i])',
        '.next a', '.chapter-next a', 'a[aria-label*="next" i]',
        '[data-next]', 'a[data-next]',
      ];
      for (const sel of selectors) {
        const el = doc.querySelector(sel);
        const h = hrefOf(el);
        if (h) return _resolveUrl(h, baseUrl);
      }

      // 2) Scan every link and button; match on class, text, aria-label, or
      //    title. Text may be nested in children (icon fonts, spans), so we
      //    read textContent + title + aria-label together.
      const isNextText = (txt) => {
        if (!txt) return false;
        const t = txt.trim().toLowerCase();
        if (!t) return false;
        if (/^next(\s+chapter|\s+page)?\s*$/.test(t)) return true;
        if (/next\s*chap/i.test(t)) return true;
        if (/next\s*>>?/.test(t)) return true;
        // Standalone arrows (with or without surrounding whitespace).
        if (/^[→⇒›»❯\u00bb]+$/i.test(t)) return true;
        return false;
      };
      for (const el of doc.querySelectorAll('a, button, [role="link"], [role="button"]')) {
        const txt = el.textContent || '';
        const label = el.getAttribute('aria-label') || '';
        const title = el.getAttribute('title') || '';
        if (isNextText(txt) || isNextText(label) || isNextText(title)) {
          const h = hrefOf(el);
          if (h) return _resolveUrl(h, baseUrl);
        }
        // Class-name heuristic as a last resort.
        const cls = (el.getAttribute('class') || '').toLowerCase();
        if (/\bnext[-_]?chapter\b/i.test(cls) || /\bnav[-_]?next\b/i.test(cls)) {
          const h = hrefOf(el);
          if (h) return _resolveUrl(h, baseUrl);
        }
      }
    }

    // 3) Regex fallback over raw HTML. Catches cases where the DOM parse
    //    stripped something or the markup is malformed. Matches <a> tags
    //    whose attributes or inner text look like a next-chapter control.
    const tagRe = /<(?:a|button|[^>]*role=["']link["'][^>]*)\b([^>]*)>([^<]{0,80})/gi;
    let m;
    while ((m = tagRe.exec(html)) !== null) {
      const attrs = m[1] || '';
      const inner = (m[2] || '').trim();
      const hrefMatch = attrs.match(/(?:href|data-(?:url|href|next))=["']([^"']+)["']/i);
      const txt = inner.toLowerCase();
      const isNext =
        /^next(\s+chapter|\s+page)?\s*$/.test(txt) ||
        /next\s*chap/i.test(txt) ||
        /^[→⇒›»❯\u00bb]+$/i.test(txt) ||
        /\bnext[-_]?chapter\b/i.test(attrs) ||
        /\bclass=["'][^"']*\bnav[-_]?next\b[^"']*["']/i.test(attrs);
      if (isNext && hrefMatch) return _resolveUrl(hrefMatch[1], baseUrl);
    }
    return null;
  }

  function _resolveUrl(href, base) {
    try { return new URL(href, base).href; } catch (e) { return href; }
  }

  // Extract manga-page image srcs from a parsed doc. Shared by the live page
  // collector and the probe fetcher so they apply identical heuristics.
  function extractPageSrcsFromDoc(doc, url) {
    const base = doc.createElement('base');
    base.href = url;
    doc.head.appendChild(base);
    const found = [];
    for (const img of doc.querySelectorAll('img')) {
      const raw = img.getAttribute('src') || img.dataset.src || img.getAttribute('data-src') || '';
      if (!raw || raw.startsWith('data:') || raw.startsWith('chrome://')) continue;
      const resolved = img.src || raw;
      const w = parseInt(img.getAttribute('width') || '0', 10);
      const h = parseInt(img.getAttribute('height') || '0', 10);
      if (w > 0 && h > 0 && w * h < 700000) continue;
      if (w > 0 && w < 200) continue;
      found.push(resolved);
    }
    return found;
  }

  // Check the *current* page DOM for an explicit next-chapter anchor.
  // Same logic as extractNextLinkFromHtml but operates on the live document.
  function findNextChapterLinkInDom() {
    const hrefOf = (el) => {
      if (!el) return null;
      return el.href || el.getAttribute('href') ||
             el.dataset.url || el.dataset.href ||
             el.getAttribute('data-url') || el.getAttribute('data-href') ||
             el.getAttribute('data-next') || null;
    };
    const selectors = [
      'a[rel="next"]', 'a.next', '.nav-next a', '#next_chapter',
      'a.next-chapter', 'a[class*="next-chapter" i]',
      'a[class*="next" i]:not([class*="nextpage" i])',
      '.next a', '.chapter-next a', 'a[aria-label*="next" i]',
      '[data-next]', 'a[data-next]',
    ];
    for (const sel of selectors) {
      const el = document.querySelector(sel);
      const h = hrefOf(el);
      if (h) return h;
    }
    const isNextText = (txt) => {
      if (!txt) return false;
      const t = txt.trim().toLowerCase();
      if (!t) return false;
      if (/^next(\s+chapter|\s+page)?\s*$/.test(t)) return true;
      if (/next\s*chap/i.test(t)) return true;
      if (/next\s*>>?/.test(t)) return true;
      if (/^[→⇒›»❯\u00bb]+$/i.test(t)) return true;
      return false;
    };
    for (const el of document.querySelectorAll('a, button, [role="link"], [role="button"]')) {
      const txt = el.textContent || '';
      const label = el.getAttribute('aria-label') || '';
      const title = el.getAttribute('title') || '';
      if (isNextText(txt) || isNextText(label) || isNextText(title)) {
        const h = hrefOf(el);
        if (h) return h;
      }
      const cls = (el.getAttribute('class') || '').toLowerCase();
      if (/\bnext[-_]?chapter\b/i.test(cls) || /\bnav[-_]?next\b/i.test(cls)) {
        const h = hrefOf(el);
        if (h) return h;
      }
    }
    return null;
  }

  // Pure URL→URLs transform. Returns an ordered candidate list for the
  // URL-math fallback. Order is what the user specified:
  //
  //   From a sub-chapter (e.g. 1.1):
  //     1. next decimal      (1.2) — if it exists, go there and keep chaining
  //     2. midpoint .5       (1.5) — catches jumps when +1 doesn't exist
  //     3. bare next integer (2)   — solid chapter
  //     4. next integer .1   (2.1) — in case the solid chapter has subs
  //
  //   From a solid chapter (e.g. 1):
  //     1. next integer      (2)
  //     2. next integer .1   (2.1)
  //
  // As long as the next decimal exists the resolver takes it, so from 1.1 it
  // goes 1.1 → 1.2 → 1.3 → … → 1.9, and only when +1 doesn't exist does it
  // try .5, then roll to the next integer.
  function generateNextChapterCandidates(url) {
    const patterns = [
      /\/chapter[-_]?(\d+(?:\.\d+)?)([^/]*)$/i,
      /\/ch(?:apter)?[-_\/]?(\d+(?:\.\d+)?)([^/]*)$/i,
      /\/(\d+(?:\.\d+)?)(?:[-_]?(?:end|final))?\/?$/,
    ];
    for (const pat of patterns) {
      const m = url.match(pat);
      if (!m) continue;
      const numStr = m[1];
      const nums = [];
      if (numStr.includes('.')) {
        const parts = numStr.split('.');
        const intPart = parts[0];
        const decPart = parts[1];
        const decNum = parseInt(decPart, 10);
        const intNum = parseInt(intPart, 10);
        const intLen = intPart.length;
        const decLen = decPart.length;
        // 1) next decimal (1.1 → 1.2). The resolver probes this first and
        //    chains, so sequential sub-chapters keep going.
        if (decNum + 1 <= 9) {
          nums.push(intPart + '.' + String(decNum + 1).padStart(decLen, '0'));
        }
        // 2) midpoint .5 (skip if we're already at/past .5).
        if (decNum < 5 && decPart !== '5') {
          nums.push(intPart + '.' + '5'.padStart(decLen, '0'));
        }
        // 3) bare next integer.
        const nextIntStr = String(intNum + 1).padStart(intLen, '0');
        nums.push(nextIntStr);
        // 4) next integer's .1.
        nums.push(nextIntStr + '.' + '1'.padStart(decLen, '0'));
      } else {
        // Solid chapter: try next integer, then its .1.
        const num = parseInt(numStr, 10);
        const padded = numStr.length > String(num).length ? numStr.length : 0;
        const next = padded ? String(num + 1).padStart(padded, '0') : String(num + 1);
        nums.push(next);
        nums.push(next + '.1');
      }
      const out = [];
      for (const n of nums) {
        const cand = url.replace(pat, (match) => match.replace(numStr, n));
        if (cand !== url && !out.includes(cand)) out.push(cand);
      }
      return out;
    }
    return [];
  }

  // Fetch a chapter URL and extract: its page images AND any next-chapter
  // link found in its HTML. Returns { srcs, nextUrl } or null on failure.
  async function probeChapterUrl(url, existingSrcs) {
    try {
      const resp = await fetch(url, { credentials: 'include' });
      if (!resp.ok) return null;
      const html = await resp.text();
      const doc = new DOMParser().parseFromString(html, 'text/html');
      const found = extractPageSrcsFromDoc(doc, url);
      const fresh = existingSrcs ? found.filter(s => !existingSrcs.has(s)) : found;
      const nextUrl = extractNextLinkFromHtml(html, url);
      return { srcs: fresh, nextUrl };
    } catch (e) {
      return null;
    }
  }

  // Async resolver: returns { url, srcs } for the first working next chapter,
  // or null. Strategy:
  //   1. (first fetch) live DOM link
  //   2. HTML-discovered link from the previous chapter's fetch (cached)
  //   3. URL-math candidate sweep with PROBE_DELAY_MS between each
  async function resolveNextChapterUrl(existingSrcs) {
    const tried = new Set();

    // Tier 1: live DOM link (only meaningful on the very first fetch, since
    // after that the DOM is stale — we never navigate during infinite scroll).
    if (!_lastNextChapterUrl) {
      const link = findNextChapterLinkInDom();
      if (link && !tried.has(link)) {
        tried.add(link);
        const probe = await probeChapterUrl(link, existingSrcs);
        if (probe && probe.srcs.length > 0) {
          _discoveredNextUrl = probe.nextUrl;
          return { url: link, srcs: probe.srcs };
        }
      }
    }

    // Tier 2: use the next-chapter link discovered in the PREVIOUS chapter's
    // HTML. This is the canonical site-provided URL — far more reliable than
    // URL math, and it's what makes chapter 3, 4, 5… chain correctly.
    if (_discoveredNextUrl && !tried.has(_discoveredNextUrl)) {
      const cand = _discoveredNextUrl;
      tried.add(cand);
      const probe = await probeChapterUrl(cand, existingSrcs);
      if (probe && probe.srcs.length > 0) {
        _discoveredNextUrl = probe.nextUrl;
        return { url: cand, srcs: probe.srcs };
      }
      // Discovered link didn't work — clear it and fall through to the sweep.
      _discoveredNextUrl = null;
    }

    // Tier 3: URL-math candidate sweep. Full 1.3..1.9, 2.1, 2 chain with a
    // delay between probes so Cloudflare / rate-limiters don't trip.
    const base = _lastNextChapterUrl || window.location.href;
    const candidates = generateNextChapterCandidates(base);
    for (let i = 0; i < candidates.length; i++) {
      const cand = candidates[i];
      if (tried.has(cand)) continue;
      tried.add(cand);
      const probe = await probeChapterUrl(cand, existingSrcs);
      if (probe && probe.srcs.length > 0) {
        _discoveredNextUrl = probe.nextUrl;
        return { url: cand, srcs: probe.srcs };
      }
      // Delay before the next probe (skip after the last candidate).
      if (i < candidates.length - 1) await _sleep(PROBE_DELAY_MS);
    }
    return null;
  }

  function maybeGoNextChapter() {
    // Two-page mode navigates the whole page, so we probe candidates before
    // navigating to avoid landing on a 404.
    toast('Loading next chapter…');
    (async () => {
      const resolved = await resolveNextChapterUrl(null);
      if (!resolved) {
        showEndOfBook();
        infiniteScrollExhausted = true;
        return;
      }
      setTimeout(() => { window.location.href = resolved.url; }, 300);
    })();
  }

  // ── End of Book pseudo page ────────────────────────────────────────────
  // When there's no next chapter, insert a centered card into the webtoon
  // column (or show it as an overlay in two-page mode) that says "End of
  // Book" with a checkmark icon.
  function showEndOfBook() {
    if (mode === 'webtoon') {
      const col = document.getElementById('mt-webtoon-column');
      if (!col) return;
      // Don't add a duplicate.
      if (document.getElementById('mt-end-of-book')) return;
      const card = document.createElement('div');
      card.id = 'mt-end-of-book';
      card.style.cssText = `width: 100%; min-height: 300px; display: flex; flex-direction: column; align-items: center; justify-content: center; gap: 16px; background: #1a1a28; color: #ccc; margin: 20px 0; border-radius: 12px;`;
      card.innerHTML = `
        <svg viewBox="0 0 24 24" width="64" height="64" fill="none" stroke="#22a552" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/><polyline points="22 4 12 14.01 9 11.01"/></svg>
        <div style="font-size: 24px; font-weight: 700; color: #fff;">End of Book</div>
        <div style="font-size: 13px; color: #888;">No more chapters detected.</div>
      `;
      col.appendChild(card);
    } else {
      // Two-page: show an overlay card.
      if (document.getElementById('mt-end-of-book')) return;
      const card = document.createElement('div');
      card.id = 'mt-end-of-book';
      card.style.cssText = `position: absolute; inset: 0; display: flex; flex-direction: column; align-items: center; justify-content: center; gap: 16px; background: rgba(0,0,0,0.9); z-index: 10;`;
      card.innerHTML = `
        <svg viewBox="0 0 24 24" width="64" height="64" fill="none" stroke="#22a552" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/><polyline points="22 4 12 14.01 9 11.01"/></svg>
        <div style="font-size: 24px; font-weight: 700; color: #fff;">End of Book</div>
        <div style="font-size: 13px; color: #888;">No more chapters detected.</div>
      `;
      stage.appendChild(card);
    }
  }

  // ── Chapter seek bar ──────────────────────────────────────────────────
  // A YouTube-style scrubber: one segment per page of the CURRENT chapter
  // only (a 12-chapter infinite-scroll session would otherwise produce
  // hairline segments nobody can hit). Segment colour is the page's state:
  //   green = translated, grey = loaded, black = not loaded yet.
  // Hold-to-open on touch, hover-the-bottom-strip on PC. Dragging previews a
  // page; releasing tweens the scroll there.
  let seekWrap = null;
  let seekTrack = null;
  let seekHit = null;
  let seekBubble = null;
  let seekBubbleImg = null;
  let seekBubbleLabel = null;
  let seekSegEls = [];        // segment elements, index-aligned to seekRange
  let seekRange = null;       // { start, end } page indices currently painted
  let seekVisible = false;
  let seekHideTimer = null;
  let seekHoldTimer = null;
  let scrubbing = false;
  let scrubIdx = 0;
  let holdStartX = 0, holdStartY = 0;
  // Guards the existing swipe handler: a hold-scrub ends with a touchend that
  // would otherwise read as a swipe and flip a page.
  let lastSeekEndTs = 0;
  const SEEK_HOLD_MS = 400;
  const SEEK_HOLD_SLOP = 12;
  // Collapsed the wrap is just the progress bar; expanded it grows in place.
  const SEEK_IDLE_H = 18;
  const SEEK_OPEN_H = 46;

  const SEG_TRANSLATED = '#22a552';
  const SEG_LOADED = '#7d7d8c';
  const SEG_UNLOADED = '#000';

  // The wrap + rail already exist (buildReaderDom builds them as the progress
  // bar). This only adds the parts that appear while seeking: the preview
  // bubble and the segment track that overlays the rail.
  function buildSeekBar() {
    if (!seekWrap || !progressBar) return;

    seekBubble = document.createElement('div');
    seekBubble.style.cssText = `
      position: absolute; bottom: 34px; left: 0; transform: translateX(-50%);
      display: none; flex-direction: column; align-items: center; gap: 4px;
      padding: 6px; background: rgba(20,20,31,0.96); border: 1px solid #3a3a4c;
      border-radius: 8px; pointer-events: none; z-index: 2;
    `;
    seekBubbleImg = document.createElement('img');
    seekBubbleImg.style.cssText = `width: 64px; height: 90px; object-fit: cover; object-position: top; border-radius: 4px; background: #111; display: block;`;
    seekBubbleLabel = document.createElement('div');
    seekBubbleLabel.style.cssText = `font-size: 11px; color: #ddd; font-weight: 600; white-space: nowrap;`;
    seekBubble.appendChild(seekBubbleImg);
    seekBubble.appendChild(seekBubbleLabel);
    seekWrap.appendChild(seekBubble);

    // Sits exactly on top of the rail and shares its 4px height while idle, so
    // the collapsed segments read as one continuous bar. Growing the segments
    // is what visually "sections off" the progress bar.
    seekTrack = document.createElement('div');
    seekTrack.id = 'mt-r-seek-track';
    seekTrack.style.cssText = `
      position: absolute; left: 0; right: 0; bottom: 0;
      display: flex; align-items: flex-end; gap: 0px;
      height: 4px; cursor: pointer; touch-action: none;
      opacity: 0; transition: opacity 160ms ease, gap 200ms ease, height 220ms cubic-bezier(0.22, 0.61, 0.36, 1);
    `;
    progressBar.appendChild(seekTrack);

    // Hit target: the 4px rail is far too thin to tap, so an invisible strip
    // covers the whole bottom band and forwards its position to the track.
    seekHit = document.createElement('div');
    seekHit.id = 'mt-r-seek-hit';
    seekHit.style.cssText = `
      position: absolute; left: 0; right: 0; bottom: 0; height: 100%;
      cursor: pointer; touch-action: none; z-index: 1;
    `;
    seekWrap.appendChild(seekHit);
  }

  // Page indices that begin a chapter. The first chapter has no boundary
  // record (it was never appended), so 0 is always a start.
  function getChapterStarts() {
    const starts = new Set([0]);
    _chapterBoundaries.forEach(b => starts.add(b.startIndex));
    return Array.from(starts).sort((a, b) => a - b);
  }

  function getChapterRangeFor(idx) {
    const starts = getChapterStarts();
    let start = 0;
    for (const s of starts) {
      if (s <= idx) start = s;
      else break;
    }
    let end = pages.length - 1;
    for (const s of starts) {
      if (s > start) { end = s - 1; break; }
    }
    return { start, end: Math.max(start, end) };
  }

  function pageState(p) {
    if (!p) return 'unloaded';
    if (p.domImg && p.domImg.hasAttribute('data-mt-translated')) return 'translated';
    if (p.loaded) return 'loaded';
    // Rendered copy or host image already decoded counts as loaded even if we
    // never saw the load event (cached images fire it before we attach).
    if (p.el && p.el.complete && p.el.naturalWidth > 0) { p.loaded = true; return 'loaded'; }
    if (p.domImg && p.domImg.complete && p.domImg.naturalWidth > 0) { p.loaded = true; return 'loaded'; }
    return 'unloaded';
  }

  function buildSeekSegments(range) {
    seekTrack.innerHTML = '';
    seekSegEls = [];
    for (let i = range.start; i <= range.end; i++) {
      const seg = document.createElement('div');
      // Full-height so the segments ARE the rail; paintSeek handles the
      // collapsed-vs-expanded look.
      seg.style.cssText = `
        flex: 1 1 0; min-width: 0; height: 100%; align-self: stretch;
        transition: background 160ms ease, border-radius 200ms ease, transform 160ms ease;
      `;
      seekTrack.appendChild(seg);
      seekSegEls.push(seg);
    }
    seekRange = range;
  }

  // Repaint colours + the highlighted segment. Cheap enough to call on every
  // image load and after every translation resync.
  function paintSeek(highlightIdx) {
    if (!seekRange || !seekSegEls.length) return;
    const hi = typeof highlightIdx === 'number' ? highlightIdx : null;
    const open = seekVisible;
    for (let i = 0; i < seekSegEls.length; i++) {
      const idx = seekRange.start + i;
      const seg = seekSegEls[i];
      const state = pageState(pages[idx]);
      seg.style.background = state === 'translated' ? SEG_TRANSLATED
        : state === 'loaded' ? SEG_LOADED : SEG_UNLOADED;
      // Unloaded segments are black on a black backdrop — outline them so the
      // remaining length of the chapter is still readable. Only once expanded;
      // collapsed the outline would fringe the flat rail.
      seg.style.boxShadow = (open && state === 'unloaded') ? 'inset 0 0 0 1px #333' : 'none';
      // Rounded + gapped only when expanded, so the collapsed state is one
      // seamless bar that matches the plain progress rail.
      seg.style.borderRadius = open ? '2px' : '0';
      const active = open && hi === idx;
      seg.style.transform = active ? 'scaleY(1.35)' : 'none';
      seg.style.outline = active ? '1px solid #fff' : 'none';
    }
  }

  // Images decode in bursts, so coalesce the repaints into one frame instead
  // of restyling every segment per load event.
  let seekPaintQueued = false;
  function markPageLoaded(p) {
    if (!p || p.loaded) return;
    p.loaded = true;
    if (!seekWrap || seekPaintQueued) return;
    seekPaintQueued = true;
    requestAnimationFrame(() => {
      seekPaintQueued = false;
      if (seekRange) paintSeek(scrubbing ? scrubIdx : null);
    });
  }

  // Rebuild if the chapter changed or pages were appended, then repaint.
  function refreshSeek(highlightIdx) {
    if (!seekTrack || !pages.length) return;
    const anchor = typeof highlightIdx === 'number' ? highlightIdx : getCurrentPageIndex();
    const range = getChapterRangeFor(anchor);
    if (!seekRange || seekRange.start !== range.start || seekRange.end !== range.end) {
      buildSeekSegments(range);
    }
    paintSeek(anchor);
  }

  // Open/close grow and shrink the progress bar in place — the element is the
  // progress bar, so fading it out would take the progress bar with it.
  function showSeek() {
    if (!seekWrap || !pages.length) return;
    // The expanded bar would sit under the controls row; don't stack them.
    if (controlsVisible) return;
    clearTimeout(seekHideTimer);
    // paintSeek reads seekVisible to decide the collapsed vs expanded look, so
    // flip it before the first paint.
    seekVisible = true;
    refreshSeek();
    seekWrap.style.height = SEEK_OPEN_H + 'px';
    seekWrap.style.paddingBottom = '14px';
    seekWrap.style.background = 'rgba(0,0,0,0.82)';
    seekTrack.style.height = '24px';
    seekTrack.style.gap = '2px';
    seekTrack.style.opacity = '1';
    // The green fill and the segments paint the same rail; hand it over.
    const fill = document.getElementById('mt-reader-progress-fill');
    if (fill) fill.style.opacity = '0';
    updateGearVisibility();
  }

  function hideSeek() {
    if (!seekWrap) return;
    clearTimeout(seekHideTimer);
    seekVisible = false;
    scrubbing = false;
    seekWrap.style.height = SEEK_IDLE_H + 'px';
    seekWrap.style.paddingBottom = '7px';
    seekWrap.style.background = 'rgba(0,0,0,0.6)';
    seekTrack.style.height = '4px';
    seekTrack.style.gap = '0px';
    seekTrack.style.opacity = '0';
    const fill = document.getElementById('mt-reader-progress-fill');
    if (fill) fill.style.opacity = '1';
    seekBubble.style.display = 'none';
    updateGearVisibility();
    paintSeek(null);
  }

  function scheduleSeekHide(delay) {
    clearTimeout(seekHideTimer);
    seekHideTimer = setTimeout(hideSeek, typeof delay === 'number' ? delay : 900);
  }

  function seekIndexFromClientX(clientX) {
    const rect = seekTrack.getBoundingClientRect();
    if (rect.width <= 0 || !seekRange) return seekRange ? seekRange.start : 0;
    const frac = Math.max(0, Math.min(1, (clientX - rect.left) / rect.width));
    const count = seekRange.end - seekRange.start + 1;
    return seekRange.start + Math.min(count - 1, Math.floor(frac * count));
  }

  function beginScrub(clientX) {
    clearTimeout(seekHoldTimer);
    seekHoldTimer = null;
    if (!pages.length) return;
    if (controlsVisible) hideControls();
    showSeek();
    scrubbing = true;
    try { if (navigator.vibrate) navigator.vibrate(12); } catch (e) {}
    updateScrub(clientX);
  }

  function updateScrub(clientX) {
    if (!seekRange) return;
    scrubIdx = seekIndexFromClientX(clientX);
    const p = pages[scrubIdx];
    paintSeek(scrubIdx);
    // Bubble follows the finger but stays inside the track.
    const rect = seekTrack.getBoundingClientRect();
    const wrapRect = seekWrap.getBoundingClientRect();
    const x = Math.max(rect.left + 40, Math.min(rect.right - 40, clientX)) - wrapRect.left;
    seekBubble.style.display = 'flex';
    seekBubble.style.left = x + 'px';
    if (p && seekBubbleImg.src !== p.src) seekBubbleImg.src = p.src;
    const inChapter = scrubIdx - seekRange.start + 1;
    const total = seekRange.end - seekRange.start + 1;
    seekBubbleLabel.textContent = `Page ${inChapter} / ${total}`;
  }

  function endScrub() {
    if (!scrubbing) return;
    scrubbing = false;
    lastSeekEndTs = Date.now();
    commitSeek(scrubIdx);
    scheduleSeekHide(400);
  }

  function cancelHold() {
    clearTimeout(seekHoldTimer);
    seekHoldTimer = null;
  }

  function commitSeek(idx) {
    const p = pages[idx];
    if (!p) return;
    if (mode === 'two-page') {
      currentSpread = Math.max(0, idx - (idx % 2));
      renderTwoPage();
      updateProgress();
      return;
    }
    if (p.el && p.el.isConnected) tweenScrollTo(p.el.offsetTop);
  }

  // Webtoon scroll is set to `smooth`, which crawls across a long jump. Drive
  // the scroll ourselves so a release lands fast in either direction.
  let scrollTweenId = null;
  function tweenScrollTo(top, duration) {
    if (!stage) return;
    if (scrollTweenId) cancelAnimationFrame(scrollTweenId);
    const dur = typeof duration === 'number' ? duration : 450;
    const start = stage.scrollTop;
    const delta = top - start;
    const prevBehavior = stage.style.scrollBehavior;
    stage.style.scrollBehavior = 'auto';
    if (Math.abs(delta) < 2 || REDUCED_MOTION) {
      stage.scrollTop = top;
      stage.style.scrollBehavior = prevBehavior;
      return;
    }
    const t0 = performance.now();
    const step = (now) => {
      const t = Math.min(1, (now - t0) / dur);
      const eased = 1 - Math.pow(1 - t, 3);
      stage.scrollTop = start + delta * eased;
      if (t < 1) {
        scrollTweenId = requestAnimationFrame(step);
      } else {
        scrollTweenId = null;
        stage.style.scrollBehavior = prevBehavior;
      }
    };
    scrollTweenId = requestAnimationFrame(step);
  }

  function wireSeekInput() {
    // PC: hovering the bar reveals it; leaving it collapses it back into the
    // plain progress rail. The band is measured against the expanded height so
    // the pointer stays "inside" once it has grown.
    overlay.addEventListener('mousemove', (e) => {
      if (scrubbing) return;
      const band = window.innerHeight - (SEEK_OPEN_H + 8);
      if (e.clientY >= band) showSeek();
      else if (seekVisible) scheduleSeekHide(100);
    });

    const onScrubMouseMove = (e) => { if (scrubbing) updateScrub(e.clientX); };
    const onScrubMouseUp = (e) => {
      document.removeEventListener('mousemove', onScrubMouseMove, true);
      document.removeEventListener('mouseup', onScrubMouseUp, true);
      if (scrubbing) { updateScrub(e.clientX); endScrub(); }
    };
    // The rail itself is only 4px tall, so the invisible hit strip covering the
    // whole bar is what actually receives the press.
    seekHit.addEventListener('mousedown', (e) => {
      e.preventDefault();
      e.stopPropagation();
      showSeek();
      scrubbing = true;
      updateScrub(e.clientX);
      document.addEventListener('mousemove', onScrubMouseMove, true);
      document.addEventListener('mouseup', onScrubMouseUp, true);
    });

    // Touch: press and hold anywhere on the page area, or touch the bar
    // directly once it's up.
    overlay.addEventListener('touchstart', (e) => {
      if (!overlay) return;
      const t = e.changedTouches[0];
      holdStartX = t.clientX; holdStartY = t.clientY;
      const onSeek = e.target.closest && e.target.closest('#mt-r-seek');
      if (onSeek) { beginScrub(t.clientX); return; }
      if (e.target.closest && e.target.closest('#mt-reader-controls, #mt-tap-guide, button')) return;
      cancelHold();
      seekHoldTimer = setTimeout(() => beginScrub(holdStartX), SEEK_HOLD_MS);
    }, { passive: true });

    overlay.addEventListener('touchmove', (e) => {
      const t = e.changedTouches[0];
      if (scrubbing) {
        // Suppress webtoon scrolling / page drags while scrubbing.
        e.preventDefault();
        updateScrub(t.clientX);
        return;
      }
      if (seekHoldTimer &&
          (Math.abs(t.clientX - holdStartX) > SEEK_HOLD_SLOP ||
           Math.abs(t.clientY - holdStartY) > SEEK_HOLD_SLOP)) {
        cancelHold();
      }
    }, { passive: false });

    overlay.addEventListener('touchend', (e) => {
      cancelHold();
      if (!scrubbing) return;
      e.preventDefault();
      endScrub();
    }, { passive: false });

    overlay.addEventListener('touchcancel', () => {
      cancelHold();
      if (scrubbing) { scrubbing = false; scheduleSeekHide(200); }
    }, { passive: true });
  }

  // ── Progress bar ──────────────────────────────────────────────────────
  function updateProgress() {
    const fill = document.getElementById('mt-reader-progress-fill');
    if (!fill) return;
    let pct = 0;
    if (mode === 'webtoon') {
      const max = stage.scrollHeight - stage.clientHeight;
      const scrolled = stage.scrollTop;
      // Per-chapter progress: compute based on the current chapter's page
      // range (chapterStartIndex → end). When a new chapter loads via
      // infinite scroll, chapterStartIndex resets so progress goes to 0%.
      if (pages.length > chapterStartIndex) {
        // Estimate: find the scroll position corresponding to chapterStartIndex.
        const col = document.getElementById('mt-webtoon-column');
        if (col && col.children[chapterStartIndex]) {
          const chapterTop = col.children[chapterStartIndex].offsetTop;
          const chapterMax = max - chapterTop;
          pct = chapterMax > 0 ? Math.max(0, ((scrolled - chapterTop) / chapterMax) * 100) : 0;
        } else {
          pct = max > 0 ? (scrolled / max) * 100 : 0;
        }
      } else {
        pct = max > 0 ? (scrolled / max) * 100 : 0;
      }
    } else {
      const chapterPages = pages.length - chapterStartIndex;
      const currentInChapter = currentSpread - chapterStartIndex;
      pct = chapterPages > 0 ? Math.max(0, ((currentInChapter + 1) / chapterPages) * 100) : 0;
    }
    pct = Math.max(0, Math.min(100, pct));
    fill.style.width = pct + '%';
    if (progressLabel) progressLabel.textContent = Math.round(pct) + '%';
  }

  // ── Gear visibility ───────────────────────────────────────────────────
  // Single source of truth for the center gear button. In two-page mode the
  // center tap zone already opens the controls bar, so the gear is redundant
  // and sits on top of the spread — it stays hidden there on PC and mobile
  // alike. Elsewhere it hides while the controls bar or the expanded seek bar
  // is showing, since those occupy the same strip.
  function updateGearVisibility() {
    if (!centerSettingsBtn) return;
    const hide = mode === 'two-page' || controlsVisible || seekVisible || gearScrollHidden;
    centerSettingsBtn.style.display = hide ? 'none' : 'flex';
  }

  // ── Controls visibility ───────────────────────────────────────────────
  function toggleControls() {
    controlsVisible = !controlsVisible;
    controlsBar.style.display = controlsVisible ? 'flex' : 'none';
    // The controls row sits directly on top of the bar; collapse it back to the
    // plain rail so the two don't overlap.
    if (controlsVisible && seekVisible) hideSeek();
    updateGearVisibility();
  }
  function showControls() {
    controlsVisible = true;
    controlsBar.style.display = 'flex';
    if (seekVisible) hideSeek();
    updateGearVisibility();
  }
  function hideControls() {
    controlsVisible = false;
    controlsBar.style.display = 'none';
    updateGearVisibility();
  }

  // ── Touch gestures (settings + swipe nav) ────────────────────────────
  function wireTouchGestures() {
    let startX = 0, startY = 0, startTime = 0;
    // Listen on overlay (not stage) so gestures work even when tap zones
    // or the controls bar are in the way. Click events on buttons still
    // fire normally because we don't preventDefault on touchend.
    overlay.addEventListener('touchstart', (e) => {
      const t = e.changedTouches[0];
      startX = t.clientX; startY = t.clientY; startTime = Date.now();
    }, { passive: true });
    overlay.addEventListener('touchend', (e) => {
      // A scrub release fires touchend on the overlay — ignore it here so it
      // doesn't also register as a swipe or reader-exit gesture.
      if (Date.now() - lastSeekEndTs < 120) return;
      const t = e.changedTouches[0];
      const dx = t.clientX - startX;
      const dy = t.clientY - startY;
      const absX = Math.abs(dx), absY = Math.abs(dy);
      // Swipe down from top 15% → exit reader.
      if (dy > 80 && absY > absX && startY < window.innerHeight * 0.15) {
        exitReader(); return;
      }
      // Horizontal swipe (left/right) → page navigation in two-page mode.
      // Capped at 1 page per swipe.
      if (absX > 50 && absX > absY && mode === 'two-page') {
        if (dx < 0) flipPage('next', false);
        else flipPage('prev', false);
      }
    }, { passive: true });
  }

  function onKeyDown(e) {
    if (!overlay) return;
    if (e.key === 'ArrowRight') goNext();
    else if (e.key === 'ArrowLeft') goPrev();
    else if (e.key === 'Escape') {
      if (controlsVisible) hideControls();
      else exitReader();
    }
  }

  // ── Mouse wheel page-flip with cooldown (PC only) ────────────────────
  // A single wheel notch = exactly one flip, then a cooldown blocks the
  // next notch for COOLDOWN_MS. No burst bypass — fast scrolling still only
  // does 1 page per cooldown. Only on PC (IS_TOUCH = false) so mobile
  // touch-scrolling in webtoon mode is unaffected.
  let wheelAccum = 0;
  let wheelCooldownUntil = 0;
  const WHEEL_THRESHOLD = 60;
  const COOLDOWN_MS = 320;

  function onWheel(e) {
    if (!overlay || mode !== 'two-page' || IS_TOUCH) return;
    const dy = e.deltaY;
    if (Math.abs(dy) < 1) return;

    e.preventDefault();
    const now = performance.now();
    wheelAccum += dy;

    if (Math.abs(wheelAccum) < WHEEL_THRESHOLD) return;

    const direction = wheelAccum > 0 ? 'next' : 'prev';

    // Hard cooldown — always enforced, no burst bypass.
    if (now < wheelCooldownUntil) {
      wheelAccum = 0;
      return;
    }

    wheelAccum = 0;
    wheelCooldownUntil = now + COOLDOWN_MS;
    if (direction === 'next') goNext();
    else goPrev();
  }

  function triggerTranslate() {
    if (!window.__mtStartTranslation) { alert('Translation not available.'); return; }
    chrome.storage.local.get(['ocrLang', 'targetLang', 'combineAmount', 'skipSfx', 'contextAware'],
      (stored) => {
        window.__mtStartTranslation(
          stored.ocrLang || 'ja',
          stored.targetLang || 'en',
          {
            combineAmount: parseInt(stored.combineAmount || '1', 10) || 1,
            skipSfx: stored.skipSfx === true,
            contextAware: stored.contextAware === true,
          }
        );
      }
    );
  }

  // ── Chapter URL tracking on exit / reload ────────────────────────────
  // When the user reads ahead via infinite scroll (chapter 1 → 2 → 3) and
  // then exits or reloads, navigate the underlying page to the chapter they
  // were actually reading instead of leaving them back on chapter 1.

  // Determine which chapter the user is currently viewing based on their
  // scroll position within the webtoon column. Returns the chapter URL.
  function getCurrentChapterUrl() {
    if (_chapterBoundaries.length === 0) {
      return _lastNextChapterUrl || window.location.href;
    }
    const col = document.getElementById('mt-webtoon-column');
    if (!col) {
      return _lastNextChapterUrl || window.location.href;
    }
    // Find the last chapter boundary whose top is at or above the current
    // scroll position. This is the chapter the user is reading right now.
    const scrollTop = stage ? stage.scrollTop : 0;
    let current = _chapterBoundaries[0];
    for (const b of _chapterBoundaries) {
      const child = col.children[b.startIndex];
      if (child && child.offsetTop <= scrollTop + 50) {
        current = b;
      } else {
        break;
      }
    }
    return current.url;
  }

  // Persist the current chapter URL to sessionStorage.
  function saveReadingPosition() {
    if (!overlay) return;
    const chapterUrl = mode === 'webtoon' ? getCurrentChapterUrl() : (_lastNextChapterUrl || window.location.href);
    if (!chapterUrl) return;
    sessionStorage.setItem('mtReaderChapterUrl', chapterUrl);
  }

  // Save the chapter URL before the page unloads (reload / tab close / nav).
  window.addEventListener('beforeunload', () => {
    if (overlay) saveReadingPosition();
  });

  function exitReader() {
    if (!overlay) return;
    // Save the current mode + sound preference so the next session restores it.
    chrome.storage.local.set({ mtReaderMode: mode, mtReaderSound: soundOn });
    // Save the chapter URL so reload / re-open lands on the right chapter.
    saveReadingPosition();
    disconnectObservers();
    removeProxyHost();
    clearTimeout(seekHideTimer);
    clearTimeout(seekHoldTimer);
    if (scrollTweenId) { cancelAnimationFrame(scrollTweenId); scrollTweenId = null; }
    scrubbing = false; seekVisible = false;
    seekWrap = null; seekTrack = null; seekHit = null; seekBubble = null;
    seekBubbleImg = null; seekBubbleLabel = null;
    seekSegEls = []; seekRange = null;
    progressBar = null; progressLabel = null;
    document.removeEventListener('keydown', onKeyDown);
    exitFullscreenIfActive();
    overlay.remove();
    overlay = null; stage = null;
    document.body.style.overflow = '';
    sessionStorage.removeItem('mtReaderActive');
    // Record the exit time so content.js can detect a recent reader return.
    sessionStorage.setItem('mtReaderExitTime', String(Date.now()));
    // If the user has scrolled to a different chapter via infinite scroll,
    // navigate the underlying page to that chapter so the URL matches.
    const targetUrl = sessionStorage.getItem('mtReaderChapterUrl');
    if (targetUrl && targetUrl !== window.location.href) {
      window.location.href = targetUrl;
    }
  }

  function toast(msg) {
    const t = document.createElement('div');
    t.textContent = msg;
    t.style.cssText = `
      position: fixed; bottom: 60px; left: 50%; transform: translateX(-50%);
      background: rgba(0,0,0,0.85); color: #fff; padding: 10px 16px;
      border-radius: 8px; z-index: 2147483648; font-size: 13px;
      transition: opacity 0.4s;
    `;
    overlay.appendChild(t);
    setTimeout(() => { t.style.opacity = '0'; }, 1800);
    setTimeout(() => t.remove(), 2300);
  }

  // ── Reload redirect + auto-enter ──────────────────────────────────────
  // If the reader was open and the user reloaded → beforeunload saved the
  // chapter URL. If it differs from the current page, redirect there so the
  // user lands on the right chapter. mtReaderActive gates this so manual
  // navigation (clicking prev-chapter on the site) isn't hijacked.
  (function handleReloadRedirect() {
    if (sessionStorage.getItem('mtReaderActive') !== '1') {
      // Reader wasn't active — clear any stale chapter URL and bail.
      sessionStorage.removeItem('mtReaderChapterUrl');
      return;
    }
    const savedChapter = sessionStorage.getItem('mtReaderChapterUrl');
    if (savedChapter && savedChapter !== window.location.href) {
      // Consume immediately so it can't loop.
      sessionStorage.removeItem('mtReaderChapterUrl');
      window.location.replace(savedChapter);
      return;
    }
    // Same chapter — consume and fall through to auto-enter.
    sessionStorage.removeItem('mtReaderChapterUrl');
  })();

  if (sessionStorage.getItem('mtReaderActive') === '1') {
    // Keep polling well past the old 8s ceiling: slow / protected sites
    // (MangaDex) can take minutes to hand over their chapter images, and a
    // single finder call right after the script loads is always too early.
    // waitForPageImages() also settles, so we don't auto-open on page 1 alone
    // while the rest of the chapter is still streaming in.
    (async () => {
      const imgs = await waitForPageImages();
      if (imgs && imgs.length > 0) window.__mtOpenReader();
      else sessionStorage.removeItem('mtReaderActive');
    })();
  }
})();
