// ============================================================================
// Shared language list for the Manga Translator extension.
//
// This mirrors the backend LANGUAGES table in app.py. It is used as an offline
// fallback for the popup / options dropdowns; when a server URL is configured
// the lists are refreshed from GET /languages so the two stay in sync.
//
// `featured: true` languages are shown first; the rest live under a
// "More languages" group revealed on demand.
// ============================================================================
const MT_LANGUAGES = [
  // ── Featured ──
  { code: 'en', name: 'English',    featured: true },
  { code: 'ja', name: 'Japanese',   featured: true },
  { code: 'ko', name: 'Korean',     featured: true },
  { code: 'zh', name: 'Chinese',    featured: true },
  { code: 'id', name: 'Indonesian', featured: true },
  { code: 'ru', name: 'Russian',    featured: true },
  { code: 'es', name: 'Spanish',    featured: true },
  // ── More ──
  { code: 'zh-tw', name: 'Chinese (Traditional)', featured: false },
  { code: 'fr',  name: 'French',     featured: false },
  { code: 'de',  name: 'German',     featured: false },
  { code: 'it',  name: 'Italian',    featured: false },
  { code: 'pt',  name: 'Portuguese', featured: false },
  { code: 'pt-br', name: 'Portuguese (Brazil)', featured: false },
  { code: 'nl',  name: 'Dutch',      featured: false },
  { code: 'pl',  name: 'Polish',     featured: false },
  { code: 'tr',  name: 'Turkish',    featured: false },
  { code: 'vi',  name: 'Vietnamese', featured: false },
  { code: 'th',  name: 'Thai',       featured: false },
  { code: 'ar',  name: 'Arabic',     featured: false },
  { code: 'he',  name: 'Hebrew',     featured: false },
  { code: 'hi',  name: 'Hindi',      featured: false },
  { code: 'el',  name: 'Greek',      featured: false },
  { code: 'uk',  name: 'Ukrainian',  featured: false },
  { code: 'cs',  name: 'Czech',      featured: false },
  { code: 'sv',  name: 'Swedish',    featured: false },
  { code: 'fi',  name: 'Finnish',    featured: false },
  { code: 'no',  name: 'Norwegian',  featured: false },
  { code: 'da',  name: 'Danish',     featured: false },
  { code: 'hu',  name: 'Hungarian',  featured: false },
  { code: 'ro',  name: 'Romanian',   featured: false },
  { code: 'fil', name: 'Filipino',   featured: false },
  { code: 'ms',  name: 'Malay',      featured: false },
  { code: 'fa',  name: 'Persian',    featured: false },
];

// Sentinel value for the "More languages…" entry. Selecting it doesn't pick a
// language — it expands the dropdown to reveal the full list.
const MT_MORE_SENTINEL = '__mt_more__';

// Render the options for a language <select>.
//  - collapsed (default): featured languages + a "⋯ More languages…" entry.
//  - expanded: every language, grouped Common / More languages.
// The chosen `selected` code always appears (if it's a non-featured language
// while collapsed, it's added so the current value stays visible/selectable).
function _mtRenderLangOptions(selectEl, list, selected, expanded) {
  const featured = list.filter(l => l.featured);
  const more = list.filter(l => !l.featured);

  selectEl.innerHTML = '';

  const addOptions = (parent, items) => {
    items.forEach(l => {
      const opt = document.createElement('option');
      opt.value = l.code;
      opt.textContent = l.name;
      parent.appendChild(opt);
    });
  };

  if (!expanded) {
    // Collapsed: featured only, plus the selected non-featured language (if any),
    // then a "More…" entry that expands the list when picked.
    const shown = featured.slice();
    if (selected && !shown.some(l => l.code === selected)) {
      const sel = more.find(l => l.code === selected);
      if (sel) shown.push(sel);
    }
    addOptions(selectEl, shown);
    if (more.length) {
      const moreOpt = document.createElement('option');
      moreOpt.value = MT_MORE_SENTINEL;
      moreOpt.textContent = '⋯ More languages…';
      selectEl.appendChild(moreOpt);
    }
  } else {
    if (featured.length) {
      const gCommon = document.createElement('optgroup');
      gCommon.label = 'Common';
      addOptions(gCommon, featured);
      selectEl.appendChild(gCommon);
    }
    if (more.length) {
      const gMore = document.createElement('optgroup');
      gMore.label = 'More languages';
      addOptions(gMore, more);
      selectEl.appendChild(gMore);
    }
  }

  if (selected) {
    const has = Array.from(selectEl.options).some(o => o.value === selected);
    if (has) selectEl.value = selected;
  }
}

// Populate a language <select> showing featured languages by default and a
// "More languages…" entry that expands to the full list on demand.
// `selected` is the code to preselect; `langs` may be a fetched server list
// (falls back to MT_LANGUAGES). Safe to call repeatedly (re-binds cleanly).
function mtPopulateLangSelect(selectEl, selected, langs) {
  if (!selectEl) return;
  const list = (langs && langs.length) ? langs : MT_LANGUAGES;
  selectEl._mtLangs = list;

  _mtRenderLangOptions(selectEl, list, selected, false);

  // Bind the expand handler once. When the user picks "More…", re-render the
  // full list and keep the previously selected real value.
  if (selectEl.dataset.mtLangBound !== '1') {
    selectEl.dataset.mtLangBound = '1';
    selectEl.addEventListener('change', () => {
      if (selectEl.value === MT_MORE_SENTINEL) {
        const prev = selectEl.dataset.mtLangPrev || '';
        _mtRenderLangOptions(selectEl, selectEl._mtLangs || MT_LANGUAGES, prev, true);
      } else {
        selectEl.dataset.mtLangPrev = selectEl.value;
      }
    });
  }
  selectEl.dataset.mtLangPrev = selectEl.value;
}

// Fetch the language list from the server, falling back to the built-in list
// on any error. Always resolves (never rejects) so callers can populate
// dropdowns unconditionally.
async function mtFetchLanguages(serverUrl) {
  if (!serverUrl) return MT_LANGUAGES;
  try {
    const res = await fetch(`${serverUrl}/languages`);
    if (!res.ok) return MT_LANGUAGES;
    const data = await res.json();
    if (data && Array.isArray(data.languages) && data.languages.length) {
      return data.languages.map(l => ({
        code: l.code, name: l.name, featured: !!l.featured,
      }));
    }
  } catch (e) {
    // Server offline or old build without /languages — use fallback.
  }
  return MT_LANGUAGES;
}
