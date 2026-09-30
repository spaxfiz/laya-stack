// Laya MLX Xiaohongshu Tagger - Background Service Worker
const DEFAULT_API_URL = "http://localhost:8088";

// In-memory cache to prevent duplicate requests
const tagCache = new Map();

// Helper to get configuration
async function getConfig() {
  const defaults = {
    apiUrl: DEFAULT_API_URL,
    enabled: true,
    showTrack: true,
    showFormat: true,
    showIntent: true,
    showViral: true,
    showClickbait: true,
    showAudience: true,
    showAd: true
  };
  return new Promise((resolve) => {
    chrome.storage.local.get(defaults, (items) => {
      resolve(items);
    });
  });
}

// Check backend health
async function checkHealth(apiUrl) {
  try {
    const res = await fetch(`${apiUrl}/api/info`, { method: "GET" });
    if (res.ok) {
      const data = await res.json();
      return { ok: true, data };
    }
    return { ok: false, error: `HTTP ${res.status}` };
  } catch (err) {
    return { ok: false, error: err.message };
  }
}

// Request tag for a single note
async function tagNote(apiUrl, note) {
  const cacheKey = `${note.note_id || ''}_${note.title}`;
  if (tagCache.has(cacheKey)) {
    return { ...tagCache.get(cacheKey), fromCache: true };
  }

  try {
    const res = await fetch(`${apiUrl}/api/xhs/tag`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        title: note.title,
        snippet: note.snippet || null,
        note_id: note.note_id || null,
        likes: note.likes || null
      })
    });

    if (!res.ok) {
      const text = await res.text();
      throw new Error(`API Error ${res.status}: ${text}`);
    }

    const data = await res.json();
    tagCache.set(cacheKey, data);
    return data;
  } catch (err) {
    console.error("[Laya Background] Tag note error:", err);
    throw err;
  }
}

// Request tags for batch of notes
async function batchTagNotes(apiUrl, notes) {
  const pendingNotes = [];
  const results = [];

  for (const n of notes) {
    const cacheKey = `${n.note_id || ''}_${n.title}`;
    if (tagCache.has(cacheKey)) {
      results.push({ ...tagCache.get(cacheKey), fromCache: true });
    } else {
      pendingNotes.push(n);
    }
  }

  if (pendingNotes.length > 0) {
    try {
      const res = await fetch(`${apiUrl}/api/xhs/batch_tag`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          notes: pendingNotes.map(n => ({
            title: n.title,
            snippet: n.snippet || null,
            note_id: n.note_id || null,
            likes: n.likes || null
          }))
        })
      });

      if (!res.ok) {
        throw new Error(`Batch API Error: ${res.status}`);
      }

      const batchData = await res.json();
      if (batchData.results) {
        batchData.results.forEach(item => {
          const cacheKey = `${item.note_id || ''}_${item.title}`;
          tagCache.set(cacheKey, item);
          results.push(item);
        });
      }
    } catch (err) {
      console.error("[Laya Background] Batch tag error:", err);
      // Fallback: tag individually
      for (const n of pendingNotes) {
        try {
          const single = await tagNote(apiUrl, n);
          results.push(single);
        } catch (e) {
          results.push({ note_id: n.note_id, title: n.title, error: e.message });
        }
      }
    }
  }

  return results;
}

// Message Listener
chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
  (async () => {
    const config = await getConfig();

    if (request.action === "CHECK_HEALTH") {
      const health = await checkHealth(request.apiUrl || config.apiUrl);
      sendResponse(health);
      return;
    }

    if (request.action === "TAG_NOTE") {
      if (!config.enabled) {
        sendResponse({ disabled: true });
        return;
      }
      try {
        const result = await tagNote(config.apiUrl, request.note);
        sendResponse({ success: true, data: result });
      } catch (err) {
        sendResponse({ success: false, error: err.message });
      }
      return;
    }

    if (request.action === "BATCH_TAG_NOTES") {
      if (!config.enabled) {
        sendResponse({ disabled: true, results: [] });
        return;
      }
      try {
        const results = await batchTagNotes(config.apiUrl, request.notes);
        sendResponse({ success: true, results });
      } catch (err) {
        sendResponse({ success: false, error: err.message });
      }
      return;
    }

    if (request.action === "CLEAR_CACHE") {
      tagCache.clear();
      sendResponse({ success: true });
      return;
    }

    if (request.action === "GET_CONFIG") {
      sendResponse(config);
      return;
    }
  })();

  return true; // Keep message channel open for async response
});
