// Laya MLX Xiaohongshu Content Script - Premium Visuals & Snippet Extraction
(() => {
  let isScanning = false;
  let taggedCount = 0;
  let totalLatency = 0;
  let appConfig = {
    enabled: true,
    showTrack: true,
    showFormat: true,
    showIntent: true,
    showViral: true,
    showAudience: true,
    showAd: true,
    showClickbait: true
  };

  // Cyberpunk Cold Obsidian & Neon Color Palettes for Tracks
  const TRACK_THEMES = {
    "科技数码": { bg: "rgba(4, 15, 26, 0.92)", text: "#00f0ff", border: "rgba(0, 240, 255, 0.45)", dot: "#00f0ff" },
    "萌宠动物": { bg: "rgba(24, 14, 4, 0.92)", text: "#f59e0b", border: "rgba(245, 158, 11, 0.45)", dot: "#f59e0b" },
    "美妆护肤": { bg: "rgba(24, 6, 16, 0.92)", text: "#ff2a85", border: "rgba(255, 42, 133, 0.45)", dot: "#ff2a85" },
    "穿搭时尚": { bg: "rgba(18, 8, 26, 0.92)", text: "#d946ef", border: "rgba(217, 70, 239, 0.45)", dot: "#d946ef" },
    "美食探店": { bg: "rgba(22, 14, 4, 0.92)", text: "#fbbf24", border: "rgba(251, 191, 36, 0.45)", dot: "#fbbf24" },
    "职场成长": { bg: "rgba(12, 10, 28, 0.92)", text: "#a855f7", border: "rgba(168, 85, 247, 0.45)", dot: "#a855f7" },
    "家居生活": { bg: "rgba(5, 20, 16, 0.92)", text: "#2dd4bf", border: "rgba(45, 212, 191, 0.45)", dot: "#2dd4bf" },
    "运动健康": { bg: "rgba(4, 22, 14, 0.92)", text: "#00ff9d", border: "rgba(0, 255, 157, 0.45)", dot: "#00ff9d" },
    "母婴育儿": { bg: "rgba(24, 12, 6, 0.92)", text: "#fb923c", border: "rgba(251, 146, 60, 0.45)", dot: "#fb923c" },
    "旅游出行": { bg: "rgba(4, 18, 24, 0.92)", text: "#38bdf8", border: "rgba(56, 189, 248, 0.45)", dot: "#38bdf8" },
    "情感心理": { bg: "rgba(16, 8, 24, 0.92)", text: "#c084fc", border: "rgba(192, 132, 252, 0.45)", dot: "#c084fc" },
    "知识技能": { bg: "rgba(8, 14, 26, 0.92)", text: "#818cf8", border: "rgba(129, 140, 248, 0.45)", dot: "#818cf8" },
    "摄影剪辑": { bg: "rgba(10, 12, 28, 0.92)", text: "#6366f1", border: "rgba(99, 102, 241, 0.45)", dot: "#6366f1" }
  };

  // Load config from background service worker
  function loadConfig() {
    chrome.runtime.sendMessage({ action: "GET_CONFIG" }, (config) => {
      if (config) {
        appConfig = config;
        updateFloatingBar();
        if (appConfig.enabled) {
          triggerScan();
        } else {
          hideAllBadges();
        }
      }
    });
  }

  // Deep extraction of note title, author, AND content snippet from Xiaohongshu card
  function extractNoteData(card) {
    // 1. Title extraction
    const titleEl = card.querySelector('.title, a.title, [class*="title"], .footer span');
    const title = titleEl ? titleEl.textContent.trim() : '';
    if (!title || title.length < 2) return null;

    // 2. Author extraction
    const authorEl = card.querySelector('.author .name, .author-wrapper, .author, [class*="author"]');
    const author = authorEl ? authorEl.textContent.trim() : '';

    // 3. Content Snippet extraction (正文片段探测)
    let snippet = '';

    // A) 检查卡片内显式描述文本 (如长卡片、搜索摘要、.desc 类)
    const descEl = card.querySelector('.desc, .description, .content, .summary, .text, [class*="desc"]');
    if (descEl && descEl.textContent.trim()) {
      snippet = descEl.textContent.trim();
    }

    // B) 检查封面图片的 alt 属性 (小红书前端会在封面 img[alt] 中填入笔记正文摘要)
    if (!snippet || snippet === title) {
      const imgEl = card.querySelector('img[alt]');
      if (imgEl && imgEl.alt && imgEl.alt.trim()) {
        const altText = imgEl.alt.trim();
        // 如果 altText 不仅是标题，且包含较长正文文本
        if (altText !== title && altText.length > 5) {
          snippet = altText;
        }
      }
    }

    // C) 检查卡片副标题或正文预览
    if (!snippet) {
      const subEl = card.querySelector('.sub-title, .note-text, .caption');
      if (subEl && subEl.textContent.trim()) {
        snippet = subEl.textContent.trim();
      }
    }

    // 截取前 200 个字符
    if (snippet && snippet.length > 200) {
      snippet = snippet.substring(0, 200) + '...';
    }

    // 4. Note ID from links
    const linkEl = card.querySelector('a[href*="/explore/"], a[href*="/discovery/item/"]');
    let noteId = null;
    if (linkEl && linkEl.href) {
      const match = linkEl.href.match(/\/(?:explore|item)\/([a-zA-Z0-9]+)/);
      if (match) noteId = match[1];
    }

    return {
      element: card,
      title,
      author,
      snippet: snippet || null,
      note_id: noteId
    };
  }

  // Find all candidate Xiaohongshu note cards
  function findNoteCards() {
    const selectors = [
      '.note-item',
      'section.note-item',
      '.feeds-container section',
      '.feed-container section',
      '.channel-feeds section',
      'div[class*="note-item"]',
      'section[class*="note"]'
    ];

    const cards = [];
    const elements = document.querySelectorAll(selectors.join(','));

    elements.forEach(el => {
      if (el.dataset.layaTagged) return;
      const data = extractNoteData(el);
      if (data) {
        cards.push(data);
      }
    });

    return cards;
  }

  // Trigger scan & batch inference
  async function triggerScan() {
    if (!appConfig.enabled || isScanning) return;
    isScanning = true;

    try {
      const candidates = findNoteCards();
      if (candidates.length === 0) {
        isScanning = false;
        return;
      }

      // Mark as pending immediately
      candidates.forEach(c => {
        c.element.dataset.layaTagged = 'pending';
      });

      // Send in batches of 6
      const BATCH_SIZE = 6;
      for (let i = 0; i < candidates.length; i += BATCH_SIZE) {
        const batch = candidates.slice(i, i + BATCH_SIZE);
        
        chrome.runtime.sendMessage({
          action: "BATCH_TAG_NOTES",
          notes: batch.map(b => ({
            title: b.title,
            snippet: b.snippet,
            note_id: b.note_id
          }))
        }, (response) => {
          if (response && response.results) {
            response.results.forEach((res, idx) => {
              const matched = batch[idx];
              if (matched && res.tags) {
                renderPremiumBadgeStrip(matched.element, res.tags, res.confidence, matched.snippet);
                matched.element.dataset.layaTagged = 'done';
                taggedCount++;
                if (res.latency_ms) totalLatency += res.latency_ms;
              } else if (matched) {
                matched.element.dataset.layaTagged = 'error';
              }
            });
            updateFloatingBar();
          }
        });
      }
    } catch (e) {
      console.error("[Laya XHS] Scan error:", e);
    } finally {
      isScanning = false;
    }
  }

  // Render high-end luxury badge strip onto card
  function renderPremiumBadgeStrip(cardElement, tags, confidence, snippet) {
    const old = cardElement.querySelector('.laya-tag-container');
    if (old) old.remove();

    const container = document.createElement('div');
    container.className = 'laya-tag-container';

    // 1. 标题党警示标签 (Clickbait - 焦点突出)
    if (appConfig.showClickbait && tags.clickbait_level) {
      const isCb = tags.is_clickbait || tags.clickbait_level.includes('标题党') || tags.clickbait_level.includes('噱头');
      const cbBadge = document.createElement('span');
      cbBadge.className = isCb ? 'laya-pill laya-pill-clickbait-warning' : 'laya-pill laya-pill-clickbait-safe';
      
      const dot = document.createElement('span');
      dot.className = isCb ? 'laya-pill-dot laya-dot-warning' : 'laya-pill-dot laya-dot-safe';
      cbBadge.appendChild(dot);

      const text = document.createElement('span');
      text.textContent = tags.clickbait_level;
      cbBadge.appendChild(text);

      // Tooltip breakdown
      const tooltip = createTooltip(
        isCb ? '🚨 标题党与噱头预警' : '✨ 内容客观真实',
        tags.clickbait_reason || (isCb ? '检测到标题夸张煽动，与正文内容存在反差' : '标题与正文客观名副其实'),
        tags.has_snippet ? `正文片段: "${snippet.substring(0, 60)}..."` : '数据源: 笔记标题 (点击展开详情可核验正文全文)'
      );
      cbBadge.appendChild(tooltip);

      container.appendChild(cbBadge);
    }

    // 2. 垂直赛道分类 (Track - 磨砂温润色调)
    if (appConfig.showTrack && tags.track) {
      const theme = TRACK_THEMES[tags.track] || {
        bg: "rgba(8, 14, 24, 0.92)",
        text: "#38bdf8",
        border: "rgba(56, 189, 248, 0.4)",
        dot: "#38bdf8"
      };

      const trackBadge = document.createElement('span');
      trackBadge.className = 'laya-pill laya-pill-track';
      trackBadge.style.background = theme.bg;
      trackBadge.style.color = theme.text;
      trackBadge.style.borderColor = theme.border;

      const dot = document.createElement('span');
      dot.className = 'laya-pill-dot';
      dot.style.background = theme.dot;
      trackBadge.appendChild(dot);

      const text = document.createElement('span');
      text.textContent = '◈ ' + tags.track;
      trackBadge.appendChild(text);

      if (confidence && confidence.track) {
        const tooltip = createTooltip('赛道领域 // TRACK', `分类: ${tags.track}`, `置信度: ${(confidence.track * 100).toFixed(1)}%`);
        trackBadge.appendChild(tooltip);
      }

      container.appendChild(trackBadge);
    }

    // 3. 选题切角 (Format)
    if (appConfig.showFormat && tags.format) {
      const formatBadge = document.createElement('span');
      formatBadge.className = 'laya-pill laya-pill-format';
      formatBadge.textContent = '# ' + tags.format;
      container.appendChild(formatBadge);
    }

    // 4. 商业属性 / 广告带货 (Ad vs Organic)
    if (appConfig.showAd && tags.is_ad !== undefined) {
      const adBadge = document.createElement('span');
      if (tags.is_ad) {
        adBadge.className = 'laya-pill laya-pill-ad';
        adBadge.innerHTML = `<span class="laya-pill-dot" style="background:#ffaa00;box-shadow:0 0 6px #ffaa00;"></span><span>// 商业推广</span>`;
        const tooltip = createTooltip('商业属性 // COMMERCIAL', '判定为品牌商业合作、软广植入或引流带货', `置信度: ${((confidence?.is_ad || 0.8) * 100).toFixed(0)}%`);
        adBadge.appendChild(tooltip);
      } else {
        adBadge.className = 'laya-pill laya-pill-organic';
        adBadge.innerHTML = `<span class="laya-pill-dot" style="background:#00ff9d;box-shadow:0 0 6px #00ff9d;"></span><span>// 原创种草</span>`;
      }
      container.appendChild(adBadge);
    }

    // 5. 爆款潜力评级 (Viral Score)
    if (appConfig.showViral && tags.viral_score !== undefined) {
      const viralBadge = document.createElement('span');
      viralBadge.className = 'laya-pill laya-pill-viral';
      viralBadge.textContent = `⚡ ${tags.viral_score}.0`;
      const tooltip = createTooltip('爆款指数 // VIRAL SCORE', `标题悬念度与情绪张力评级: ${tags.viral_score}/5`, '单次前向校准评级');
      viralBadge.appendChild(tooltip);
      container.appendChild(viralBadge);
    }

    // 6. 目标人群画像 (Audience)
    if (appConfig.showAudience && tags.target_audience) {
      const audBadge = document.createElement('span');
      audBadge.className = 'laya-pill laya-pill-audience';
      audBadge.textContent = `:: ${tags.target_audience}`;
      container.appendChild(audBadge);
    }

    // Insert into card DOM (prefer inside footer before title)
    const footer = cardElement.querySelector('.footer, .info, .card-info');
    if (footer) {
      footer.insertBefore(container, footer.firstChild);
    } else {
      cardElement.appendChild(container);
    }
  }

  function createTooltip(header, body, footer) {
    const tip = document.createElement('div');
    tip.className = 'laya-tooltip';
    tip.innerHTML = `
      <div class="laya-tt-header">${escapeHtml(header)}</div>
      <div class="laya-tt-body">${escapeHtml(body)}</div>
      ${footer ? `<div class="laya-tt-footer">${escapeHtml(footer)}</div>` : ''}
    `;
    return tip;
  }

  function escapeHtml(str) {
    if (!str) return '';
    return String(str).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
  }

  function hideAllBadges() {
    document.querySelectorAll('.laya-tag-container').forEach(el => el.remove());
    document.querySelectorAll('[data-laya-tagged]').forEach(el => delete el.dataset.layaTagged);
  }

  // Floating Monitor Bar at bottom-right
  function createFloatingBar() {
    if (document.getElementById('laya-floating-bar')) return;

    const bar = document.createElement('div');
    bar.id = 'laya-floating-bar';
    bar.className = 'laya-floating-bar';
    bar.innerHTML = `
      <div class="laya-fb-content">
        <span class="laya-fb-dot"></span>
        <span class="laya-fb-title">Laya MLX 打标</span>
        <span class="laya-fb-stat" id="laya-stat-text">已标 0 篇</span>
      </div>
      <button id="laya-toggle-btn" class="laya-fb-toggle" title="开启或暂停自动打标">
        ${appConfig.enabled ? '运行中' : '已暂停'}
      </button>
    `;

    document.body.appendChild(bar);

    document.getElementById('laya-toggle-btn').addEventListener('click', () => {
      appConfig.enabled = !appConfig.enabled;
      chrome.storage.local.set({ enabled: appConfig.enabled });
      updateFloatingBar();
      if (appConfig.enabled) {
        triggerScan();
      } else {
        hideAllBadges();
      }
    });
  }

  function updateFloatingBar() {
    const statText = document.getElementById('laya-stat-text');
    const toggleBtn = document.getElementById('laya-toggle-btn');
    if (statText) {
      statText.textContent = `已标 ${taggedCount} 篇`;
    }
    if (toggleBtn) {
      toggleBtn.textContent = appConfig.enabled ? '运行中' : '已暂停';
      toggleBtn.className = `laya-fb-toggle ${appConfig.enabled ? 'laya-fb-on' : 'laya-fb-off'}`;
    }
  }

  // -------------------------------------------------------------------------
  // LAZY LOADING / INFINITE SCROLL OBSERVER
  // -------------------------------------------------------------------------
  let debounceTimer = null;

  function initObserver() {
    const observer = new MutationObserver((mutations) => {
      let shouldScan = false;
      for (const mutation of mutations) {
        if (mutation.addedNodes && mutation.addedNodes.length > 0) {
          for (const node of mutation.addedNodes) {
            if (node.nodeType === Node.ELEMENT_NODE) {
              if (
                node.classList?.contains('note-item') ||
                node.querySelector?.('.note-item, .title, a[href*="/explore/"]')
              ) {
                shouldScan = true;
                break;
              }
            }
          }
        }
        if (shouldScan) break;
      }

      if (shouldScan) {
        clearTimeout(debounceTimer);
        debounceTimer = setTimeout(() => {
          triggerScan();
        }, 220); // 220ms debounce for smooth scroll
      }
    });

    observer.observe(document.body, {
      childList: true,
      subtree: true
    });

    window.addEventListener('scroll', () => {
      clearTimeout(debounceTimer);
      debounceTimer = setTimeout(() => {
        triggerScan();
      }, 300);
    }, { passive: true });
  }

  function init() {
    loadConfig();
    createFloatingBar();
    initObserver();
    setTimeout(triggerScan, 800);
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();
