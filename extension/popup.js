// Popup configuration logic
document.addEventListener('DOMContentLoaded', () => {
  const apiUrlInput = document.getElementById('api-url');
  const showTrack = document.getElementById('show-track');
  const showFormat = document.getElementById('show-format');
  const showAd = document.getElementById('show-ad');
  const showViral = document.getElementById('show-viral');
  const showClickbait = document.getElementById('show-clickbait');
  const showAudience = document.getElementById('show-audience');
  const autoEnabled = document.getElementById('auto-enabled');

  const connStatus = document.getElementById('conn-status');
  const connText = document.getElementById('conn-text');
  const modelBadge = document.getElementById('model-badge');

  const saveBtn = document.getElementById('save-btn');
  const openStudioBtn = document.getElementById('open-studio-btn');

  // Load existing config
  chrome.storage.local.get({
    apiUrl: 'http://localhost:8088',
    enabled: true,
    showTrack: true,
    showFormat: true,
    showIntent: true,
    showViral: true,
    showClickbait: true,
    showAudience: true,
    showAd: true
  }, (items) => {
    apiUrlInput.value = items.apiUrl;
    autoEnabled.checked = items.enabled;
    showTrack.checked = items.showTrack;
    showFormat.checked = items.showFormat;
    showAd.checked = items.showAd;
    showViral.checked = items.showViral;
    if (showClickbait) showClickbait.checked = items.showClickbait;
    showAudience.checked = items.showAudience;

    checkBackend(items.apiUrl);
  });

  // Check connection to backend
  function checkBackend(url) {
    connText.textContent = '检测中...';
    connStatus.style.background = 'rgba(234, 179, 8, 0.15)';
    connStatus.style.color = '#facc15';
    connStatus.style.borderColor = 'rgba(234, 179, 8, 0.3)';

    chrome.runtime.sendMessage({ action: 'CHECK_HEALTH', apiUrl: url }, (res) => {
      if (res && res.ok) {
        connText.textContent = '已连接';
        connStatus.style.background = 'rgba(16, 185, 129, 0.15)';
        connStatus.style.color = '#34d399';
        connStatus.style.borderColor = 'rgba(16, 185, 129, 0.3)';
        if (res.data && res.data.model) {
          modelBadge.textContent = res.data.model.replace('local-cache: ', '');
        }
      } else {
        connText.textContent = '未连接';
        connStatus.style.background = 'rgba(239, 68, 68, 0.15)';
        connStatus.style.color = '#f87171';
        connStatus.style.borderColor = 'rgba(239, 68, 68, 0.3)';
        modelBadge.textContent = '服务离线';
      }
    });
  }

  apiUrlInput.addEventListener('change', () => {
    checkBackend(apiUrlInput.value.trim());
  });

  // Save Settings
  saveBtn.addEventListener('click', () => {
    const config = {
      apiUrl: apiUrlInput.value.trim() || 'http://localhost:8088',
      enabled: autoEnabled.checked,
      showTrack: showTrack.checked,
      showFormat: showFormat.checked,
      showAd: showAd.checked,
      showViral: showViral.checked,
      showClickbait: showClickbait ? showClickbait.checked : true,
      showAudience: showAudience.checked
    };

    chrome.storage.local.set(config, () => {
      saveBtn.textContent = '✓ 已保存并下发';
      chrome.runtime.sendMessage({ action: "CLEAR_CACHE" });

      // Notify active Xiaohongshu tabs to reload config and rescan
      chrome.tabs.query({ url: '*://*.xiaohongshu.com/*' }, (tabs) => {
        tabs.forEach(tab => {
          chrome.tabs.reload(tab.id);
        });
      });

      setTimeout(() => {
        saveBtn.textContent = '保存配置并重新扫描';
        saveBtn.style.background = '#ff2442';
      }, 1500);
    });
  });

  // Open local studio
  openStudioBtn.addEventListener('click', () => {
    const url = apiUrlInput.value.trim() || 'http://localhost:8088';
    chrome.tabs.create({ url });
  });
});
