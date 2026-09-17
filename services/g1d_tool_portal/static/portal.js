const text = value => String(value == null ? '' : value);

function escapeHtml(value) {
  return text(value).replace(/[&<>"']/g, character => ({
    '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'
  })[character]);
}

function card(tool) {
  const state = tool.state || 'unknown';
  const stateLabel = state === 'online' ? '在线' : state === 'degraded' ? '异常' : state === 'offline' ? '离线' : '检查中';
  const location = tool.location === 'body' ? '本体' : '盒子';
  const details = tool.details || {};
  const version = details.version ? `<span class="version">v${escapeHtml(details.version)}</span>` : '';
  const features = (details.features || []).map(value => `<span class="feature">${escapeHtml(value)}</span>`).join('');
  const facts = (details.facts || []).map(item => `
    <div class="fact"><span>${escapeHtml(item.label)}</span><strong>${escapeHtml(item.value)}</strong></div>
  `).join('');
  const issue = tool.error ? `<div class="issue" title="${escapeHtml(tool.error)}">${escapeHtml(tool.error)}</div>` : '';
  return `<a class="tool ${escapeHtml(tool.kind)}" href="${escapeHtml(tool.url)}" target="_blank" rel="noreferrer">
    <div class="tool-top">
      <div class="title-line"><h3>${escapeHtml(tool.name)}</h3>${version}</div>
      <span class="status ${state}"><span class="dot"></span>${stateLabel}</span>
    </div>
    <div class="description">${escapeHtml(tool.description)}</div>
    ${features ? `<div class="features">${features}</div>` : ''}
    ${facts ? `<div class="facts">${facts}</div>` : ''}
    ${issue}
    <div class="meta">
      <span class="port">${escapeHtml(tool.port)}</span>
      <span class="host">${location}<br>${escapeHtml(tool.public_host)}:${escapeHtml(tool.port)}</span>
    </div>
  </a>`;
}

async function refresh() {
  try {
    const response = await fetch('/api/catalog', {cache: 'no-store'});
    const data = await response.json();
    document.getElementById('bodyHost').textContent = `本体 ${data.body_public_host}`;
    document.getElementById('boxHost').textContent = `盒子 ${data.box_public_host}`;
    document.getElementById('portalVersion').textContent = `目录 v${data.version || '--'}`;
    document.getElementById('frontendGrid').innerHTML = data.tools.filter(item => item.kind === 'frontend').map(card).join('');
    document.getElementById('runtimeGrid').innerHTML = data.tools.filter(item => item.kind === 'runtime').map(card).join('');
    document.getElementById('apiGrid').innerHTML = data.tools.filter(item => item.kind === 'api').map(card).join('');
    document.getElementById('updated').textContent = `更新 ${new Date().toLocaleTimeString()}`;
  } catch (error) {
    document.getElementById('updated').textContent = `目录读取失败: ${error.message}`;
  }
}

refresh();
setInterval(refresh, 5000);
