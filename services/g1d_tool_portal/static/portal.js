const text = value => String(value == null ? '' : value);
    function escapeHtml(value) {
      return text(value).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
    }
    function card(tool) {
      const state = tool.state || 'unknown';
      const label = state === 'online' ? '在线' : state === 'degraded' ? '异常' : state === 'offline' ? '离线' : '检查中';
      const location = tool.location === 'body' ? '本体' : '盒子';
      return `<a class="tool" href="${escapeHtml(tool.url)}" target="_blank" rel="noreferrer">
        <div class="tool-top"><h3>${escapeHtml(tool.name)}</h3><span class="status ${state}"><span class="dot"></span>${label}</span></div>
        <div class="description">${escapeHtml(tool.description)}</div>
        <div class="meta"><span class="port">${escapeHtml(tool.port)}</span><span class="host">${location}<br>${escapeHtml(tool.public_host)}:${escapeHtml(tool.port)}</span></div>
      </a>`;
    }
    async function refresh() {
      try {
        const response = await fetch('/api/catalog', {cache:'no-store'});
        const data = await response.json();
        document.getElementById('bodyHost').textContent = `本体 ${data.body_public_host}`;
        document.getElementById('boxHost').textContent = `盒子 ${data.box_public_host}`;
        document.getElementById('frontendGrid').innerHTML = data.tools.filter(x => x.kind === 'frontend').map(card).join('');
        document.getElementById('apiGrid').innerHTML = data.tools.filter(x => x.kind === 'api').map(card).join('');
        document.getElementById('updated').textContent = `更新 ${new Date().toLocaleTimeString()}`;
      } catch (error) {
        document.getElementById('updated').textContent = `目录读取失败: ${error.message}`;
      }
    }
    refresh();
    setInterval(refresh, 5000);
