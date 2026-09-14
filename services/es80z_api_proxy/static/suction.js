const state = { health: null, status: null, busy: false, isEvs: false, isEs: false };
    const byId = id => document.getElementById(id);

    async function api(path, options = {}) {
      const response = await fetch('/api/v1' + path, {
        cache: 'no-store',
        headers: { 'Content-Type': 'application/json', ...(options.headers || {}) },
        ...options
      });
      const payload = await response.json();
      if (!response.ok || payload.ok === false) {
        const message = payload.error || payload.detail?.error || JSON.stringify(payload.detail || payload);
        throw new Error(message);
      }
      return payload;
    }

    function setClass(node, kind) {
      node.classList.remove('good', 'warn', 'bad');
      if (kind) node.classList.add(kind);
    }

    function statusData(payload) {
      const data = payload && typeof payload.data === 'object' ? payload.data : {};
      return data.result && typeof data.result.status === 'object' ? data.result.status : data;
    }

    function render() {
      const health = state.health || {};
      const status = statusData(state.status || {});
      const device = health.device || state.status?.device || {};
      const service = health.service || state.status?.service || '--';
      const model = String(device.model || '--');
      const identity = (service + ' ' + model).toLowerCase();
      state.isEvs = identity.includes('evs01');
      state.isEs = identity.includes('es80z');

      byId('modelText').textContent = `${model} · ${device.baudrate || '--'} baud · 地址 ${device.slave_id ?? '--'}`;
      byId('serviceValue').textContent = service;
      byId('deviceValue').textContent = model;
      byId('readyValue').textContent = status.is_ready ? '就绪' : '未就绪';
      setClass(byId('readyValue'), status.is_ready ? 'good' : 'bad');
      byId('suctionValue').textContent = status.is_sucked ? '吸取中' : status.is_released ? '已释放' : '未知';
      setClass(byId('suctionValue'), status.is_sucked ? 'warn' : status.is_released ? 'good' : '');
      const fault = Number(status.fault || 0);
      byId('faultValue').textContent = String(fault);
      setClass(byId('faultValue'), fault === 0 ? 'good' : 'bad');
      byId('vacuumValue').textContent = status.vacuum_percent == null ? '无反馈' : `${status.vacuum_percent}%`;
      byId('parameterPanel').classList.toggle('hidden', !state.isEvs);
      byId('esNotice').classList.toggle('hidden', !state.isEs);
      byId('stopButton').textContent = '停止';
      byId('releaseButton').textContent = state.isEs ? '释放（反吹）' : '释放';
      byId('rawJson').textContent = JSON.stringify({ health: state.health, status: state.status }, null, 2);
      byId('updatedAt').textContent = new Date().toLocaleTimeString();
      byId('onlineDot').className = 'dot online';
      byId('onlineText').textContent = '在线';
    }

    async function refresh() {
      if (state.busy) return;
      try {
        const [health, status] = await Promise.all([api('/health'), api('/status')]);
        state.health = health;
        state.status = status;
        render();
      } catch (error) {
        byId('onlineDot').className = 'dot offline';
        byId('onlineText').textContent = '离线';
        byId('eventLog').textContent = `状态读取失败\n${error.message}`;
      }
    }

    async function command(name, extra = {}) {
      if (name === 'suck' && !confirm('确认开始吸取？')) return;
      state.busy = true;
      byId('controlSection').classList.add('busy');
      const started = performance.now();
      try {
        const payload = await api('/command', {
          method: 'POST',
          body: JSON.stringify({ command: name, wait: true, wait_timeout: 3, ...extra })
        });
        byId('eventLog').textContent = `${new Date().toLocaleTimeString()}  ${name} 成功  ${Math.round(performance.now() - started)} ms\n${JSON.stringify(payload, null, 2)}`;
      } catch (error) {
        byId('eventLog').textContent = `${new Date().toLocaleTimeString()}  ${name} 失败\n${error.message}`;
      } finally {
        state.busy = false;
        byId('controlSection').classList.remove('busy');
        await refresh();
      }
    }

    document.querySelectorAll('[data-command]').forEach(button => {
      button.addEventListener('click', () => command(button.dataset.command));
    });
    byId('setParams').addEventListener('click', () => command('set_params', {
      max_vacuum: Number(byId('maxVacuum').value),
      min_vacuum: Number(byId('minVacuum').value)
    }));
    refresh();
    setInterval(refresh, 1500);
