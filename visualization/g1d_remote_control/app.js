const dom = {
  modeText: document.querySelector("#modeText"),
  robotIp: document.querySelector("#robotIp"),
  executeMode: document.querySelector("#executeMode"),
  lastAction: document.querySelector("#lastAction"),
  logBox: document.querySelector("#logBox"),
  stopBtn: document.querySelector("#stopBtn"),
  clearLogBtn: document.querySelector("#clearLogBtn"),
  commandSpeed: document.querySelector("#commandSpeed"),
  commandSpeedValue: document.querySelector("#commandSpeedValue"),
  controlButtons: Array.from(document.querySelectorAll(".control-button")),
  columnButtons: Array.from(document.querySelectorAll('[data-action="column_up"], [data-action="column_down"]')),
};

const ACTION_LABELS = {
  forward: "前进",
  back: "后退",
  turn_left: "左转",
  turn_right: "右转",
  column_up: "升降柱升",
  column_down: "升降柱降",
  stop: "停止",
};

let executeEnabled = false;
let columnConfigured = false;
let activeButton = null;
let holdToken = 0;
let feedbackTimer = null;
let actionClockTimer = null;
let actionClockAction = null;

init();

function init() {
  bindRange(dom.commandSpeed, dom.commandSpeedValue);

  dom.clearLogBtn.addEventListener("click", () => {
    dom.logBox.textContent = "日志已清空";
  });
  dom.stopBtn.addEventListener("click", sendStop);

  for (const button of dom.controlButtons) {
    button.addEventListener("pointerdown", (event) => startAction(event, button));
    button.addEventListener("pointerup", endAction);
    button.addEventListener("pointercancel", endAction);
    button.addEventListener("pointerleave", endAction);
    button.addEventListener("click", (event) => event.preventDefault());
  }

  refreshHealth();
}

function bindRange(input, output) {
  const update = () => {
    output.textContent = Number(input.value).toFixed(2);
  };
  input.addEventListener("input", update);
  update();
}

async function refreshHealth() {
  try {
    const data = await fetchJson("/health");
    executeEnabled = Boolean(data.execute_enabled);
    columnConfigured = Boolean(data.column_control_configured);
    dom.robotIp.textContent = data.robot_ip || "未获取";
    dom.robotIp.title = data.network_interface
      ? `${data.network_interface_kind || "network"}: ${data.network_interface}`
      : "";
    dom.executeMode.textContent = executeEnabled ? "执行模式" : "预览模式";
    dom.executeMode.classList.toggle("is-execute", executeEnabled);
    dom.executeMode.classList.toggle("is-preview", !executeEnabled);
    dom.executeMode.classList.remove("is-offline");
    dom.modeText.textContent = executeEnabled
      ? "会调用机器人 SDK，请保持周围安全"
      : "本地预览，不会调用机器人 SDK";
    updateColumnButtons();
    appendLog(data);
  } catch (error) {
    dom.executeMode.textContent = "离线";
    dom.robotIp.textContent = "离线";
    dom.robotIp.title = "";
    dom.executeMode.classList.remove("is-execute", "is-preview");
    dom.executeMode.classList.add("is-offline");
    dom.modeText.textContent = String(error.message || error);
    columnConfigured = false;
    updateColumnButtons();
  }
}

function startAction(event, button) {
  event.preventDefault();
  if (button.disabled) {
    appendLog({
      ok: false,
      action: button.dataset.action,
      error: "按钮当前不可用",
    });
    return;
  }
  if (activeButton && activeButton !== button) {
    endAction();
  }
  holdToken += 1;
  const token = holdToken;
  activeButton = button;
  button.classList.add("is-active");
  button.setPointerCapture?.(event.pointerId);
  startActionClock(button.dataset.action);
  startHoldCommand(button, token);
}

function updateColumnButtons() {
  for (const button of dom.columnButtons) {
    button.disabled = !columnConfigured;
    button.title = columnConfigured ? "" : "升降柱命令不可用";
  }
}

function endAction() {
  const wasHolding = Boolean(activeButton);
  holdToken += 1;
  stopActionClock();
  if (activeButton) {
    activeButton.classList.remove("is-active");
    activeButton = null;
  }
  if (wasHolding) {
    sendStop({ fromHoldRelease: true });
  }
}

async function startHoldCommand(button, token) {
  const action = button.dataset.action;
  await sendAction(action, { hold: true });
  if (activeButton !== button || token !== holdToken) {
    return;
  }
}

async function sendAction(action, options = {}) {
  const payload = commandPayload(action, options);
  if (!options.hold) {
    showFeedback(action, "pending");
  }
  try {
    const data = await fetchJson("/api/command", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    appendLog(data);
    if (data.ok === false) {
      stopActionClock();
      showFeedback(action, "error", data);
    } else if (!options.hold) {
      showFeedback(action, "ok", data);
    }
  } catch (error) {
    const data = { ok: false, action, error: String(error.message || error) };
    appendLog(data);
    stopActionClock();
    showFeedback(action, "error", data);
  }
}

async function sendStop(options = {}) {
  if (!options.fromHoldRelease) {
    holdToken += 1;
    stopActionClock();
    if (activeButton) {
      activeButton.classList.remove("is-active");
      activeButton = null;
    }
  }
  if (!options.fromHoldRelease) {
    showFeedback("stop", "pending");
  }
  try {
    const data = await fetchJson("/api/stop", { method: "POST" });
    appendLog(data);
    if (!options.fromHoldRelease || data.ok === false) {
      showFeedback("stop", data.ok === false ? "error" : "stop", data);
    }
  } catch (error) {
    const data = { ok: false, action: "stop", error: String(error.message || error) };
    appendLog(data);
    showFeedback("stop", "error", data);
  }
}

function commandPayload(action, options = {}) {
  const speed = Number(dom.commandSpeed.value);
  return {
    action,
    speed,
    hold: Boolean(options.hold),
  };
}

async function fetchJson(url, options = {}) {
  const response = await fetch(url, options);
  const text = await response.text();
  let data;
  try {
    data = text ? JSON.parse(text) : {};
  } catch {
    data = { ok: false, error: text || response.statusText };
  }
  if (!response.ok) {
    throw new Error(data.error || response.statusText);
  }
  return data;
}

function appendLog(data) {
  const lines = dom.logBox.textContent === "等待操作" ? [] : dom.logBox.textContent.split("\n\n");
  lines.unshift(formatLogEntry(data));
  dom.logBox.textContent = lines.slice(0, 8).join("\n\n");
}

function formatLogEntry(data) {
  const time = nowWithMs();
  if (data.service === "g1d_remote_control") {
    const mode = data.execute_enabled ? "执行模式" : "预览模式";
    const column = data.column_control_configured ? "升降已配置" : "升降未配置";
    return `[${time}] 服务正常 | ${mode} | ${column}`;
  }
  const action = ACTION_LABELS[data.command?.action || data.action] || data.command?.action || data.action || "命令";
  if (data.ok === false) {
    return `[${time}] ${action} 失败 | ${data.error || "未知错误"}`;
  }
  const executed = data.executed ? "已执行" : "预览";
  const argv = Array.isArray(data.argv) ? `\n${data.argv.join(" ")}` : "";
  const reason = data.reason ? ` | ${data.reason}` : "";
  return `[${time}] ${action} | ${executed}${reason}${argv}`;
}

function nowWithMs() {
  const date = new Date();
  const base = date.toLocaleTimeString("zh-CN", { hour12: false });
  return `${base}.${String(date.getMilliseconds()).padStart(3, "0")}`;
}

function startActionClock(action) {
  actionClockAction = action;
  updateActionClock();
  if (actionClockTimer) {
    window.clearInterval(actionClockTimer);
  }
  actionClockTimer = window.setInterval(updateActionClock, 100);
}

function stopActionClock() {
  if (actionClockTimer) {
    window.clearInterval(actionClockTimer);
    actionClockTimer = null;
  }
  actionClockAction = null;
}

function updateActionClock() {
  if (!actionClockAction) return;
  const label = ACTION_LABELS[actionClockAction] || actionClockAction;
  dom.lastAction.textContent = `${label} ${nowWithMs()}`;
}

function showFeedback(action, state, data = null) {
  const label = ACTION_LABELS[action] || action || "命令";
  const text = `${label} ${nowWithMs()}`;
  dom.lastAction.textContent = text;
  dom.lastAction.parentElement?.classList.remove("feedback-pending", "feedback-ok", "feedback-error", "feedback-stop");
  dom.lastAction.parentElement?.classList.add(`feedback-${state}`);
  if (feedbackTimer) {
    window.clearTimeout(feedbackTimer);
  }
  feedbackTimer = window.setTimeout(() => {
    dom.lastAction.parentElement?.classList.remove("feedback-pending", "feedback-ok", "feedback-error", "feedback-stop");
  }, 900);
}
