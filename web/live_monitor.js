let socket = null;
let predictions = [];
let mockTimer = null;
let mockTimestamp = 0;

const maxHistory = 100;

document.addEventListener("DOMContentLoaded", () => {
  bindControls();
  drawProbabilityChart();
  addEventLog("Dashboard loaded.", "ok");
});

function bindControls() {
  document.getElementById("connectBtn").addEventListener("click", connectWebSocket);
  document.getElementById("disconnectBtn").addEventListener("click", disconnectWebSocket);
  document.getElementById("startSessionBtn").addEventListener("click", startSession);
  document.getElementById("startTrialBtn").addEventListener("click", startTrial);
  document.getElementById("endTrialBtn").addEventListener("click", endTrial);
  document.getElementById("endSessionBtn").addEventListener("click", endSession);
  document.getElementById("resetBtn").addEventListener("click", resetSession);
  document.getElementById("sendSampleBtn").addEventListener("click", sendMockSample);
  document.getElementById("startStreamBtn").addEventListener("click", startMockStream);
  document.getElementById("stopStreamBtn").addEventListener("click", stopMockStream);
}

function connectWebSocket() {
  const url = document.getElementById("wsUrl").value.trim();
  if (!url) {
    addEventLog("WebSocket URL is empty.", "error");
    return;
  }
  if (socket && socket.readyState === WebSocket.OPEN) {
    addEventLog("Already connected.", "ok");
    return;
  }
  setConnectionStatus("connecting");
  addEventLog(`Connecting to ${url}...`);
  try {
    socket = new WebSocket(url);
  } catch (error) {
    setConnectionStatus("error");
    addEventLog(`Connection failed: ${error.message}`, "error");
    return;
  }
  socket.onopen = () => {
    setConnectionStatus("connected");
    addEventLog("Connected.", "ok");
  };
  socket.onclose = () => {
    setConnectionStatus("disconnected");
    stopMockStream();
    addEventLog("Disconnected.");
  };
  socket.onerror = () => {
    setConnectionStatus("error");
    addEventLog("WebSocket error. Check that the API is running.", "error");
  };
  socket.onmessage = handleMessage;
}

function disconnectWebSocket() {
  stopMockStream();
  if (!socket) {
    addEventLog("No active WebSocket connection.");
    return;
  }
  socket.close();
  socket = null;
  setConnectionStatus("disconnected");
}

function sendMessage(message) {
  if (!socket || socket.readyState !== WebSocket.OPEN) {
    addEventLog("Cannot send message before connecting.", "error");
    return false;
  }
  socket.send(JSON.stringify(message));
  addEventLog(`Sent ${message.type}.`, "ok");
  return true;
}

function startSession() {
  sendMessage({
    type: "start_session",
    session_id: document.getElementById("sessionId").value.trim() || "DASHBOARD_SESSION_001"
  });
}

function startTrial() {
  mockTimestamp = 0;
  sendMessage({
    type: "start_trial",
    trial_id: document.getElementById("trialId").value.trim() || "DASHBOARD_TRIAL_001"
  });
}

function endTrial() {
  sendMessage({ type: "end_trial" });
}

function endSession() {
  stopMockStream();
  sendMessage({ type: "end_session" });
}

function resetSession() {
  mockTimestamp = 0;
  predictions = [];
  updateHistoryTable();
  drawProbabilityChart();
  sendMessage({ type: "reset" });
}

function sendMockSample() {
  const noise = () => (Math.random() - 0.5) * 0.06;
  const blink = Math.random() < 0.03 ? 1 : 0;
  const saccade = Math.random() < 0.10 ? 1 : 0;
  const sample = {
    type: "sample",
    data: {
      timestamp: Number(mockTimestamp.toFixed(3)),
      gaze_x: clamp(0.5 + noise(), 0, 1),
      gaze_y: clamp(0.5 + noise(), 0, 1),
      pupil_left: 3.2 + noise(),
      pupil_right: 3.2 + noise(),
      blink: blink,
      fixation: saccade ? 0 : 1,
      saccade: saccade,
      validity: Math.random() < 0.98 ? 1 : 0
    }
  };
  const sent = sendMessage(sample);
  if (sent) {
    mockTimestamp += 0.05;
  }
}

function startMockStream() {
  if (mockTimer) {
    addEventLog("Mock stream is already running.");
    return;
  }
  if (!socket || socket.readyState !== WebSocket.OPEN) {
    addEventLog("Connect before starting mock stream.", "error");
    return;
  }
  mockTimer = setInterval(sendMockSample, 50);
  addEventLog("Mock stream started.", "ok");
}

function stopMockStream() {
  if (mockTimer) {
    clearInterval(mockTimer);
    mockTimer = null;
    addEventLog("Mock stream stopped.");
  }
}

function handleMessage(event) {
  let message;
  try {
    message = JSON.parse(event.data);
  } catch (error) {
    addEventLog("Invalid JSON response from server.", "error");
    return;
  }

  if (message.type === "prediction") {
    updatePredictionPanel(message);
    addPredictionToHistory(message);
    addEventLog(`Prediction received: ${formatNumber(message.probability, 3)} (${message.risk_category})`, "ok");
  } else if (message.type === "error") {
    addEventLog(`Server error: ${message.message}`, "error");
  } else {
    addEventLog(`Received ${message.type || "message"}.`, "ok");
  }
}

function updatePredictionPanel(prediction) {
  document.getElementById("modelType").textContent = prediction.model_type || "-";
  document.getElementById("probability").textContent = formatNumber(prediction.probability, 3);
  document.getElementById("smoothedProbability").textContent = formatNumber(prediction.smoothed_probability, 3);
  document.getElementById("validRatio").textContent = formatNumber(prediction.valid_ratio, 3);
  document.getElementById("sampleCount").textContent = prediction.sample_count ?? "-";
  document.getElementById("latencyMs").textContent = `${formatNumber(prediction.latency_ms, 2)} ms`;
  document.getElementById("currentTimestamp").textContent = formatNumber(prediction.timestamp, 3);
  updateRiskBadge(prediction.risk_category || "insufficient_data");
}

function updateRiskBadge(riskCategory) {
  const badge = document.getElementById("riskBadge");
  badge.textContent = riskCategory;
  badge.className = `risk-badge ${riskCategory}`;
}

function addPredictionToHistory(prediction) {
  predictions.push(prediction);
  if (predictions.length > maxHistory) {
    predictions = predictions.slice(predictions.length - maxHistory);
  }
  updateHistoryTable();
  drawProbabilityChart();
}

function updateHistoryTable() {
  const body = document.getElementById("historyTable");
  body.innerHTML = "";
  const recent = predictions.slice(-20).reverse();
  recent.forEach((prediction, index) => {
    const row = document.createElement("tr");
    row.innerHTML = `
      <td>${predictions.length - index}</td>
      <td>${formatNumber(prediction.timestamp, 3)}</td>
      <td>${formatNumber(prediction.probability, 3)}</td>
      <td>${formatNumber(prediction.smoothed_probability, 3)}</td>
      <td>${prediction.risk_category || "-"}</td>
      <td>${formatNumber(prediction.valid_ratio, 3)}</td>
      <td>${prediction.sample_count ?? "-"}</td>
      <td>${formatNumber(prediction.latency_ms, 2)} ms</td>
    `;
    body.appendChild(row);
  });
}

function drawProbabilityChart() {
  const canvas = document.getElementById("probabilityChart");
  const ctx = canvas.getContext("2d");
  const width = canvas.width;
  const height = canvas.height;
  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = "#ffffff";
  ctx.fillRect(0, 0, width, height);

  drawGridLine(ctx, width, height, 0.4, "#f59e0b");
  drawGridLine(ctx, width, height, 0.7, "#dc2626");
  drawGridLine(ctx, width, height, 0.0, "#cbd5e1");
  drawGridLine(ctx, width, height, 1.0, "#cbd5e1");

  const values = predictions.map(item => Number(item.smoothed_probability ?? item.probability)).filter(value => Number.isFinite(value));
  if (values.length < 2) {
    return;
  }
  ctx.beginPath();
  ctx.strokeStyle = "#2563eb";
  ctx.lineWidth = 2;
  values.forEach((value, index) => {
    const x = (index / (values.length - 1)) * (width - 24) + 12;
    const y = height - 12 - value * (height - 24);
    if (index === 0) {
      ctx.moveTo(x, y);
    } else {
      ctx.lineTo(x, y);
    }
  });
  ctx.stroke();
}

function drawGridLine(ctx, width, height, value, color) {
  const y = height - 12 - value * (height - 24);
  ctx.beginPath();
  ctx.strokeStyle = color;
  ctx.lineWidth = value === 0 || value === 1 ? 1 : 1.5;
  ctx.setLineDash(value === 0 || value === 1 ? [] : [6, 4]);
  ctx.moveTo(12, y);
  ctx.lineTo(width - 12, y);
  ctx.stroke();
  ctx.setLineDash([]);
}

function addEventLog(message, level = "info") {
  const log = document.getElementById("eventLog");
  const entry = document.createElement("div");
  entry.className = `event-entry ${level}`;
  entry.textContent = `[${new Date().toLocaleTimeString()}] ${message}`;
  log.prepend(entry);
  while (log.children.length > 80) {
    log.removeChild(log.lastChild);
  }
}

function setConnectionStatus(status) {
  const badge = document.getElementById("connectionStatus");
  badge.textContent = status;
  badge.className = `status-badge ${status}`;
}

function formatNumber(value, digits) {
  const numeric = Number(value);
  return Number.isFinite(numeric) ? numeric.toFixed(digits) : "-";
}

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}
