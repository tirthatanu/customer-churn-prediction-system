(() => {
  'use strict';
  const $ = (id) => document.getElementById(id);
  const form = $('prediction-form');
  const fields = ['age', 'gender', 'tenure', 'usage_frequency', 'support_calls', 'payment_delay',
                  'subscription_type', 'contract_length', 'total_spend', 'last_interaction'];
  const numeric = new Set(['age', 'tenure', 'usage_frequency', 'support_calls', 'payment_delay',
                            'total_spend', 'last_interaction']);
  const defaults = {
    age: 30, gender: 'Female', tenure: 12, usage_frequency: 15, support_calls: 4,
    payment_delay: 10, subscription_type: 'Standard', contract_length: 'Monthly',
    total_spend: 906, last_interaction: 9,
  };
  const pct = (value, digits = 1) => `${(value * 100).toFixed(digits)}%`;

  async function getJson(url, options) {
    let response;
    try {
      response = await fetch(url, options);
    } catch (_error) {
      throw new Error('Cannot reach the API. Make sure the FastAPI server is running.');
    }
    let body;
    try { body = await response.json(); } catch (_error) { throw new Error('The server returned an unreadable response.'); }
    if (!response.ok) {
      const problem = body.error;
      if (problem?.code === 'VALIDATION_ERROR') {
        const first = problem.details?.[0];
        throw new Error(first ? `${first.field}: ${first.message}` : 'Please check the form inputs.');
      }
      throw new Error(problem?.message || body.detail || `Request failed (${response.status}).`);
    }
    return body;
  }

  async function loadModelInfo() {
    const status = $('api-status');
    try {
      await getJson('/health');
      const model = await getJson('/model/info');
      $('metric-accuracy').textContent = pct(model.test_metrics.accuracy);
      $('metric-f1').textContent = pct(model.test_metrics.f1);
      $('metric-rows').textContent = Number(model.test_rows).toLocaleString();
      $('metric-model').textContent = model.model_name;
      $('quality-advisory').hidden = model.test_metrics.accuracy >= 0.65;
      status.classList.remove('offline');
      status.querySelector('span:last-child').textContent = 'MODEL ONLINE';
    } catch (_error) {
      status.classList.add('offline');
      status.querySelector('span:last-child').textContent = 'API OFFLINE';
    }
  }

  function parseForm() {
    const object = {};
    for (const name of fields) {
      const input = $(name);
      if (numeric.has(name)) {
        const value = Number(input.value);
        if (!input.value.trim() || !Number.isFinite(value)) throw new Error(`Enter a valid ${name.replace(/_/g, ' ')}.`);
        object[name] = value;
      } else object[name] = input.value;
    }
    return object;
  }

  function showPrediction(result) {
    $('result-empty').hidden = true;
    $('result-content').hidden = false;
    const chance = Math.max(0, Math.min(1, result.probability));
    const percentage = Math.round(chance * 1000) / 10;
    const tone = result.risk_level.toLowerCase();
    const color = tone === 'high' ? '#dc637a' : tone === 'medium' ? '#dba64d' : '#26b795';
    $('prob-ring').style.background = `conic-gradient(${color} 0% ${percentage}%, #eceefa ${percentage}% 100%)`;
    $('prob-label').textContent = pct(chance);
    $('outcome-badge').textContent = result.prediction === 'Churn' ? 'CHURN PREDICTED' : 'RETENTION PREDICTED';
    $('outcome-badge').className = `outcome-badge ${result.prediction === 'Churn' ? 'churn' : 'no-churn'}`;
    $('result-timestamp').textContent = new Date().toLocaleTimeString(undefined, {hour: '2-digit', minute: '2-digit'});
    $('risk-chip').textContent = `${result.risk_level.toUpperCase()} RISK`;
    $('risk-chip').className = `risk-chip ${tone}`;
    $('risk-heading').textContent = tone === 'high' ? 'Retention attention needed' : tone === 'medium' ? 'Monitor this customer' : 'Lower churn likelihood';
    $('risk-description').textContent = tone === 'high'
      ? 'This customer has a higher predicted likelihood of leaving. Consider a timely retention follow-up.'
      : tone === 'medium' ? 'This customer shows moderate predicted churn risk. Track engagement and service experience.'
      : 'The model estimates lower churn risk for this profile. Continue delivering a positive customer experience.';
    $('risk-indicator').style.left = `${Math.max(2, Math.min(98, percentage))}%`;
  }

  function renderHistory(body) {
    const tbody = $('history-body');
    tbody.replaceChildren();
    if (!body.predictions.length) {
      const row = tbody.insertRow();
      const cell = row.insertCell(); cell.colSpan = 4;
      cell.textContent = 'No predictions yet. Run your first prediction above.';
      cell.className = 'history-empty';
    }
    for (const item of body.predictions) {
      const row = tbody.insertRow();
      const stamp = new Date(item.created_at_utc);
      const time = stamp.toLocaleString(undefined, {timeZone: 'UTC', month:'short', day:'numeric', hour:'2-digit', minute:'2-digit'});
      const cols = [time, item.prediction, pct(item.probability), item.risk_level];
      cols.forEach((label, index) => {
        const cell = row.insertCell();
        if (index === 3) {
          const tag = document.createElement('span'); tag.className = `history-tag ${item.risk_level.toLowerCase()}`;
          tag.textContent = label; cell.appendChild(tag);
        } else cell.textContent = label;
      });
    }
    $('history-count').textContent = `Showing ${body.predictions.length} of ${body.count.toLocaleString()} predictions`;
  }

  async function refreshHistory() {
    try { renderHistory(await getJson('/predictions?limit=8')); }
    catch (error) { $('history-count').textContent = error.message; }
  }

  form.addEventListener('submit', async (event) => {
    event.preventDefault();
    const message = $('form-message'); message.hidden = true;
    if (!form.reportValidity()) return;
    const button = $('submit-btn');
    button.disabled = true; button.querySelector('span:first-child').textContent = 'Analyzing customer…';
    try {
      const result = await getJson('/predict', {method: 'POST', headers: {'Content-Type': 'application/json'},
                                                body: JSON.stringify(parseForm())});
      showPrediction(result); await refreshHistory();
    } catch (error) {
      message.hidden = false; message.textContent = error.message;
    } finally {
      button.disabled = false; button.querySelector('span:first-child').textContent = 'Generate prediction';
    }
  });

  $('reset-btn').addEventListener('click', () => {
    for (const [field, value] of Object.entries(defaults)) $(field).value = value;
    $('form-message').hidden = true;
    $('result-empty').hidden = false;
    $('result-content').hidden = true;
  });
  $('refresh-btn').addEventListener('click', refreshHistory);
  loadModelInfo();
  refreshHistory();
})();
