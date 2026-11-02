/**
 * FraudShield Drift-AI: Client Controller
 * Handles streamlined 9-feature manual entry, preset loading, and real-time risk evaluation.
 */

// Key feature names exposed on the UI
const VISIBLE_FEATURES = ['Amount', 'V14', 'V17', 'V12', 'V10', 'V23', 'V22', 'V25', 'V15'];

const STATE = {
  metadata: null,
  activeSection: 'inference',
  currentFeatures: {},
};

document.addEventListener('DOMContentLoaded', async () => {
  try {
    const res = await fetch('/api/metadata');
    STATE.metadata = await res.json();

    // Initialize all 29 features to baseline mean (0.0)
    STATE.metadata.features.forEach(col => {
      STATE.currentFeatures[col] = STATE.metadata.feature_stats[col] ? STATE.metadata.feature_stats[col].mean : 0.0;
    });

    // Load the initial test case: Concept Drift Evasive Fraud
    const initialPreset = STATE.metadata.presets.find(p => p.id === 'preset_concept_drift_fraud');
    if (initialPreset) {
      populateFormValues(initialPreset.values);
      executeInference();
    }
  } catch (err) {
    console.error('Initialization error:', err);
  }
});

// Section Switcher (Inference vs Results)
function switchSection(sectionId) {
  document.querySelectorAll('.nav-tab').forEach(t => t.classList.remove('active'));
  document.querySelectorAll('.page-section').forEach(s => s.classList.remove('active'));

  if (sectionId === 'inference') {
    document.getElementById('tab-inference-btn').classList.add('active');
    document.getElementById('section-inference').classList.add('active');
  } else {
    document.getElementById('tab-results-btn').classList.add('active');
    document.getElementById('section-results').classList.add('active');
  }
  STATE.activeSection = sectionId;
}

// Preset Loader
function applyPreset(presetId, chipEl) {
  if (!STATE.metadata) return;

  document.querySelectorAll('.preset-chip').forEach(c => c.classList.remove('active'));
  if (chipEl) chipEl.classList.add('active');

  const preset = STATE.metadata.presets.find(p => p.id === presetId);
  if (preset) {
    populateFormValues(preset.values);
    executeInference();
  }
}

// Clear all inputs
function clearFields(chipEl) {
  document.querySelectorAll('.preset-chip').forEach(c => c.classList.remove('active'));
  if (chipEl) chipEl.classList.add('active');

  VISIBLE_FEATURES.forEach(col => {
    const input = document.getElementById(`input-${col}`);
    if (input) input.value = '';
    STATE.currentFeatures[col] = 0.0;
  });
}

// Populate Form Inputs
function populateFormValues(values) {
  // Update internal state
  Object.assign(STATE.currentFeatures, values);

  // Update visible fields
  VISIBLE_FEATURES.forEach(col => {
    const input = document.getElementById(`input-${col}`);
    if (input && values[col] !== undefined) {
      input.value = values[col].toFixed(2);
    }
  });
}

// Execute Inference on Form Submit
async function executeInference(e) {
  if (e) e.preventDefault();

  const submitBtn = document.getElementById('btn-submit');
  if (submitBtn) {
    submitBtn.disabled = true;
    submitBtn.innerHTML = `Analyzing...`;
  }

  // Read visible inputs into state
  VISIBLE_FEATURES.forEach(col => {
    const input = document.getElementById(`input-${col}`);
    if (input) {
      STATE.currentFeatures[col] = input.value !== '' ? parseFloat(input.value) : 0.0;
    }
  });

  const modelId = document.getElementById('model-select').value;
  const threshold = parseFloat(document.getElementById('threshold-val').value) || 0.50;

  try {
    const res = await fetch('/api/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        model: modelId,
        threshold: threshold,
        features: STATE.currentFeatures,
      }),
    });

    if (!res.ok) throw new Error(`HTTP Error: ${res.status}`);
    const data = await res.json();

    renderInferenceOutput(data);

  } catch (err) {
    console.error('Inference error:', err);
  } finally {
    if (submitBtn) {
      submitBtn.disabled = false;
      submitBtn.innerHTML = `
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2"><path d="M5 12h14"/><path d="m12 5 7 7-7 7"/></svg>
        Analyze Transaction Risk
      `;
    }
  }
}

// Render Risk Assessment Output
function renderInferenceOutput(data) {
  const { is_fraud, probability_pct, risk_level, threshold, latency_ms, comparison, model_id } = data;

  // Latency
  document.getElementById('latency-tag').textContent = `⚡ ${latency_ms} ms`;

  // Status Banner
  const banner = document.getElementById('verdict-banner');
  const iconWrap = document.getElementById('verdict-icon-wrap');
  const headline = document.getElementById('verdict-headline');
  const subline = document.getElementById('verdict-subline');

  if (is_fraud) {
    banner.className = 'verdict-banner status-fraud';
    iconWrap.innerHTML = `🚨`;
    headline.textContent = 'DECLINED — FRAUD DETECTED';
    subline.textContent = `Transaction probability (${probability_pct.toFixed(1)}%) exceeds active threshold (${threshold.toFixed(2)}).`;
    document.getElementById('val-action').textContent = 'Decline & Alert Cardholder';
    document.getElementById('val-action').style.color = '#ef4444';
  } else {
    banner.className = 'verdict-banner status-approved';
    iconWrap.innerHTML = `✅`;
    headline.textContent = 'APPROVED — LOW RISK';
    subline.textContent = `Feature values match authentic customer transaction patterns.`;
    document.getElementById('val-action').textContent = 'Approve & Settle Transaction';
    document.getElementById('val-action').style.color = '#10b981';
  }

  // Meter Score Bar
  const meterText = document.getElementById('meter-score-text');
  const meterFill = document.getElementById('meter-bar-fill');
  const thresholdPin = document.getElementById('threshold-pin');

  meterText.textContent = `${probability_pct.toFixed(1)}%`;
  meterFill.style.width = `${Math.min(100, Math.max(0, probability_pct))}%`;
  meterFill.className = `meter-bar-fill ${is_fraud ? 'fill-danger' : ''}`;
  thresholdPin.style.left = `${threshold * 100}%`;

  // Summary Table
  document.getElementById('val-risk-level').textContent = risk_level;
  document.getElementById('val-risk-level').style.color = is_fraud ? '#ef4444' : '#10b981';
  document.getElementById('val-model').textContent = model_id;

  // Concept Drift Diagnostic Box
  renderDriftDiagnostic(comparison);
}

// Render Concept Drift Diagnostic
function renderDriftDiagnostic(comparison) {
  const { frozen_model, adapted_model, drift_evasion_detected, historical_caught_both } = comparison;

  const box = document.getElementById('drift-diagnostic-box');
  const statusPill = document.getElementById('drift-status-pill');
  const alertText = document.getElementById('drift-alert-text');

  const frozenScore = document.getElementById('frozen-score-pill');
  const frozenVerdict = document.getElementById('frozen-verdict-pill');
  const adaptedScore = document.getElementById('adapted-score-pill');
  const adaptedVerdict = document.getElementById('adapted-verdict-pill');

  frozenScore.textContent = `${frozen_model.probability_pct.toFixed(1)}%`;
  frozenVerdict.textContent = frozen_model.is_fraud ? '❌ Flagged Fraud' : '🟢 Approved (Benign)';
  frozenVerdict.style.color = frozen_model.is_fraud ? '#ef4444' : '#10b981';

  adaptedScore.textContent = `${adapted_model.probability_pct.toFixed(1)}%`;
  adaptedVerdict.textContent = adapted_model.is_fraud ? '🚨 Caught Fraud' : '🟢 Approved (Benign)';
  adaptedVerdict.style.color = adapted_model.is_fraud ? '#ef4444' : '#10b981';

  if (drift_evasion_detected) {
    box.className = 'drift-box alert-active';
    statusPill.className = 'drift-status-pill pill-alert';
    statusPill.textContent = 'DRIFT EVASION ACTIVE';
    alertText.innerHTML = `
      <strong>🚨 Concept Drift Evasion Detected:</strong> Fraudsters shifted evasion signals to swapped features (V23, V22). The <em>Unmaintained Frozen Model</em> was tricked (scored <strong>${frozen_model.probability_pct.toFixed(1)}%</strong>), while our <em>Periodic Retrained Model</em> caught the evasion (<strong>${adapted_model.probability_pct.toFixed(1)}% risk</strong>).
    `;
  } else if (historical_caught_both) {
    box.className = 'drift-box';
    statusPill.className = 'drift-status-pill';
    statusPill.textContent = 'Pre-Drift Fraud';
    alertText.innerHTML = `
      <strong>Pre-Drift Fraud Signature:</strong> Both models identified this transaction as fraudulent based on classical historical indicators.
    `;
  } else {
    box.className = 'drift-box';
    statusPill.className = 'drift-status-pill';
    statusPill.textContent = 'In Agreement';
    alertText.innerHTML = `
      Both classification models are in mutual agreement regarding this transaction. No drift discrepancy detected.
    `;
  }
}
