const analyzeBtn = document.getElementById('analyzeBtn');
const outputSection = document.getElementById('output-section');

const riskLabel = document.getElementById('riskLabel');
const riskBanner = document.getElementById('riskBanner');
const riskMeterFill = document.getElementById('riskMeterFill');
const riskConfidence = document.getElementById('riskConfidence');
const riskIndicatorCount = document.getElementById('riskIndicatorCount');
const indicatorGrid = document.getElementById('indicatorGrid');

const responseCard = document.getElementById('responseCard');
const responseResult = document.getElementById('responseResult');
const responseConfidence = document.getElementById('responseConfidence');
const lrScore = document.getElementById('lrScore');
const rfScore = document.getElementById('rfScore');

const summaryMessage = document.getElementById('summaryMessage');
const summaryResponseWrap = document.getElementById('summaryResponseWrap');
const summaryResponse = document.getElementById('summaryResponse');
const summaryStats = document.getElementById('summaryStats');
const recommendations = document.getElementById('recommendations');

const INDICATOR_DEFS = [
  { key: 'urgency_keywords', label: 'Urgency', icon: '⏱️', desc: 'Pressure to act immediately' },
  { key: 'threat_indicators', label: 'Threats', icon: '⚠️', desc: 'Threats of suspension, fines, or arrest' },
  { key: 'authority_claims', label: 'Authority Claims', icon: '🏛️', desc: 'Poses as bank, government, or law enforcement' },
  { key: 'fake_fee_mentions', label: 'Financial Requests / Fake Fees', icon: '💳', desc: 'Asks for a fee, deposit, or payment' },
  { key: 'emotional_manipulation', label: 'Emotional Manipulation', icon: '❤️', desc: 'Uses fear, guilt, or reward to influence you' },
  { key: 'isolation_attempts', label: 'Isolation Attempts', icon: '🤐', desc: 'Asks you to keep it secret or act alone' }
];

const RISK_STYLES = {
  LOW: { className: 'risk-low', text: 'LOW RISK' },
  MEDIUM: { className: 'risk-medium', text: 'MEDIUM RISK' },
  HIGH: { className: 'risk-high', text: 'HIGH RISK' },
  CRITICAL: { className: 'risk-critical', text: 'CRITICAL RISK — Likely Scam' }
};

const formatNumber = (value) => (typeof value === 'number' ? value.toFixed(2) : '-');

const buildIndicatorGrid = (metadata) => {
  indicatorGrid.innerHTML = '';
  INDICATOR_DEFS.forEach((def) => {
    const detected = !!metadata[def.key];
    const card = document.createElement('div');
    card.className = `result-card indicator-card ${detected ? 'detected' : 'not-detected'}`;
    card.innerHTML = `
      <div class="indicator-icon">${def.icon}</div>
      <div class="indicator-body">
        <h3>${def.label}</h3>
        <p>${def.desc}</p>
      </div>
      <div class="indicator-status">${detected ? 'DETECTED' : 'Not found'}</div>
    `;
    indicatorGrid.appendChild(card);
  });
};

analyzeBtn.addEventListener('click', async () => {
  const message = document.getElementById('message').value.trim();
  const userResponse = document.getElementById('user_response').value.trim();

  if (!message) {
    alert('Please provide a message to analyze.');
    return;
  }

  const payload = {
    message: message,
    user_response: userResponse,
    metadata: {}
  };

  let data;
  analyzeBtn.disabled = true;
  analyzeBtn.textContent = 'Analyzing...';
  try {
    const response = await fetch('/analyze', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });
    data = await response.json().catch(() => ({}));
    if (!response.ok) {
      alert(data.error || 'Unable to analyze the message.');
      return;
    }
  } catch (err) {
    alert('Unable to reach the analysis server.');
    return;
  } finally {
    analyzeBtn.disabled = false;
    analyzeBtn.textContent = 'Analyze Message';
  }

  const risk = data.risk_assessment;
  const metadata = data.analysis.metadata;

  const style = RISK_STYLES[risk.label] || RISK_STYLES.LOW;
  riskLabel.textContent = style.text;
  riskBanner.className = `risk-banner ${style.className}`;
  riskMeterFill.style.width = `${risk.confidence_percent}%`;
  riskConfidence.textContent = `${risk.confidence_percent}% risk score`;
  riskIndicatorCount.textContent = risk.total_indicators;

  buildIndicatorGrid(metadata);

  if (data.response_effectiveness) {
    responseCard.hidden = false;
    const ensemble = data.response_effectiveness.ensemble;
    const lr = data.response_effectiveness.logistic_regression;
    const rf = data.response_effectiveness.random_forest;

    responseResult.textContent = ensemble.label;
    responseResult.className = `result-badge ${ensemble.label === 'EFFECTIVE' ? 'success' : 'failure'}`;
    responseConfidence.textContent = `${(ensemble.score * 100).toFixed(1)}%`;
    lrScore.textContent = `${(lr.score * 100).toFixed(1)}% (${lr.label})`;
    rfScore.textContent = `${(rf.score * 100).toFixed(1)}% (${rf.label})`;
  } else {
    responseCard.hidden = true;
  }

  summaryMessage.textContent = data.message;
  if (data.user_response) {
    summaryResponseWrap.hidden = false;
    summaryResponse.textContent = data.user_response;
  } else {
    summaryResponseWrap.hidden = true;
  }

  const scamStats = data.analysis.scam || {};
  summaryStats.innerHTML = `
    <div><strong>Word count:</strong> ${scamStats.word_count ?? '-'}</div>
    <div><strong>Sentiment polarity:</strong> ${formatNumber(scamStats.polarity)}</div>
    <div><strong>Subjectivity:</strong> ${formatNumber(scamStats.subjectivity)}</div>
  `;

  recommendations.innerHTML = '';
  data.recommendations.forEach((text) => {
    const item = document.createElement('li');
    item.className = 'recommendation-item';
    item.textContent = text;
    recommendations.appendChild(item);
  });

  outputSection.hidden = false;
  outputSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
});
