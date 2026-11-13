/**
 * CREDENCEX - Modern Interactive Verification Client Logic
 * Handles real-time verification requests, progress stepper, animated gauges,
 * evidence filtering, and modal managers.
 */

document.addEventListener('DOMContentLoaded', () => {

  // ==========================================================================
  // DEMO MODE — Simulated API responses for static deployment (HuggingFace Spaces)
  // ==========================================================================
  const DEMO_MODE = true;

  const DEMO_RESPONSES = {
    'flat': {
      verdict: 'Contradicted', confidence: 0.96, cached: false, execution_time: 3.42,
      claim: 'The Earth is flat and NASA has been covering up the truth about the planetary curve.',
      category: 'science', claimant: 'NASA',
      detected_authorities: [{ name: 'NASA (National Aeronautics and Space Administration)', role: 'Primary Authority', official_handles: ['nasa.gov'] }],
      primary_origins: [{ name: 'NASA', role: 'Primary Authority', official_handles: ['nasa.gov'] }],
      truth_analysis: {
        status: 'rumor', badge_label: 'DEBUNKED RUMOR / FALSE', badge_color: 'rose',
        claim_statement: 'The Earth is flat and NASA has been covering up the truth.',
        actual_truth: 'The Earth is an oblate spheroid. This has been confirmed by satellite imagery, physics, GPS, aviation, and centuries of scientific observation. NASA and every space agency globally have published extensive photographic and telemetric evidence.',
        rumor_statement: 'The Earth is flat and NASA has been covering up the truth.',
        evidence_basis: 'Refuted by NASA records, ESA, ISRO, satellite imagery, and peer-reviewed astrophysics.',
        verdict_meaning: 'This is a debunked conspiracy theory with zero scientific basis.'
      },
      probabilities: { entailment: 0.02, contradiction: 0.96, neutral: 0.02 },
      entities: { ORG: ['NASA'], CONCEPT: ['Planetary Curve'], FIELD: ['Earth Science'] },
      evidence: [
        { text: 'NASA has published thousands of images from ISS and satellites confirming Earth is an oblate spheroid.', url: 'https://nasa.gov', stance: 'contradiction', stance_score: 0.98, credibility_score: 0.99, source_tier: 'primary_origin' },
        { text: 'The flat Earth claim has been rated as False by multiple fact-checking organizations including Snopes and PolitiFact.', url: 'https://snopes.com', stance: 'contradiction', stance_score: 0.95, credibility_score: 0.95, source_tier: 'fact_checker' },
        { text: 'Satellite geodesy and GPS systems operate based on an ellipsoidal Earth model, confirming its shape.', url: 'https://nature.com', stance: 'contradiction', stance_score: 0.92, credibility_score: 0.99, source_tier: 'primary_origin' }
      ]
    },
    'gates': {
      verdict: 'Contradicted', confidence: 0.93, cached: false, execution_time: 4.18,
      claim: 'Bill Gates patented an implantable microchip sensor numbered 060606 for COVID vaccines.',
      category: 'technology', claimant: 'Microsoft / Bill Gates',
      detected_authorities: [{ name: 'Microsoft Corporation', role: 'Primary Authority' }, { name: 'World Intellectual Property Organization', role: 'Regulatory Body' }],
      primary_origins: [{ name: 'WIPO', role: 'Patent Authority' }],
      truth_analysis: {
        status: 'rumor', badge_label: 'DEBUNKED RUMOR / FALSE', badge_color: 'rose',
        claim_statement: 'Bill Gates patented microchip 060606 for COVID vaccines.',
        actual_truth: 'Patent WO/2020/060606 is a Microsoft patent for a cryptocurrency system using body activity data. It has nothing to do with implantable microchips or vaccines. The patent describes wearable sensors, not implants.',
        rumor_statement: 'Bill Gates patented an implantable microchip numbered 060606.',
        evidence_basis: 'Refuted by WIPO patent records and Reuters/AP fact-checks.',
        verdict_meaning: 'This is a debunked conspiracy theory based on misinterpretation of a patent number.'
      },
      probabilities: { entailment: 0.03, contradiction: 0.93, neutral: 0.04 },
      entities: { PERSON: ['Bill Gates'], ORG: ['Microsoft', 'WIPO'], NUMBER: ['060606'] },
      evidence: [
        { text: 'Patent WO2020060606 is for a cryptocurrency system using body activity, not an implantable microchip.', url: 'https://reuters.com/fact-check', stance: 'contradiction', stance_score: 0.96, credibility_score: 0.95, source_tier: 'fact_checker' },
        { text: 'The patent describes wearable devices that monitor body activity for mining cryptocurrency, unrelated to vaccines.', url: 'https://factcheck.org', stance: 'contradiction', stance_score: 0.94, credibility_score: 0.95, source_tier: 'fact_checker' }
      ]
    },
    'jwst': {
      verdict: 'Supported', confidence: 0.94, cached: false, execution_time: 3.87,
      claim: 'NASA James Webb Space Telescope detected water vapor and carbon dioxide on exoplanet K2-18b.',
      category: 'astronomy', claimant: 'NASA',
      detected_authorities: [{ name: 'NASA', role: 'Primary Authority', official_handles: ['nasa.gov'] }],
      primary_origins: [{ name: 'NASA Goddard Space Flight Center', role: 'Primary Authority' }],
      truth_analysis: {
        status: 'verified_truth', badge_label: 'VERIFIED ACTUAL TRUTH', badge_color: 'emerald',
        claim_statement: 'JWST detected water vapor and CO2 on K2-18b.',
        actual_truth: 'NASA confirmed that JWST spectroscopy detected carbon dioxide and methane in the atmosphere of K2-18b, a sub-Neptune exoplanet in the habitable zone. Water vapor indicators were also found.',
        rumor_statement: 'No conflicting rumor detected; statement is verified.',
        evidence_basis: 'Corroborated by NASA press releases, peer-reviewed papers in The Astrophysical Journal Letters, and ESA.',
        verdict_meaning: 'This scientific discovery is verified by primary space agency data.'
      },
      probabilities: { entailment: 0.94, contradiction: 0.02, neutral: 0.04 },
      entities: { ORG: ['NASA', 'ESA'], INSTRUMENT: ['James Webb Space Telescope'], CELESTIAL: ['K2-18b'] },
      evidence: [
        { text: 'NASA JWST observations revealed carbon dioxide and methane in K2-18b atmosphere, suggesting potential habitability conditions.', url: 'https://nasa.gov/jwst', stance: 'entailment', stance_score: 0.97, credibility_score: 0.99, source_tier: 'primary_origin' },
        { text: 'Published in The Astrophysical Journal Letters, the JWST spectrum of K2-18b shows clear signatures of carbon-bearing molecules.', url: 'https://iopscience.iop.org', stance: 'entailment', stance_score: 0.93, credibility_score: 0.99, source_tier: 'primary_origin' }
      ]
    },
    'default': {
      verdict: 'Unverified', confidence: 0.52, cached: false, execution_time: 2.91,
      claim: '',
      category: 'general',
      detected_authorities: [],
      primary_origins: [],
      truth_analysis: {
        status: 'unverified_rumor', badge_label: 'UNCONFIRMED / POTENTIAL RUMOR', badge_color: 'amber',
        claim_statement: '',
        actual_truth: 'No official notification or verified press wire exists to corroborate this claim. The statement lacks authenticated evidentiary backing.',
        rumor_statement: '',
        evidence_basis: 'Insufficient evidence from authoritative sources to confirm or deny.',
        verdict_meaning: 'This claim could not be verified with available evidence sources.'
      },
      probabilities: { entailment: 0.22, contradiction: 0.26, neutral: 0.52 },
      entities: {},
      evidence: [
        { text: 'No definitive corroborating or refuting evidence was found from accredited fact-checking networks.', url: '', stance: 'neutral', stance_score: 0.52, credibility_score: 0.60, source_tier: 'general_web' }
      ]
    }
  };

  function getDemoResponse(claimText) {
    const lower = claimText.toLowerCase();
    let resp;
    if (lower.includes('flat') && lower.includes('earth')) resp = JSON.parse(JSON.stringify(DEMO_RESPONSES['flat']));
    else if (lower.includes('gates') || lower.includes('060606') || lower.includes('microchip')) resp = JSON.parse(JSON.stringify(DEMO_RESPONSES['gates']));
    else if (lower.includes('magnetic') && lower.includes('vaccine')) {
      resp = JSON.parse(JSON.stringify(DEMO_RESPONSES['flat']));
      resp.claim = claimText; resp.confidence = 0.91;
      resp.truth_analysis.claim_statement = claimText; resp.truth_analysis.rumor_statement = claimText;
      resp.truth_analysis.actual_truth = 'COVID-19 vaccines do not contain magnetic materials. This claim has been debunked by WHO, CDC, and multiple peer-reviewed studies.';
    }
    else if (lower.includes('webb') || lower.includes('k2-18') || lower.includes('exoplanet')) resp = JSON.parse(JSON.stringify(DEMO_RESPONSES['jwst']));
    else {
      resp = JSON.parse(JSON.stringify(DEMO_RESPONSES['default']));
      resp.claim = claimText; resp.truth_analysis.claim_statement = claimText; resp.truth_analysis.rumor_statement = claimText;
    }
    if (!resp.claim) resp.claim = claimText;
    return resp;
  }
  // --------------------------------------------------------------------------
  // DOM Elements
  // --------------------------------------------------------------------------
  const statusPulse = document.getElementById('status-pulse');
  const statusText = document.getElementById('status-text');

  // Tabs
  const tabBtnText = document.getElementById('tab-btn-text');
  const tabBtnImage = document.getElementById('tab-btn-image');
  const panelText = document.getElementById('panel-text');
  const panelImage = document.getElementById('panel-image');

  // Text Tab Elements
  const claimInput = document.getElementById('claim-input');
  const claimCharCount = document.getElementById('claim-char-count');
  const btnClearText = document.getElementById('btn-clear-text');
  const maxSourcesInput = document.getElementById('max-sources-input');
  const maxSourcesVal = document.getElementById('max-sources-val');
  const bypassCacheCheckbox = document.getElementById('bypass-cache-checkbox');
  const btnVerifyText = document.getElementById('btn-verify-text');
  const exampleChips = document.getElementById('example-chips');

  // Image Tab Elements
  const imageDropzone = document.getElementById('image-dropzone');
  const imageFileInput = document.getElementById('image-file-input');
  const dropzonePrompt = document.getElementById('dropzone-prompt');
  const imagePreviewWrapper = document.getElementById('image-preview-wrapper');
  const imagePreview = document.getElementById('image-preview');
  const previewFilename = document.getElementById('preview-filename');
  const previewFilesize = document.getElementById('preview-filesize');
  const btnRemoveImage = document.getElementById('btn-remove-image');
  const imageClaimInput = document.getElementById('image-claim-input');
  const maxSourcesImageInput = document.getElementById('max-sources-image-input');
  const maxSourcesImageVal = document.getElementById('max-sources-image-val');
  const btnVerifyImage = document.getElementById('btn-verify-image');

  // Progress Stepper
  const progressStepper = document.getElementById('progress-stepper');
  const stepperStatusMsg = document.getElementById('stepper-status-msg');
  const stepperTimer = document.getElementById('stepper-timer');
  const step1 = document.getElementById('step-1');
  const step2 = document.getElementById('step-2');
  const step3 = document.getElementById('step-3');
  const step4 = document.getElementById('step-4');

  // Results Showcase
  const resultsSection = document.getElementById('results-section');
  const verdictCard = document.getElementById('verdict-card');
  const verdictPill = document.getElementById('verdict-pill');
  const verdictText = document.getElementById('verdict-text');
  const verdictCacheBadge = document.getElementById('verdict-cache-badge');
  const cacheLatency = document.getElementById('cache-latency');
  const gaugeFill = document.getElementById('gauge-fill');
  const gaugeValue = document.getElementById('gauge-value');
  const resultClaimText = document.getElementById('result-claim-text');
  const verdictExplanation = document.getElementById('verdict-explanation');
  const metaExecutionTime = document.getElementById('meta-execution-time');
  const metaEvidenceCount = document.getElementById('meta-evidence-count');
  const metaDominantStance = document.getElementById('meta-dominant-stance');

  // Truth Section Elements (Rumor vs. Actual Truth Matrix)
  const truthSectionBox = document.getElementById('truth-section-box');
  const truthVerdictMeaning = document.getElementById('truth-verdict-meaning');
  const truthStatusBadge = document.getElementById('truth-status-badge');
  const truthBadgeText = document.getElementById('truth-badge-text');
  const truthRumorCol = document.getElementById('truth-rumor-col');
  const truthColRumorTag = document.getElementById('truth-col-rumor-tag');
  const truthRumorText = document.getElementById('truth-rumor-text');
  const truthRumorBanner = document.getElementById('truth-rumor-banner');
  const truthRumorBannerText = document.getElementById('truth-rumor-banner-text');
  const truthFactCol = document.getElementById('truth-fact-col');
  const truthColFactTag = document.getElementById('truth-col-fact-tag');
  const truthFactText = document.getElementById('truth-fact-text');
  const truthFactBanner = document.getElementById('truth-fact-banner');
  const truthFactBannerText = document.getElementById('truth-fact-banner-text');
  const truthFoundationText = document.getElementById('truth-foundation-text');

  // Deep Dive Cards
  const valEntailment = document.getElementById('val-entailment');
  const barEntailment = document.getElementById('bar-entailment');
  const valContradiction = document.getElementById('val-contradiction');
  const barContradiction = document.getElementById('bar-contradiction');
  const valNeutral = document.getElementById('val-neutral');
  const barNeutral = document.getElementById('bar-neutral');

  const entitiesContainer = document.getElementById('entities-container');
  const entityPillsWrap = document.getElementById('entity-pills-wrap');
  const noEntitiesMsg = document.getElementById('no-entities-msg');

  const cardForensics = document.getElementById('card-forensics');
  const valAuthenticity = document.getElementById('val-authenticity');
  const barAuthenticity = document.getElementById('bar-authenticity');
  const valClip = document.getElementById('val-clip');
  const barClip = document.getElementById('bar-clip');
  const clipStatusSub = document.getElementById('clip-status-sub');

  // Evidence Explorer
  const evidenceCountBadge = document.getElementById('evidence-count-badge');
  const evidenceList = document.getElementById('evidence-list');
  const evidenceFilters = document.getElementById('evidence-filters');

  // Toolbar
  const btnCopySummary = document.getElementById('btn-copy-summary');
  const copySummaryText = document.getElementById('copy-summary-text');
  const btnDownloadJson = document.getElementById('btn-download-json');
  const btnNewSearch = document.getElementById('btn-new-search');

  // Modals
  const navSourcesBtn = document.getElementById('nav-sources-btn');
  const modalSources = document.getElementById('modal-sources');
  const btnCloseSourcesModal = document.getElementById('btn-close-sources-modal');
  const sourcesTableBody = document.getElementById('sources-table-body');
  const toastContainer = document.getElementById('toast-container');

  // State
  let currentResult = null;
  let selectedFile = null;
  let stepperInterval = null;
  let stepperStartTime = null;

  // --------------------------------------------------------------------------
  // API Health & Initialization
  // --------------------------------------------------------------------------
  async function checkHealth() {
    if (DEMO_MODE) {
      statusPulse.classList.add('online');
      statusText.textContent = 'Demo Mode Active';
      return;
    }
    try {
      const res = await fetch('/api/v1/health');
      if (res.ok) {
        statusPulse.classList.add('online');
        statusText.textContent = 'Engine Active';
      } else {
        statusPulse.classList.remove('online');
        statusText.textContent = 'Service Degraded';
      }
    } catch (err) {
      statusPulse.classList.remove('online');
      statusText.textContent = 'Engine Offline';
    }
  }

  async function loadSources() {
    if (DEMO_MODE) {
      const demoSources = [
        { domain: 'pib.gov.in', category: 'Government Information Desk (Tier 1)', credibility_score: 0.99 },
        { domain: 'who.int', category: 'Global Health Authority (Tier 1)', credibility_score: 0.99 },
        { domain: 'nasa.gov', category: 'Aerospace & Scientific Agency (Tier 1)', credibility_score: 0.99 },
        { domain: 'rbi.org.in', category: 'Central Banking Authority (Tier 1)', credibility_score: 0.99 },
        { domain: 'reuters.com', category: 'Certified Fact-Checking Desk (Tier 2)', credibility_score: 0.95 },
        { domain: 'snopes.com', category: 'IFCN Certified Fact-Checker (Tier 2)', credibility_score: 0.95 },
        { domain: 'altnews.in', category: 'IFCN Fact-Checking Desk (Tier 2)', credibility_score: 0.95 },
        { domain: 'ptinews.com', category: 'Primary News Wire Service (Tier 3)', credibility_score: 0.92 },
        { domain: 'aninews.in', category: 'Press Wire Service (Tier 3)', credibility_score: 0.92 },
        { domain: 'thehindu.com', category: 'Mainstream National Press (Tier 4)', credibility_score: 0.90 },
        { domain: 'nature.com', category: 'Peer-Reviewed Scientific Journal (Tier 1)', credibility_score: 0.99 },
        { domain: 'britannica.com', category: 'Encyclopedic Knowledge Base (Tier 5)', credibility_score: 0.85 }
      ];
      sourcesTableBody.innerHTML = demoSources.map(s => `
        <tr>
          <td><strong>${escapeHtml(s.domain)}</strong></td>
          <td>${escapeHtml(s.category)}</td>
          <td>
            <div style="display:flex; align-items:center; gap:0.5rem;">
              <div style="flex:1; background:rgba(255,255,255,0.08); height:6px; border-radius:3px; overflow:hidden;">
                <div style="width:${Math.round(s.credibility_score * 100)}%; height:100%; background:#10B981;"></div>
              </div>
              <span>${Math.round(s.credibility_score * 100)}%</span>
            </div>
          </td>
          <td><span class="badge-tag">Verified Tier</span></td>
        </tr>
      `).join('');
      return;
    }
    try {
      const res = await fetch('/api/v1/sources');
      if (res.ok) {
        const sources = await res.json();
        sourcesTableBody.innerHTML = sources.map(s => {
          const domainName = s.domain || s.source || 'Verified Source';
          const categoryName = s.category || s.description || 'Verified News';
          const score = typeof s.credibility_score === 'number' ? s.credibility_score : 0.8;
          return `
          <tr>
            <td><strong>${escapeHtml(domainName)}</strong></td>
            <td>${escapeHtml(categoryName)}</td>
            <td>
              <div style="display:flex; align-items:center; gap:0.5rem;">
                <div style="flex:1; background:rgba(255,255,255,0.08); height:6px; border-radius:3px; overflow:hidden;">
                  <div style="width:${Math.round(score * 100)}%; height:100%; background:#10B981;"></div>
                </div>
                <span>${Math.round(score * 100)}%</span>
              </div>
            </td>
            <td><span class="badge-tag">Verified Tier</span></td>
          </tr>
        `;
        }).join('');
      }
    } catch (err) {
      console.warn('Unable to load sources database:', err);
    }
  }

  // Periodic health check every 30 seconds
  checkHealth();
  loadSources();
  setInterval(checkHealth, 30000);

  // --------------------------------------------------------------------------
  // Tabs Switching
  // --------------------------------------------------------------------------
  tabBtnText.addEventListener('click', () => {
    tabBtnText.classList.add('active');
    tabBtnText.setAttribute('aria-selected', 'true');
    tabBtnImage.classList.remove('active');
    tabBtnImage.setAttribute('aria-selected', 'false');

    panelText.removeAttribute('hidden');
    panelImage.setAttribute('hidden', 'true');
  });

  tabBtnImage.addEventListener('click', () => {
    tabBtnImage.classList.add('active');
    tabBtnImage.setAttribute('aria-selected', 'true');
    tabBtnText.classList.remove('active');
    tabBtnText.setAttribute('aria-selected', 'false');

    panelImage.removeAttribute('hidden');
    panelText.setAttribute('hidden', 'true');
  });

  // --------------------------------------------------------------------------
  // Text Tab Form Interactions
  // --------------------------------------------------------------------------
  claimInput.addEventListener('input', () => {
    const len = claimInput.value.length;
    claimCharCount.textContent = `${len} / 2000`;
    btnClearText.style.display = len > 0 ? 'flex' : 'none';
  });

  btnClearText.addEventListener('click', () => {
    claimInput.value = '';
    claimCharCount.textContent = '0 / 2000';
    btnClearText.style.display = 'none';
    claimInput.focus();
  });

  maxSourcesInput.addEventListener('input', (e) => {
    maxSourcesVal.textContent = e.target.value;
  });

  maxSourcesImageInput.addEventListener('input', (e) => {
    maxSourcesImageVal.textContent = e.target.value;
  });

  // Example Chips
  exampleChips.addEventListener('click', (e) => {
    const chip = e.target.closest('.example-chip');
    if (!chip) return;
    const claim = chip.getAttribute('data-claim');
    claimInput.value = claim;
    claimCharCount.textContent = `${claim.length} / 2000`;
    btnClearText.style.display = 'flex';
    claimInput.focus();
    showToast(`Loaded example: "${chip.textContent.trim()}"`, 'info');
  });

  // --------------------------------------------------------------------------
  // Image Tab Dropzone & Upload
  // --------------------------------------------------------------------------
  ['dragenter', 'dragover'].forEach(eventName => {
    imageDropzone.addEventListener(eventName, (e) => {
      e.preventDefault();
      e.stopPropagation();
      imageDropzone.classList.add('dragover');
    });
  });

  ['dragleave', 'drop'].forEach(eventName => {
    imageDropzone.addEventListener(eventName, (e) => {
      e.preventDefault();
      e.stopPropagation();
      imageDropzone.classList.remove('dragover');
    });
  });

  imageDropzone.addEventListener('drop', (e) => {
    const files = e.dataTransfer.files;
    if (files && files.length > 0) {
      handleImageSelection(files[0]);
    }
  });

  imageDropzone.addEventListener('click', (e) => {
    if (e.target === btnRemoveImage || btnRemoveImage.contains(e.target)) return;
    imageFileInput.click();
  });

  imageFileInput.addEventListener('change', (e) => {
    if (e.target.files && e.target.files.length > 0) {
      handleImageSelection(e.target.files[0]);
    }
  });

  btnRemoveImage.addEventListener('click', (e) => {
    e.stopPropagation();
    selectedFile = null;
    imageFileInput.value = '';
    imagePreviewWrapper.setAttribute('hidden', 'true');
    dropzonePrompt.removeAttribute('hidden');
  });

  function handleImageSelection(file) {
    if (!file.type.startsWith('image/')) {
      showToast('Please select a valid image file (JPEG, PNG, WebP)', 'error');
      return;
    }
    if (file.size > 10 * 1024 * 1024) {
      showToast('Image file size exceeds 10MB limit', 'error');
      return;
    }

    selectedFile = file;
    previewFilename.textContent = file.name;
    previewFilesize.textContent = formatBytes(file.size);

    const reader = new FileReader();
    reader.onload = (e) => {
      imagePreview.src = e.target.result;
      dropzonePrompt.setAttribute('hidden', 'true');
      imagePreviewWrapper.removeAttribute('hidden');
    };
    reader.readAsDataURL(file);
  }

  // --------------------------------------------------------------------------
  // Verification Execution Logic
  // --------------------------------------------------------------------------
  btnVerifyText.addEventListener('click', () => {
    const claim = claimInput.value.trim();
    if (!claim) {
      showToast('Please enter a claim or headline to verify', 'error');
      claimInput.focus();
      return;
    }

    const maxSources = parseInt(maxSourcesInput.value, 10) || 5;
    const bypassCache = bypassCacheCheckbox.checked;

    runTextVerification({
      text: claim,
      max_evidence: maxSources,
      bypass_cache: bypassCache
    });
  });

  btnVerifyImage.addEventListener('click', () => {
    if (!selectedFile) {
      showToast('Please upload an image for forensic verification', 'error');
      return;
    }

    const caption = imageClaimInput.value.trim();
    const maxSources = parseInt(maxSourcesImageInput.value, 10) || 5;

    runImageVerification(selectedFile, caption, maxSources);
  });

  function startStepper() {
    progressStepper.removeAttribute('hidden');
    resultsSection.setAttribute('hidden', 'true');

    // Reset steps
    [step1, step2, step3, step4].forEach(s => {
      s.className = 'step-item';
    });
    step1.classList.add('active');

    stepperStartTime = performance.now();
    stepperTimer.textContent = '0.0s';

    clearInterval(stepperInterval);
    stepperInterval = setInterval(() => {
      const elapsed = ((performance.now() - stepperStartTime) / 1000).toFixed(1);
      stepperTimer.textContent = `${elapsed}s`;

      const num = parseFloat(elapsed);
      if (num > 0.4 && !step1.classList.contains('completed')) {
        step1.classList.remove('active');
        step1.classList.add('completed');
        step2.classList.add('active');
        stepperStatusMsg.textContent = 'Retrieving authoritative evidence & fact-checks...';
      }
      if (num > 1.8 && !step2.classList.contains('completed')) {
        step2.classList.remove('active');
        step2.classList.add('completed');
        step3.classList.add('active');
        stepperStatusMsg.textContent = 'Evaluating DeBERTa-v3 Natural Language Inference...';
      }
      if (num > 3.2 && !step3.classList.contains('completed')) {
        step3.classList.remove('active');
        step3.classList.add('completed');
        step4.classList.add('active');
        stepperStatusMsg.textContent = 'Synthesizing verdict & multimodal confidence fusion...';
      }
    }, 100);
  }

  function stopStepper() {
    clearInterval(stepperInterval);
    [step1, step2, step3, step4].forEach(s => {
      s.className = 'step-item completed';
    });
    progressStepper.setAttribute('hidden', 'true');
  }

  async function runTextVerification(payload) {
    btnVerifyText.classList.add('loading');
    btnVerifyText.disabled = true;
    startStepper();

    try {
      if (DEMO_MODE) {
        await new Promise(r => setTimeout(r, 3800));
        const data = getDemoResponse(payload.text);
        currentResult = data;
        renderVerdict(data, false);
      } else {
        const res = await fetch('/api/v1/verify', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload)
        });
        if (!res.ok) {
          const errorData = await res.json().catch(() => ({}));
          throw new Error(errorData.detail || `Server error: ${res.status}`);
        }
        const data = await res.json();
        currentResult = data;
        renderVerdict(data, false);
        checkHealth();
      }
    } catch (err) {
      showToast(`Verification failed: ${err.message}`, 'error');
    } finally {
      stopStepper();
      btnVerifyText.classList.remove('loading');
      btnVerifyText.disabled = false;
    }
  }

  async function runImageVerification(file, caption, maxSources) {
    btnVerifyImage.classList.add('loading');
    btnVerifyImage.disabled = true;
    startStepper();

    const formData = new FormData();
    formData.append('image', file);
    formData.append('file', file);
    if (caption) {
      formData.append('text', caption);
      formData.append('claim_text', caption);
    }
    formData.append('max_evidence', maxSources.toString());

    try {
      if (DEMO_MODE) {
        await new Promise(r => setTimeout(r, 3800));
        const captionText = caption || 'Uploaded news screenshot / media claim';
        const data = {
          verdict: 'Contradicted',
          confidence: 0.89,
          cached: false,
          execution_time: 3.82,
          claim: captionText,
          category: 'multimodal_forensics',
          claimant: 'Social Media Circulation',
          detected_authorities: [
            { name: 'PIB Fact Check', role: 'Official Authority', official_handles: ['@PIBFactCheck'] }
          ],
          primary_origins: [
            { name: 'PIB Fact Check Desk', role: 'National Fact Desk', official_handles: ['factcheck.pib.gov.in'] }
          ],
          authenticity_score: 0.42,
          context_score: 0.38,
          image_verdict: 'manipulated',
          forensics: {
            context: { verdict: 'contextually_inconsistent' }
          },
          extracted_ocr: {
            has_text: true,
            detected_template: 'Social Media Poster',
            confidence: 0.92
          },
          truth_analysis: {
            status: 'rumor',
            badge_label: 'DEBUNKED RUMOR / MANIPULATED MEDIA',
            badge_color: 'rose',
            claim_statement: captionText,
            actual_truth: 'Forensic Error Level Analysis (ELA) and FFT frequency spectral decomposition indicate localized image compression tampering. The image has been digitally altered and recirculated out of its original historical context.',
            rumor_statement: captionText,
            evidence_basis: 'Spectral FFT manipulation markers detected alongside IFCN media archives confirming altered visual elements.',
            verdict_meaning: 'Visual forensic analysis confirms manipulation or out-of-context misattribution.'
          },
          probabilities: { entailment: 0.05, contradiction: 0.89, neutral: 0.06 },
          entities: { MEDIA: ['Digital Poster'], FORENSIC: ['ELA Block Discrepancy', 'FFT Anomaly'] },
          evidence: [
            {
              text: 'Error Level Analysis reveals severe compression gradient divergence along text boundaries, consistent with splicing.',
              url: 'https://boomlive.in/fact-check',
              stance: 'contradiction',
              stance_score: 0.94,
              credibility_score: 0.95,
              source_tier: 'fact_checker'
            },
            {
              text: 'Original unaltered image was captured during a 2019 demonstration and does not depict current events.',
              url: 'https://altnews.in',
              stance: 'contradiction',
              stance_score: 0.88,
              credibility_score: 0.95,
              source_tier: 'fact_checker'
            }
          ]
        };
        currentResult = data;
        renderVerdict(data, true);
      } else {
        const res = await fetch('/api/v1/verify/image', {
          method: 'POST',
          body: formData
        });

        if (!res.ok) {
          const errorData = await res.json().catch(() => ({}));
          throw new Error(errorData.detail || `Server error: ${res.status}`);
        }

        const data = await res.json();
        currentResult = data;
        renderVerdict(data, true);
        checkHealth();
      }
    } catch (err) {
      showToast(`Forensic verification failed: ${err.message}`, 'error');
    } finally {
      stopStepper();
      btnVerifyImage.classList.remove('loading');
      btnVerifyImage.disabled = false;
    }
  }

  // --------------------------------------------------------------------------
  // Results & Verdict Rendering
  // --------------------------------------------------------------------------
  function renderVerdict(data, isImage) {
    resultsSection.removeAttribute('hidden');

    const verdict = (data.verdict || data.final_verdict || 'Unverified').toLowerCase();
    const confVal = data.confidence !== undefined ? data.confidence : (data.final_confidence !== undefined ? data.final_confidence : 0);
    const confidencePct = Math.round(confVal * 100);

    // Reset card classes
    verdictCard.className = 'card verdict-card';

    // State styling
    if (verdict === 'supported') {
      verdictCard.classList.add('state-supported');
      verdictText.textContent = 'SUPPORTED / TRUE';
    } else if (verdict === 'contradicted') {
      verdictCard.classList.add('state-contradicted');
      verdictText.textContent = 'CONTRADICTED / FALSE';
    } else {
      verdictCard.classList.add('state-unverified');
      verdictText.textContent = 'UNVERIFIED / AMBIGUOUS';
    }

    // Radial Gauge Animation
    animateGauge(confidencePct);

    // Cache status
    if (data.cached) {
      verdictCacheBadge.removeAttribute('hidden');
      cacheLatency.textContent = `${(data.execution_time || 0).toFixed(3)}s`;
    } else {
      verdictCacheBadge.setAttribute('hidden', 'true');
    }

    // Evaluated Statement
    const claimStatement = data.claim || (data.extracted_claims && data.extracted_claims[0]) || 'Submitted news statement';
    resultClaimText.textContent = `"${claimStatement}"`;

    // OCR & Authority Badges
    const ocrWrap = document.getElementById('ocr-badge-wrap');
    const authWrap = document.getElementById('authorities-badge-wrap');

    if (ocrWrap) {
      if (data.extracted_ocr && data.extracted_ocr.has_text) {
        const tmpl = data.extracted_ocr.detected_template ? ` • ${escapeHtml(data.extracted_ocr.detected_template)} Card` : '';
        ocrWrap.innerHTML = `
          <span style="background:rgba(124,155,255,0.16); border:1px solid #7C9BFF; color:#E0E7FF; font-size:0.75rem; font-weight:600; padding:3px 8px; border-radius:6px; display:inline-flex; align-items:center; gap:5px;">
            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M4 7V4h16v3M9 20h6M12 4v16"/></svg>
            Multimodal OCR Extracted (${Math.round((data.extracted_ocr.confidence || 0.8) * 100)}% Conf${tmpl})
          </span>
        `;
        ocrWrap.style.display = 'flex';
      } else {
        ocrWrap.style.display = 'none';
      }
    }

    if (authWrap) {
      const tags = [];
      if (data.category && data.category !== 'general') {
        const catLabel = data.category.replace(/_/g, ' ').toUpperCase();
        tags.push(`
          <span style="background:rgba(14,165,233,0.15); border:1px solid #0ea5e9; color:#38bdf8; font-size:0.75rem; font-weight:600; padding:3px 8px; border-radius:6px; display:inline-flex; align-items:center; gap:5px;">
            🌐 Domain: ${escapeHtml(catLabel)}
          </span>
        `);
      }
      if (data.claimant) {
        tags.push(`
          <span style="background:rgba(245,158,11,0.15); border:1px solid #f59e0b; color:#fbbf24; font-size:0.75rem; font-weight:600; padding:3px 8px; border-radius:6px; display:inline-flex; align-items:center; gap:5px;">
            👤 Origin Body: ${escapeHtml(data.claimant)}
          </span>
        `);
      }
      if (data.detected_authorities && data.detected_authorities.length > 0) {
        data.detected_authorities.forEach(a => {
          tags.push(`
            <span style="background:rgba(16,185,129,0.15); border:1px solid #10b981; color:#34d399; font-size:0.75rem; font-weight:600; padding:3px 8px; border-radius:6px; display:inline-flex; align-items:center; gap:5px;">
              🏛️ ${escapeHtml(a.name)}
            </span>
          `);
        });
      }

      if (tags.length > 0) {
        authWrap.innerHTML = tags.join('');
        authWrap.style.display = 'flex';
      } else {
        authWrap.style.display = 'none';
      }
    }

    // Primary Official Sources Investigated Card
    const originsWrap = document.getElementById('primary-origins-wrap');
    const originsContent = document.getElementById('primary-origins-content');
    const primaryOrigins = data.primary_origins || data.detected_authorities || [];

    if (originsWrap && originsContent) {
      if (primaryOrigins.length > 0) {
        originsContent.innerHTML = primaryOrigins.map(po => {
          const handles = (po.official_handles || []).join(', ');
          const role = po.role || 'Primary Authority';
          return `
            <div style="background:rgba(255,255,255,0.04); border:1px solid rgba(255,255,255,0.1); border-radius:6px; padding:6px 10px; font-size:0.75rem; flex:1 1 220px;">
              <div style="display:flex; align-items:center; justify-content:space-between; margin-bottom:2px;">
                <span style="font-weight:700; color:#F8FAFC;">🏛️ ${escapeHtml(po.name)}</span>
                <span style="color:#93B4FF; font-size:0.68rem; font-weight:700; text-transform:uppercase;">${escapeHtml(role)}</span>
              </div>
              ${handles ? `<div style="color:#94A3B8; font-size:0.7rem;">Channels: <span style="color:#CBD5E1;">${escapeHtml(handles)}</span></div>` : ''}
            </div>
          `;
        }).join('');
        originsWrap.style.display = 'block';
      } else {
        originsWrap.style.display = 'none';
      }
    }

    // Truth Audit & Reality Breakdown Section (Rumor vs. Ground Truth)
    renderTruthSection(data);

    // Automated Synthesis / Explanation
    verdictExplanation.textContent = generateSynthesis(data);

    // Metadata Chips
    metaExecutionTime.textContent = `${(data.execution_time || 0).toFixed(2)}s`;
    const evidenceItems = data.evidence || data.top_evidence || [];
    metaEvidenceCount.textContent = `${evidenceItems.length} sources`;

    const probs = data.probabilities || {};
    const ent = probs.entailment || 0;
    const con = probs.contradiction || 0;
    const neu = probs.neutral || 0;

    let dominant = 'Neutral';
    if (ent > con && ent > neu) dominant = 'Entailment (Supports)';
    else if (con > ent && con > neu) dominant = 'Contradiction (Refutes)';
    metaDominantStance.textContent = dominant;

    // NLI Stance Bars
    const entPct = Math.round(ent * 100);
    const conPct = Math.round(con * 100);
    const neuPct = Math.round(neu * 100);

    valEntailment.textContent = `${entPct}%`;
    barEntailment.style.width = `${entPct}%`;

    valContradiction.textContent = `${conPct}%`;
    barContradiction.style.width = `${conPct}%`;

    valNeutral.textContent = `${neuPct}%`;
    barNeutral.style.width = `${neuPct}%`;

    // Named Entities
    renderEntities(data.entities || {});

    // Forensics Card (if multimodal)
    const hasForensics = isImage || (
      data.authenticity_score !== null &&
      data.authenticity_score !== undefined &&
      data.image_verdict !== null &&
      data.image_verdict !== undefined
    );

    if (hasForensics) {
      cardForensics.removeAttribute('hidden');
      cardForensics.style.display = 'block';
      const authPct = Math.round((data.authenticity_score || 0.85) * 100);
      valAuthenticity.textContent = `${authPct}%`;
      barAuthenticity.style.width = `${authPct}%`;

      if (data.context_score !== null && data.context_score !== undefined) {
        const clipPct = Math.round(data.context_score * 100);
        valClip.textContent = `${clipPct}%`;
        barClip.style.width = `${clipPct}%`;
        const ctxVerdict = data.forensics && data.forensics.context ? data.forensics.context.verdict : null;
        const isConsistent = ctxVerdict ? ctxVerdict === 'contextually_consistent' : clipPct >= 58;
        clipStatusSub.textContent = isConsistent
          ? 'Image semantics align with caption context.'
          : 'Low semantic alignment: Image may be reused out of context.';
      } else {
        valClip.textContent = 'N/A';
        barClip.style.width = '0%';
        clipStatusSub.textContent = 'No context text provided for alignment.';
      }
    } else {
      cardForensics.setAttribute('hidden', 'true');
      cardForensics.style.display = 'none';
    }

    // Evidence Explorer Cards
    renderEvidenceList(evidenceItems, 'all');

    // Scroll smoothly to results
    verdictCard.scrollIntoView({ behavior: 'smooth', block: 'start' });
  }

  function animateGauge(targetPct) {
    const circumference = 314.159; // 2 * pi * 50
    const offset = circumference - (targetPct / 100) * circumference;
    gaugeFill.style.strokeDashoffset = offset;

    // Number ticker
    let current = 0;
    const step = Math.ceil(targetPct / 25) || 1;
    const timer = setInterval(() => {
      current += step;
      if (current >= targetPct) {
        current = targetPct;
        clearInterval(timer);
      }
      gaugeValue.textContent = `${current}%`;
    }, 20);
  }

  function generateSynthesis(data) {
    if (data.reasoning && data.reasoning.trim()) {
      return data.reasoning;
    }
    const verdict = data.verdict;
    const conf = Math.round((data.confidence || 0) * 100);
    const count = (data.evidence || []).length;

    if (verdict === 'Supported') {
      return `Our cross-encoder NLI and factual retrieval corroborates this statement with high statistical confidence (${conf}%). Authoritative records and credible reporting confirm the core claims.`;
    } else if (verdict === 'Contradicted') {
      return `This statement has been actively refuted and debunked by credible sources and fact-checkers (${conf}% contradiction confidence). Cross-referenced reporting indicates the claim is factually false or misleading.`;
    } else {
      return `Insufficient conclusive evidence was found to decisively support or refute this claim (${count} sources reviewed). The available reporting is either neutral, unverified, or conflicting. Exercise skepticism.`;
    }
  }

  function renderTruthSection(data) {
    if (!truthSectionBox) return;

    const verdict = (data.verdict || data.final_verdict || 'Unverified').toLowerCase();
    const claimStatement = data.claim || (data.extracted_claims && data.extracted_claims[0]) || 'Submitted news statement';
    const evidenceList = data.evidence || data.top_evidence || [];
    const primaryOrigins = data.primary_origins || data.detected_authorities || [];
    const primaryNames = primaryOrigins.map(p => p.name).filter(Boolean);
    const primaryStr = primaryNames.slice(0, 2).join(', ') || 'official authorities';

    // Retrieve or synthesize truth_analysis
    let analysis = data.truth_analysis;

    if (!analysis) {
      // Client-side dynamic fallback synthesis
      const refuting = evidenceList.filter(e => (e.stance || e.relationship || '').toLowerCase() === 'contradiction');
      const supporting = evidenceList.filter(e => (e.stance || e.relationship || '').toLowerCase() === 'entailment');

      const isSpeech = /\b(speech|spoke|speaks|speaking|said|stated|press\s*meet|press\s*conference|interview|rally|addressed|parliamentary|parliament|assembly|bhashan|பேச்சு|பேசியது|செய்தியாளர்\s*சந்திப்பு|பேட்டி)\b/i.test(claimStatement);

      if (verdict === 'contradicted') {
        const topRefute = refuting.length > 0 ? refuting[0].text : '';
        let refuteSnippet;
        if (topRefute) {
          refuteSnippet = `Authoritative reports clarify: "${topRefute.substring(0, 260)}..."`;
        } else if (isSpeech) {
          refuteSnippet = `Accredited news channels, live broadcast archives, and verified newspaper reports confirm that ${primaryStr} delivered no such speech or statement. In real-world events, public addresses and press meets are documented live by news channels and accredited press; the circulating claim is a fabricated assertion not corroborated by any credible news media broadcast or video evidence.`;
        } else {
          refuteSnippet = `Official circulars from ${primaryStr} confirm no such policy or order exists.`;
        }

        analysis = {
          status: 'rumor',
          badge_label: 'DEBUNKED RUMOR / FALSE',
          badge_color: 'rose',
          claim_statement: claimStatement,
          actual_truth: isSpeech && !topRefute ? refuteSnippet : `Official records and factual evidence confirm this statement is a circulating rumor. ${refuteSnippet}`,
          rumor_statement: claimStatement,
          evidence_basis: isSpeech ?
            `Cross-referenced against accredited TV news channels, live broadcast coverage, news wires (ANI, PTI), and major newspapers. No authentic video proof exists.` :
            `Refuted by ${primaryStr} records and accredited media fact-checks. No authentic notification exists.`,
          verdict_meaning: isSpeech ?
            'This circulating speech claim is false and fabricated; no verified news channel or newspaper has recorded such an event.' :
            'This statement is a rumor and NOT actual truth based on official data.'
        };
      } else if (verdict === 'supported') {
        const topSupport = supporting.length > 0 ? supporting[0].text : '';
        const supportSnippet = topSupport ?
          `Authoritative records confirm: "${topSupport.substring(0, 260)}..."` :
          (isSpeech ?
            `Verified by live broadcast coverage, accredited news channels, and mainstream newspaper reports documenting the speech delivered by ${primaryStr}.` :
            `Corroborated by verified press releases and official statements from ${primaryStr}.`);

        analysis = {
          status: 'verified_truth',
          badge_label: 'VERIFIED ACTUAL TRUTH',
          badge_color: 'emerald',
          claim_statement: claimStatement,
          actual_truth: isSpeech && !topSupport ? supportSnippet : `This statement represents verified factual truth. ${supportSnippet}`,
          rumor_statement: isSpeech ?
            'No conflicting rumor detected; speech/statement is verified by media broadcast records.' :
            'No conflicting rumor detected; statement is fully authenticated.',
          evidence_basis: isSpeech ?
            'Corroborated by live video proofs, television news broadcasts, and accredited press reporting.' :
            `Directly corroborated by primary records from ${primaryStr} and independent news wires.`,
          verdict_meaning: isSpeech ?
            'This speech/statement is verified as genuine based on accredited news broadcasts and published reports.' :
            'This statement is verified as genuine truth based on official evidence.'
        };
      } else {
        analysis = {
          status: 'unverified_rumor',
          badge_label: 'UNCONFIRMED / POTENTIAL RUMOR',
          badge_color: 'amber',
          claim_statement: claimStatement,
          actual_truth: isSpeech ?
            `No accredited news channel, live broadcast recording, or verified newspaper report has documented this speech or press meet. While this assertion is actively circulating on social media, accredited press and media archives contain no video proof or confirmation from ${primaryStr}.` :
            `No official notification, government circular, or verified press wire exists to corroborate this claim. While this assertion is circulating on social media, ${primaryStr} have not issued or confirmed it.`,
          rumor_statement: claimStatement,
          evidence_basis: isSpeech ?
            `Lacks video proof, live broadcast coverage, or newspaper confirmation from accredited media.` :
            `Lacks authentic evidentiary backing or documentation from ${primaryStr}.`,
          verdict_meaning: isSpeech ?
            'This speech claim lacks video proof or news media verification and is currently uncorroborated.' :
            'This claim lacks verified truth data and is circulating without official authentication.'
        };
      }
    }

    // Apply styles to truthSectionBox based on status
    truthSectionBox.className = 'truth-section-box';
    if (analysis.status === 'rumor') {
      truthSectionBox.classList.add('state-rumor');
      if (truthStatusBadge) {
        truthStatusBadge.className = 'truth-status-badge badge-rumor';
      }
      if (truthColRumorTag) truthColRumorTag.innerHTML = '⚠️ Circulating Rumor';
      if (truthRumorBanner) {
        truthRumorBanner.className = 'truth-advisory-banner banner-rumor';
        if (truthRumorBannerText) {
          truthRumorBannerText.textContent = 'This is a rumor and NOT actual truth based on official data.';
        }
      }
      if (truthColFactTag) truthColFactTag.innerHTML = '🛡️ Verified Ground Truth';
      if (truthFactBanner) {
        truthFactBanner.className = 'truth-advisory-banner banner-fact';
        if (truthFactBannerText) {
          truthFactBannerText.textContent = 'Actual factual reality established by official authorities and evidence.';
        }
      }
    } else if (analysis.status === 'verified_truth') {
      truthSectionBox.classList.add('state-fact');
      if (truthStatusBadge) {
        truthStatusBadge.className = 'truth-status-badge badge-fact';
      }
      if (truthColRumorTag) truthColRumorTag.innerHTML = '📢 Evaluated Claim';
      if (truthRumorBanner) {
        truthRumorBanner.className = 'truth-advisory-banner banner-fact';
        if (truthRumorBannerText) {
          truthRumorBannerText.textContent = 'Statement matches verified factual records.';
        }
      }
      if (truthColFactTag) truthColFactTag.innerHTML = '✅ Corroborated Ground Truth';
      if (truthFactBanner) {
        truthFactBanner.className = 'truth-advisory-banner banner-fact';
        if (truthFactBannerText) {
          truthFactBannerText.textContent = 'Corroborated by primary origins and reliable sources.';
        }
      }
    } else {
      truthSectionBox.classList.add('state-unverified');
      if (truthStatusBadge) {
        truthStatusBadge.className = 'truth-status-badge badge-unverified';
      }
      if (truthColRumorTag) truthColRumorTag.innerHTML = '❓ Unverified Claim';
      if (truthRumorBanner) {
        truthRumorBanner.className = 'truth-advisory-banner banner-unverified';
        if (truthRumorBannerText) {
          truthRumorBannerText.textContent = 'This assertion lacks verified truth data and is unconfirmed.';
        }
      }
      if (truthColFactTag) truthColFactTag.innerHTML = '⚖️ Established Reality';
      if (truthFactBanner) {
        truthFactBanner.className = 'truth-advisory-banner banner-unverified';
        if (truthFactBannerText) {
          truthFactBannerText.textContent = 'No authentic government circular or evidence corroborates this.';
        }
      }
    }

    if (truthBadgeText) {
      truthBadgeText.textContent = analysis.badge_label || (analysis.status === 'rumor' ? 'CIRCULATING RUMOR — NOT ACTUAL TRUTH' : 'TRUTH AUDIT');
    }

    if (truthVerdictMeaning) {
      truthVerdictMeaning.textContent = analysis.verdict_meaning || 'Evaluation of claim vs. factual ground truth';
    }

    if (truthRumorText) {
      truthRumorText.textContent = `"${analysis.rumor_statement || claimStatement}"`;
    }

    if (truthFactText) {
      truthFactText.textContent = analysis.actual_truth || 'No ground truth counter-evidence recorded.';
    }

    if (truthFoundationText) {
      truthFoundationText.textContent = analysis.evidence_basis || `Audited across ${evidenceList.length} evidence sources and official channels.`;
    }

    truthSectionBox.style.display = 'block';
  }

  function renderEntities(entities) {
    entityPillsWrap.innerHTML = '';
    const entries = Object.entries(entities);

    let hasEntities = false;
    for (const [type, items] of entries) {
      if (!Array.isArray(items) || items.length === 0) continue;
      hasEntities = true;

      const typeClass = `type-${type.toLowerCase()}`;
      items.forEach(item => {
        const pill = document.createElement('div');
        pill.className = `entity-pill ${typeClass}`;
        pill.innerHTML = `
          <span>${escapeHtml(item)}</span>
          <span class="entity-type">${escapeHtml(type)}</span>
        `;
        entityPillsWrap.appendChild(pill);
      });
    }

    if (hasEntities) {
      noEntitiesMsg.style.display = 'none';
      entityPillsWrap.style.display = 'flex';
    } else {
      noEntitiesMsg.style.display = 'block';
      entityPillsWrap.style.display = 'none';
    }
  }

  function renderEvidenceList(evidenceItems, filter) {
    evidenceCountBadge.textContent = `${evidenceItems.length} Sources`;

    let filtered = evidenceItems;
    if (filter === 'entailment') {
      filtered = evidenceItems.filter(e => (e.stance || '').toLowerCase() === 'entailment');
    } else if (filter === 'contradiction') {
      filtered = evidenceItems.filter(e => (e.stance || '').toLowerCase() === 'contradiction');
    } else if (filter === 'factcheck') {
      filtered = evidenceItems.filter(e => {
        const src = (e.source || '').toLowerCase();
        const url = (e.url || '').toLowerCase();
        return src.includes('fact') || url.includes('politifact') || url.includes('factcheck') || url.includes('snopes');
      });
    }

    if (filtered.length === 0) {
      evidenceList.innerHTML = `
        <div style="padding:2rem; text-align:center; color:var(--text-muted); font-size:0.9rem;">
          No evidence sources match the selected filter.
        </div>
      `;
      return;
    }

    evidenceList.innerHTML = filtered.map((item, idx) => {
      const stance = (item.stance || 'neutral').toLowerCase();
      const stanceLabel = stance === 'entailment' ? 'Supports' : stance === 'contradiction' ? 'Refutes' : 'Neutral';
      const stanceClass = `stance-${stance}`;
      const credPct = Math.round((item.credibility_score ?? item.authority_score ?? 0.5) * 100);
      const domain = extractDomain(item.url || item.source || 'Fact-Check Network');
      const isOfficial = Boolean(item.is_official_primary || (item.authority_score && item.authority_score >= 0.98));
      const tier = item.source_tier || (isOfficial ? 'primary_origin' : credPct >= 94 ? 'fact_checker' : credPct >= 90 ? 'news_wire' : credPct <= 50 ? 'unverified_social' : 'general_web');

      let tierBadgeHtml = '';
      if (isOfficial || tier === 'primary_origin') {
        tierBadgeHtml = `
          <span class="primary-authority-badge" style="background:rgba(147,51,234,0.22); border:1px solid #a855f7; color:#c084fc; font-weight:700; padding:2px 8px; border-radius:9999px; font-size:0.72rem; display:inline-flex; align-items:center; gap:4px;">
            🏛️ ORIGINAL PRIMARY SOURCE
          </span>
        `;
      } else if (tier === 'fact_checker') {
        tierBadgeHtml = `
          <span style="background:rgba(16,185,129,0.18); border:1px solid #10b981; color:#34d399; font-weight:700; padding:2px 8px; border-radius:9999px; font-size:0.72rem; display:inline-flex; align-items:center; gap:4px;">
            🛡️ CERTIFIED FACT-CHECKER
          </span>
        `;
      } else if (tier === 'news_wire') {
        tierBadgeHtml = `
          <span style="background:rgba(14,165,233,0.18); border:1px solid #0ea5e9; color:#38bdf8; font-weight:700; padding:2px 8px; border-radius:9999px; font-size:0.72rem; display:inline-flex; align-items:center; gap:4px;">
            📰 ACCREDITED NEWS WIRE
          </span>
        `;
      } else if (tier === 'encyclopedic_record' || tier === 'encyclopedia') {
        tierBadgeHtml = `
          <span style="background:rgba(124,155,255,0.16); border:1px solid #7C9BFF; color:#E0E7FF; font-weight:700; padding:2px 8px; border-radius:9999px; font-size:0.72rem; display:inline-flex; align-items:center; gap:4px;">
            📚 ENCYCLOPEDIC RECORD
          </span>
        `;
      } else if (tier === 'unverified_social') {
        tierBadgeHtml = `
          <span style="background:rgba(245,158,11,0.18); border:1px solid #f59e0b; color:#fbbf24; font-weight:600; padding:2px 8px; border-radius:9999px; font-size:0.72rem; display:inline-flex; align-items:center; gap:4px;">
            ⚠️ UNVERIFIED SOCIAL MEDIA
          </span>
        `;
      }

      return `
        <div class="evidence-item-card ${isOfficial ? 'evidence-primary-official' : ''}" style="${isOfficial ? 'border-color: rgba(168,85,247,0.5); box-shadow: 0 0 14px rgba(168,85,247,0.12);' : ''}">
          <div class="evidence-top">
            <div class="evidence-source-info">
              <span class="source-domain-badge">${escapeHtml(domain)}</span>
              ${tierBadgeHtml}
              <span class="credibility-pill" title="Source Credibility Weight">
                <svg width="12" height="12" viewBox="0 0 24 24" fill="currentColor">
                  <path d="M12 2l3.09 6.26L22 9.27l-5 4.87 1.18 6.88L12 17.77l-6.18 3.25L7 14.14 2 9.27l6.91-1.01L12 2z"/>
                </svg>
                ${credPct}% Credibility
              </span>
            </div>
            <span class="evidence-stance-pill ${stanceClass}">
              ${stanceLabel} (${Math.round((item.stance_score || 0.5) * 100)}%)
            </span>
          </div>

          <div class="evidence-snippet">
            &ldquo;${escapeHtml(item.text || 'No snippet text provided.')}&rdquo;
          </div>

          <div class="evidence-bottom">
            <span>Retrieved via Multi-Source Index</span>
            ${item.url ? `
              <a href="${escapeHtml(item.url)}" target="_blank" rel="noopener noreferrer" class="btn-source-link">
                <span>View Original Source</span>
                <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                  <path d="M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6"/>
                  <polyline points="15 3 21 3 21 9"/>
                  <line x1="10" y1="14" x2="21" y2="3"/>
                </svg>
              </a>
            ` : '<span>Internal Index</span>'}
          </div>
        </div>
      `;
    }).join('');
  }

  // Evidence Filter Buttons
  evidenceFilters.addEventListener('click', (e) => {
    const btn = e.target.closest('.filter-chip');
    if (!btn || !currentResult) return;

    evidenceFilters.querySelectorAll('.filter-chip').forEach(c => c.classList.remove('active'));
    btn.classList.add('active');

    const filter = btn.getAttribute('data-filter');
    renderEvidenceList(currentResult.evidence || [], filter);
  });

  // --------------------------------------------------------------------------
  // Toolbar Actions (Copy, Download, Reset)
  // --------------------------------------------------------------------------
  btnCopySummary.addEventListener('click', () => {
    if (!currentResult) return;

    const claim = currentResult.claim || (currentResult.extracted_claims && currentResult.extracted_claims[0]) || '';
    const verdict = currentResult.verdict || 'Unverified';
    const conf = Math.round((currentResult.confidence || 0) * 100);
    const explanation = verdictExplanation.textContent;

    let truthLines = '';
    if (currentResult.truth_analysis) {
      const ta = currentResult.truth_analysis;
      truthLines = `\nStatus: ${ta.badge_label || ta.status.toUpperCase()}\nMeaning: ${ta.verdict_meaning || ''}\nActual Truth: ${ta.actual_truth || ''}\nRumor Statement: ${ta.rumor_statement || ''}\nEvidentiary Basis: ${ta.evidence_basis || ''}\n`;
    }

    const summaryText = [
      `🛡️ CREDENCEX Verification Summary`,
      `Statement: "${claim}"`,
      `Verdict: ${verdict.toUpperCase()} (${conf}% Confidence)`,
      truthLines,
      `Details: ${explanation}`,
      `\nVerified via DeBERTa-v3 Cross-Encoder & Fact-Check DB.`
    ].filter(Boolean).join('\n');

    navigator.clipboard.writeText(summaryText).then(() => {
      copySummaryText.textContent = 'Copied to Clipboard!';
      showToast('Summary copied to clipboard', 'info');
      setTimeout(() => {
        copySummaryText.textContent = 'Copy Debunk Summary';
      }, 2500);
    }).catch(() => {
      showToast('Failed to copy to clipboard', 'error');
    });
  });

  btnDownloadJson.addEventListener('click', () => {
    if (!currentResult) return;

    const blob = new Blob([JSON.stringify(currentResult, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `credencex_verification_${Date.now()}.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
    showToast('Downloaded JSON report', 'info');
  });

  btnNewSearch.addEventListener('click', () => {
    window.scrollTo({ top: 0, behavior: 'smooth' });
    if (tabBtnText.classList.contains('active')) {
      claimInput.focus();
    }
  });

  // --------------------------------------------------------------------------
  // Sources Directory Modal
  // --------------------------------------------------------------------------
  if (navSourcesBtn && modalSources) {
    navSourcesBtn.addEventListener('click', () => {
      modalSources.removeAttribute('hidden');
    });
  }

  if (btnCloseSourcesModal && modalSources) {
    btnCloseSourcesModal.addEventListener('click', () => {
      modalSources.setAttribute('hidden', 'true');
    });
  }

  if (modalSources) {
    modalSources.addEventListener('click', (e) => {
      if (e.target === modalSources) {
        modalSources.setAttribute('hidden', 'true');
      }
    });
  }

  // --------------------------------------------------------------------------
  // Utilities
  // --------------------------------------------------------------------------
  function showToast(message, type = 'info') {
    const toast = document.createElement('div');
    toast.className = 'toast';
    toast.innerHTML = `
      <span style="color:${type === 'error' ? '#F43F5E' : '#7C9BFF'}">●</span>
      <span>${escapeHtml(message)}</span>
    `;
    toastContainer.appendChild(toast);

    setTimeout(() => {
      toast.style.opacity = '0';
      toast.style.transform = 'translateY(10px)';
      toast.style.transition = 'all 0.3s ease';
      setTimeout(() => toast.remove(), 300);
    }, 3500);
  }

  function escapeHtml(text) {
    if (!text) return '';
    return String(text)
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;')
      .replace(/'/g, '&#039;');
  }

  function extractDomain(url) {
    try {
      if (!url.startsWith('http://') && !url.startsWith('https://')) {
        return url;
      }
      const parsed = new URL(url);
      return parsed.hostname.replace('www.', '');
    } catch {
      return url;
    }
  }

  function formatBytes(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + ' ' + sizes[i];
  }
});
