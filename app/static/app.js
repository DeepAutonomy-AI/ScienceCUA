const appState = {
  files: [],
  urls: [],
  currentRunId: null,
  runPayload: null,
  events: [],
  eventOffset: 0,
  pollTimer: null,
  auditFilter: "all",
  evidenceView: "tree",
  evidenceTreeRunId: null,
  evidenceTreeMarkup: null,
  evidenceTreeOpenState: new Map(),
  skillsRunId: null,
  skillsMarkup: null,
  skillOpenState: new Map(),
  openReferenceCatalog: null,
  shownFailureRunIds: new Set(),
  language: "English",
  sourceMode: "manual",
  manualSourceKind: "file",
  manualSourceDrafts: { file: [], url: [] },
  sourceModeChosen: false,
  setupStage: "target",
  generationCoverage: "medium",
  generationCoverageRunId: null,
  shownCandidateRunIds: new Set(),
  candidatesRunId: null,
  candidateRequestPending: false,
  shownScopeOutlineIds: new Set(),
  scopeDrafts: new Map(),
  scopeRunId: null,
  scopeOutlineArtifactId: null,
  scopeRequestPending: false,
  serviceHealth: "Connecting",
  editedModelFields: new Set(),
  formIssues: new Map(),
};

const GENERATION_COVERAGE_LEVELS = Object.freeze([
  { value: "low", label: "Low", tasks: 8, references: 16 },
  { value: "medium", label: "Medium", tasks: 16, references: 24 },
  { value: "high", label: "High", tasks: 22, references: 28 },
  { value: "ultra", label: "Ultra", tasks: 28, references: 36 },
]);

const $ = (selector) => document.querySelector(selector);
const $$ = (selector) => Array.from(document.querySelectorAll(selector));

document.addEventListener("DOMContentLoaded", () => {
  initializeLanguage();
  bindForm();
  bindGenerationCoverage();
  loadModelDefaults();
  bindTabs();
  bindSkillMotion();
  bindFilters();
  bindReferenceCatalogDialog();
  bindPipelineErrorDialog();
  bindSourceDialogs();
  addUrlRow();
  checkHealth();
  const requestedRun = new URLSearchParams(window.location.search).get("run");
  refreshRuns().then(() => {
    if (requestedRun) selectRun(requestedRun);
  });
  initializeSetupFlow({ focus: !requestedRun });
  refreshIcons();
});

function t(message, values = {}) {
  const translated = appState.language === "Chinese" ? UI_ZH[message] || message : message;
  return translated.replace(/\{(\w+)\}/g, (_, key) => String(values[key] ?? `{${key}}`));
}

function initializeLanguage() {
  try {
    appState.language = localStorage.getItem("sciencecua.language") === "Chinese" ? "Chinese" : "English";
  } catch { /* A restricted browser can still switch language for this page. */ }
  $$("[data-language]").forEach((button) => button.addEventListener("click", () => {
    appState.language = button.dataset.language;
    try { localStorage.setItem("sciencecua.language", appState.language); } catch { /* Optional preference storage. */ }
    applyLanguage();
    renderInputSources();
    renderSourceMode();
    if (appState.runPayload) renderRun();
    if (appState.openReferenceCatalog) renderReferenceCatalogEntries(appState.openReferenceCatalog, $("#reference-catalog-filter").value);
    if ($("#source-candidates-dialog").open) renderCandidateChoices();
    if ($("#source-scope-dialog").open) renderScopeChoices();
    refreshRuns();
  }));
  applyLanguage();
}

function applyLanguage() {
  document.documentElement.lang = appState.language === "Chinese" ? "zh-CN" : "en";
  $$("[data-i18n]").forEach((node) => { node.textContent = t(node.dataset.i18n); });
  for (const [attribute, dataKey] of [["placeholder", "i18nPlaceholder"], ["title", "i18nTitle"], ["aria-label", "i18nAria"]]) {
    $$(`[data-${dataKey.replace(/[A-Z]/g, (letter) => `-${letter.toLowerCase()}`)}]`).forEach((node) => node.setAttribute(attribute, t(node.dataset[dataKey])));
  }
  $$("[data-language]").forEach((button) => {
    const active = button.dataset.language === appState.language;
    button.classList.toggle("active", active);
    button.setAttribute("aria-pressed", String(active));
  });
  const health = $("#service-status");
  if (health) health.lastElementChild.textContent = t(appState.serviceHealth);
  renderModelConfigurationSummary();
}

function openDialog(dialog) {
  if (typeof dialog.showModal === "function") {
    if (!dialog.open) dialog.showModal();
  } else dialog.setAttribute("open", "");
  refreshIcons();
}

function closeDialog(dialog) {
  if (typeof dialog.close === "function") dialog.close();
  else dialog.removeAttribute("open");
}

function openSourceModeDialog() {
  if (!targetEnvironmentComplete()) {
    continueToSources();
    return;
  }
  openDialog($("#source-mode-dialog"));
}

function bindSourceDialogs() {
  $("#change-source-mode").addEventListener("click", openSourceModeDialog);
  $("#choose-manual-source").addEventListener("click", () => chooseSourceMode("manual"));
  $("#choose-discovery-source").addEventListener("click", () => chooseSourceMode("discovery"));
  $("#source-mode-back").addEventListener("click", backToTargetEnvironment);
  $("#source-mode-dialog").addEventListener("cancel", (event) => {
    if (!appState.sourceModeChosen) {
      event.preventDefault();
      backToTargetEnvironment();
    }
  });
  $("#source-candidates-close").addEventListener("click", () => closeDialog($("#source-candidates-dialog")));
  $("#open-source-candidates").addEventListener("click", showSourceCandidates);
  $("#source-candidates-form").addEventListener("submit", selectSourceCandidate);
  $("#source-candidates-retry").addEventListener("click", searchSourcesAgain);
  $("#source-candidates-manual").addEventListener("click", () => {
    closeDialog($("#source-candidates-dialog"));
    chooseSourceMode("manual");
  });
  $("#candidate-api-key").addEventListener("input", (event) => {
    $("#run-form input[name='api_key']").value = event.target.value;
  });
  $("#source-scope-close").addEventListener("click", () => closeDialog($("#source-scope-dialog")));
  $("#open-source-scope").addEventListener("click", showScopeChoices);
  $("#source-scope-form").addEventListener("submit", submitScopeSelection);
  $("#source-scope-form").addEventListener("change", () => {
    if (isResumingScope()) return;
    const draft = currentScopeDraft();
    if (!draft) return;
    draft.mode = $("#source-scope-form input[name='scope_mode']:checked")?.value || "full";
    draft.section_ids = $$("#source-scope-sections input[name='scope_section_id']:checked").map((input) => input.value);
    updateScopeControls();
  });
  $("#scope-api-key").addEventListener("input", (event) => {
    $("#run-form input[name='api_key']").value = event.target.value;
  });
  renderSourceMode();
}

function initializeSetupFlow({ focus = true } = {}) {
  appState.setupStage = "target";
  renderSourceMode();
  if (focus) guideToSection($("#target-environment"));
}

function renderSetupFlow() {
  const stage = appState.setupStage;
  $("#continue-to-sources").hidden = stage !== "target";
  $("#sources-configuration").hidden = stage === "target";
  $("#continue-to-model").hidden = stage !== "sources" || !appState.sourceModeChosen || appState.sourceMode !== "manual";
  // Existing runs must still expose model settings for continuation and T5 reruns.
  $("#model-configuration").hidden = stage !== "model" && !appState.currentRunId;
  $("#run-button").hidden = stage !== "model";
  renderGenerationCoverage();
}

function generationCoveragePlan(value = appState.generationCoverage) {
  return GENERATION_COVERAGE_LEVELS.find((level) => level.value === value) || GENERATION_COVERAGE_LEVELS[1];
}

function generationCoverageLocked() {
  const state = appState.runPayload?.state;
  return Boolean(appState.currentRunId && state && (state.active_task
    || ["queued", "running", "cancelling", "waiting_for_input", "paused"].includes(state.status)
    || isAwaitingSource(state) || isAwaitingScope(state)));
}

function restoreGenerationCoverage(state) {
  if (!appState.currentRunId || appState.generationCoverageRunId === appState.currentRunId) return;
  appState.generationCoverage = generationCoveragePlan(state.generation_coverage ?? "medium").value;
  appState.generationCoverageRunId = appState.currentRunId;
}

function setGenerationCoverage(value) {
  if (generationCoverageLocked() || !GENERATION_COVERAGE_LEVELS.some((level) => level.value === value)) return;
  appState.generationCoverage = value;
  renderGenerationCoverage();
}

function renderGenerationCoverage() {
  const card = $("#generation-coverage");
  if (!card) return;
  const level = generationCoveragePlan();
  const index = GENERATION_COVERAGE_LEVELS.indexOf(level);
  const locked = generationCoverageLocked();
  card.hidden = appState.setupStage !== "model" && !appState.currentRunId;
  card.setAttribute("data-level", level.value);
  card.setAttribute("data-locked", String(locked));
  card.style?.setProperty("--coverage-fill", `${index / 3 * 100}%`);
  const slider = $("#generation-coverage-slider");
  slider.value = String(index);
  slider.disabled = locked;
  slider.setAttribute("aria-valuetext", t(level.label));
  slider.setAttribute("title", locked ? t("This run's coverage is saved. It stays unchanged while the run is in progress.") : "");
  $("#generation-coverage-value").value = level.value;
  $$("[data-coverage-level]").forEach((button) => {
    button.setAttribute("aria-pressed", String(button.dataset.coverageLevel === level.value));
    button.disabled = locked;
  });
}

function bindGenerationCoverage() {
  $("#generation-coverage-slider").addEventListener("input", (event) => {
    setGenerationCoverage(GENERATION_COVERAGE_LEVELS[Number(event.target.value)]?.value);
  });
  $$("[data-coverage-level]").forEach((button) => {
    button.addEventListener("click", () => setGenerationCoverage(button.dataset.coverageLevel));
  });
  renderGenerationCoverage();
}

function collectTargetEnvironmentIssues() {
  const labels = { software_name: "Software", target_version: "Target Version", platform: "Platform" };
  return Object.entries(labels).flatMap(([name, label]) => {
    const field = $("#run-form [name='" + name + "']");
    if (!String(field.value || "").trim()) return [{ field, message: t("Please fill in {field}.", { field: t(label) }) }];
    if (field.validity?.valid === false) return [{ field, message: t("Please check {field}.", { field: t(label) }) }];
    return [];
  });
}

function targetEnvironmentComplete() {
  return collectTargetEnvironmentIssues().length === 0;
}

function continueToSources() {
  const issues = collectTargetEnvironmentIssues();
  if (issues.length) {
    appState.setupStage = "target";
    renderSetupFlow();
    showRunFormIssues(issues);
    return;
  }
  for (const field of appState.formIssues.keys()) clearRunFormIssue(field);
  appState.setupStage = "sources";
  renderSetupFlow();
  openSourceModeDialog();
}

function backToTargetEnvironment() {
  closeDialog($("#source-mode-dialog"));
  appState.setupStage = "target";
  renderSetupFlow();
  guideToSection($("#target-environment"));
}

function collectSourceMaterialIssues() {
  if (appState.sourceMode !== "manual") return [];
  const activeUrls = appState.urls.filter((item) => item.url.trim());
  const field = appState.manualSourceKind === "url" ? $("#url-list .url-value") : $("#dropzone");
  if (appState.files.length + activeUrls.length !== 1) {
    return [{ field, message: t("Please provide exactly one PDF, HTML file, or manual URL.") }];
  }
  if (activeUrls.length) {
    try {
      const url = new URL(activeUrls[0].url.trim());
      if (!["http:", "https:", "file:"].includes(url.protocol) || url.username || url.password) throw new Error();
    } catch {
      return [{ field, message: t("Please enter a valid manual URL.") }];
    }
  }
  return [];
}

function continueToModelConfiguration() {
  if (!targetEnvironmentComplete()) { continueToSources(); return; }
  if (!appState.sourceModeChosen) { openSourceModeDialog(); return; }
  const issues = collectSourceMaterialIssues();
  if (issues.length) { showRunFormIssues(issues); return; }
  for (const field of appState.formIssues.keys()) clearRunFormIssue(field);
  appState.setupStage = "model";
  renderSetupFlow();
  guideToSection($("#model-configuration"));
}

function chooseSourceMode(mode) {
  if (!targetEnvironmentComplete()) { backToTargetEnvironment(); continueToSources(); return; }
  appState.sourceMode = mode;
  appState.sourceModeChosen = true;
  appState.setupStage = "sources";
  closeDialog($("#source-mode-dialog"));
  renderSourceMode();
  updateSourceCount();
  if (mode === "discovery") continueToModelConfiguration();
  else guideToSection($("#sources-configuration"));
}

function guideToSection(target) {
  if (target.open === false) target.open = true;
  const reducedMotion = window.matchMedia?.("(prefers-reduced-motion: reduce)").matches;
  target.scrollIntoView({ behavior: reducedMotion ? "auto" : "smooth", block: "center" });
  target.classList.remove("guided-section");
  // Restart the highlight only after an intentional navigation action.
  void target.offsetWidth;
  target.classList.add("guided-section");
  if (target.id === "target-environment") {
    const firstEmpty = ["software_name", "target_version", "platform"]
      .map((name) => $(`#run-form [name='${name}']`))
      .find((field) => !String(field.value || "").trim());
    (firstEmpty || $("#run-form [name='software_name']"))?.focus({ preventScroll: true });
  } else if (target.id === "sources-configuration") {
    (appState.manualSourceKind === "url" ? $("#url-list .url-value") : $("#dropzone"))?.focus({ preventScroll: true });
  } else if (target.id === "model-configuration") {
    const fields = ["api_base_url", "api_model", "api_key"].map((name) => $("#api-fields [name='" + name + "']"));
    (fields.find((field) => !field.value.trim()) || fields[0])?.focus({ preventScroll: true });
  }
}

function renderSourceMode() {
  const discovery = appState.sourceMode === "discovery";
  $("#source-mode-label").textContent = t(!appState.sourceModeChosen ? "Choose material" : discovery ? "Search for material" : "Your own material");
  $("#source-mode-hint").textContent = t(discovery
    ? "The pipeline will search for manuals after you configure your model."
    : "Add one manual to ground your Skills.");
  $("#source-mode-hint").hidden = !discovery;
  $("#manual-source-inputs").hidden = discovery || !appState.sourceModeChosen;
  $("#source-count").hidden = discovery || !appState.sourceModeChosen;
  $("#model-configuration-hint").hidden = !discovery;
  if (discovery) $("#source-guide").hidden = true;
  renderManualSourceMode();
  renderSetupFlow();
}

function switchManualSourceKind(kind) {
  if (!["file", "url"].includes(kind) || kind === appState.manualSourceKind) return;
  if (kind === "url") {
    appState.manualSourceDrafts.file = appState.files;
    appState.files = [];
    appState.urls = appState.manualSourceDrafts.url.length ? appState.manualSourceDrafts.url : appState.urls;
    appState.manualSourceDrafts.url = [];
    if (!appState.urls.length) appState.urls = [{ id: crypto.randomUUID(), url: "", version: "unknown" }];
  } else {
    appState.manualSourceDrafts.url = appState.urls;
    appState.urls = [];
    appState.files = appState.manualSourceDrafts.file;
    appState.manualSourceDrafts.file = [];
  }
  appState.manualSourceKind = kind;
  $("#form-error").textContent = "";
  renderInputSources();
}

function renderManualSourceMode() {
  const fileMode = appState.manualSourceKind === "file";
  $("#file-source-panel").hidden = !fileMode;
  $("#url-source-panel").hidden = fileMode;
  $("#dropzone").hidden = appState.files.length > 0;
  $("#replace-file").hidden = !appState.files.length;
  $$("[data-source-input]").forEach((button) => {
    button.setAttribute("aria-pressed", String(button.dataset.sourceInput === appState.manualSourceKind));
  });
}

function renderModelConfigurationSummary() {
  const model = $("#api-fields [name='api_model']")?.value?.trim() || "";
  const baseUrl = $("#api-fields [name='api_base_url']")?.value?.trim() || "";
  const configured = Boolean(model && baseUrl);
  const needsAttention = [...appState.formIssues.keys()].some((field) =>
    ["api_base_url", "api_model", "api_key"].includes(field.name));
  const summary = $("#model-summary");
  if (summary) { summary.textContent = model || t("Choose a model"); summary.title = model; }
  const badge = $("#model-configuration-state");
  if (badge) {
    badge.textContent = t(needsAttention ? "Needs attention" : configured ? "Configured" : "Set up");
    badge.title = badge.textContent;
    badge.setAttribute?.("aria-label", badge.textContent);
    badge.setAttribute?.("data-state", needsAttention ? "invalid" : configured ? "configured" : "empty");
  }
}

function discoveryCandidates(payload = appState.runPayload) {
  return (payload?.source_discovery?.candidates || []).slice(0, 3);
}

function discoveryEmptyMessage() {
  const discovery = appState.runPayload?.source_discovery || {};
  // Render host classifications only. Model summaries and upstream errors may
  // contain unverified explanations, request details, or credentials.
  const explanations = {
    search_unsupported: "Sorry, the configured search service did not provide verifiable web search. Please check the search configuration.",
    search_authentication: "Sorry, the search service rejected the API credentials or access permissions. Please check the search configuration.",
    search_quota: "Sorry, the search service has no available quota. Please check its quota or billing settings.",
    search_rate_limit: "Sorry, the search service is rate limited. Please wait before retrying.",
    search_configuration: "Sorry, the search configuration could not be used. Please check the search settings.",
    search_unavailable: "Sorry, the search service failed, so this search could not be completed.",
    source_authentication: "Sorry, some inspected documentation pages required sign-in or authorization, and no usable manual was saved.",
    source_access_denied: "Sorry, some inspected documentation pages refused access, and no usable manual was saved.",
    source_not_found: "Sorry, some inspected documentation links returned 404, and no usable manual was saved.",
    source_unavailable: "Sorry, the inspected documentation could not be retrieved, and no usable manual was saved.",
    no_suitable_sources: "Sorry, no usable manual was found within this search.",
  };
  const categoryReasons = {
    unsupported: "search_unsupported", protocol: "search_unsupported", unverified_retrieval: "search_unsupported",
    authentication: "search_authentication", quota: "search_quota", rate_limit: "search_rate_limit",
    configuration: "search_configuration",
  };
  const hostReason = discovery.failure_summary?.primary_reason;
  const legacyReason = categoryReasons[discovery.search_error?.category];
  const reason = typeof explanations[hostReason] === "string" ? hostReason
    : typeof legacyReason === "string" ? legacyReason
      : ["unavailable", "partial"].includes(discovery.search_status) ? "search_unavailable" : "no_suitable_sources";
  let explanation = explanations[reason];
  if (reason === "no_suitable_sources") {
    if (discovery.completion_reason === "time_budget_reached") {
      explanation = "Sorry, the search time limit was reached before a usable manual was saved.";
    } else if (["runtime_interrupted", "session_ended"].includes(discovery.completion_reason)) {
      explanation = "Sorry, search was interrupted before a usable manual was saved.";
    }
  }
  const failedInspections = discovery.failure_summary?.inspection_counts?.failed;
  const accessNotice = reason === "no_suitable_sources" && Number.isInteger(failedInspections) && failedInspections > 0
    ? `${t("Some inspected material links could not be accessed.")} ` : "";
  return `${t(explanation)} ${accessNotice}${t("You can search again, upload a PDF or HTML manual, or provide a directly accessible official manual link.")}`;
}

function isAwaitingSource(state = appState.runPayload?.state || {}) {
  return state.phase === "awaiting_source_selection"
    || (state.status === "waiting_for_input" && !isAwaitingScope(state));
}

function showSourceCandidates() {
  if (!appState.currentRunId || !isAwaitingSource()) return;
  appState.candidatesRunId = appState.currentRunId;
  $("#source-candidates-error").textContent = "";
  renderCandidateChoices();
  openDialog($("#source-candidates-dialog"));
}

function safeSourceUrl(value) {
  try {
    const url = new URL(String(value));
    return ["http:", "https:"].includes(url.protocol) ? url.href : "";
  } catch { return ""; }
}

function renderCandidateChoices() {
  const list = $("#source-candidates-list");
  const selectedId = list.querySelector("input[name='candidate_id']:checked")?.value;
  const candidates = discoveryCandidates();
  const requiresKey = Boolean(appState.runPayload?.state?.requires_api_key);
  $("#candidate-api-key-field").hidden = !requiresKey;
  $("#candidate-model-route").hidden = !requiresKey;
  const savedModel = appState.runPayload?.state?.model || {};
  $("#candidate-model-route").textContent = t("Continue with saved model: {model} at {url}", {
    model: savedModel.model || "", url: savedModel.base_url || "",
  });
  $("#candidate-api-key").required = requiresKey;
  if (requiresKey && !$("#candidate-api-key").value) {
    $("#candidate-api-key").value = $("#run-form input[name='api_key']").value;
  }
  const profile = appState.runPayload?.state?.target_profile || {};
  $("#source-candidates-summary").textContent = t(candidates.length
    ? "Compare the candidates with your target version {version}. Choose one manual. PDF files will be downloaded; HTML manuals will be crawled."
    : discoveryEmptyMessage(), { version: profile.target_version || t("unknown") });
  const completionReason = appState.runPayload?.source_discovery?.completion_reason;
  const notice = appState.runPayload?.source_discovery?.search_status === "partial"
    ? "Some searches failed. Previously verified and saved manuals are available below."
    : completionReason === "time_budget_reached"
      ? "Search time limit reached. Saved manuals are available below; version verification may be incomplete."
      : ["runtime_interrupted", "session_ended"].includes(completionReason)
        ? "Search ended. Previously verified and saved manuals are available below."
        : "";
  $("#source-candidates-notice").textContent = t(notice);
  $("#source-candidates-notice").hidden = !notice || !candidates.length;
  list.innerHTML = candidates.map((candidate) => {
    const url = safeSourceUrl(candidate.url);
    const origin = ({ official: "Official", third_party: "Third party", unknown: "Unknown source" })[candidate.source_type] || "Unknown source";
    const fields = [
      ["Document version", candidate.document_version || t("unknown")],
      ["Version match", relationLabel(candidate.version_relation) || t("Unknown version")],
      ["Version evidence", candidate.version_evidence],
      ["Manual language", displayLanguage(candidate.language)],
      ["Platform", t(candidate.platform || "unknown")],
      ["Material scope", candidate.html_scope?.scope_summary],
      ["Included topics", candidate.html_scope?.included_sections?.join("; ")],
      ["Outside this collection", candidate.html_scope?.excluded_sections?.join("; ")],
      ["Scope limitations", candidate.html_scope?.limitations?.join("; ")],
      ["Manual root assessment", candidate.root_assessment],
      ["Recommendation", candidate.reason],
    ].filter(([, value]) => value);
    return `<label class="source-candidate">
      <input type="radio" name="candidate_id" value="${escapeHtml(candidate.candidate_id)}" ${candidate.candidate_id === selectedId ? "checked" : ""} required>
      <span class="source-candidate-body">
        <span class="source-candidate-top"><strong>${escapeHtml(candidate.title || candidate.url)}</strong><span class="candidate-type">${escapeHtml(String(candidate.modality || "").toUpperCase())} · ${escapeHtml(t(origin))}</span></span>
        ${url ? `<a href="${escapeHtml(url)}" target="_blank" rel="noopener noreferrer">${escapeHtml(candidate.url)}<i data-lucide="external-link" aria-hidden="true"></i></a>` : `<span>${escapeHtml(candidate.url)}</span>`}
        <dl>${fields.map(([label, value]) => `<dt>${escapeHtml(t(label))}</dt><dd>${escapeHtml(typeof value === "string" ? value : JSON.stringify(value))}</dd>`).join("")}</dl>
      </span>
    </label>`;
  }).join("");
  $("#source-candidates-continue").hidden = !candidates.length;
  $("#source-candidates-continue").disabled = appState.candidateRequestPending;
  $("#source-candidates-retry").disabled = appState.candidateRequestPending;
  refreshIcons();
}

function displayLanguage(value) {
  return ({ Chinese: t("Chinese"), English: t("English"), zh: t("Chinese"), en: t("English") })[value] || value;
}

function continuationApiKey() {
  const key = $("#run-form input[name='api_key']").value.trim();
  if (appState.runPayload?.state?.requires_api_key && !key) {
    throw new Error(t("Enter your model API Key in Model Configuration before continuing."));
  }
  return key;
}

async function submitDiscoveryAction(action, body) {
  if (appState.candidateRequestPending || appState.candidatesRunId !== appState.currentRunId) return;
  const runId = appState.currentRunId;
  appState.candidateRequestPending = true;
  $("#source-candidates-error").textContent = "";
  $("#source-candidates-continue").disabled = true;
  $("#source-candidates-retry").disabled = true;
  try {
    const apiKey = continuationApiKey();
    const response = await fetch(`/api/runs/${encodeURIComponent(runId)}/${action}`, {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ ...body, api_key: apiKey }),
    });
    const payload = await response.json();
    if (!response.ok) throw new Error(formatApiError(payload.detail));
    if (appState.currentRunId !== runId) return;
    closeDialog($("#source-candidates-dialog"));
    if (action === "search-again") appState.shownCandidateRunIds.delete(runId);
    clearTimeout(appState.pollTimer);
    if (appState.currentRunId === runId) await pollRun();
  } catch (error) {
    if (appState.currentRunId !== runId) return;
    $("#source-candidates-error").textContent = error.message;
    if (appState.runPayload?.state?.requires_api_key) $("#model-configuration").open = true;
  } finally {
    appState.candidateRequestPending = false;
    $("#source-candidates-continue").disabled = false;
    $("#source-candidates-retry").disabled = false;
  }
}

async function selectSourceCandidate(event) {
  event.preventDefault();
  const candidateId = $("#source-candidates-list input[name='candidate_id']:checked")?.value;
  if (!candidateId || !discoveryCandidates().some((candidate) => candidate.candidate_id === candidateId)) {
    $("#source-candidates-error").textContent = t("Choose one manual to continue.");
    return;
  }
  await submitDiscoveryAction("select-source", { candidate_id: candidateId });
}

async function searchSourcesAgain() {
  await submitDiscoveryAction("search-again", {});
}

function renderSourceSelectionState(state) {
  const waiting = isAwaitingSource(state);
  $("#source-selection-notice").hidden = !waiting;
  $("#source-selection-notice-text").textContent = t(discoveryCandidates().length
    ? "Search is complete. Choose one manual to continue the pipeline."
    : discoveryEmptyMessage());
  $("#credential-notice").hidden = !state.requires_api_key;
  if (state.requires_api_key) $("#model-configuration").open = true;
  if (!waiting && $("#source-candidates-dialog").open) closeDialog($("#source-candidates-dialog"));
  if (waiting && !appState.shownCandidateRunIds.has(appState.currentRunId)) {
    appState.shownCandidateRunIds.add(appState.currentRunId);
    if (!$("#source-mode-dialog").open) showSourceCandidates();
  }
}

function isResumingScope(state = appState.runPayload?.state || {}) {
  return state.phase === "scope_selected" && state.active_task === false;
}

function isAwaitingScope(state = appState.runPayload?.state || {}) {
  return (state.phase === "awaiting_scope_selection" || isResumingScope(state))
    && !["cancelled", "cancelling", "failed", "completed", "degraded"].includes(state.status);
}

function scopeSections(scope = appState.runPayload?.source_scope) {
  return Array.isArray(scope?.sections) ? scope.sections.filter((section) => (section.depth ?? 0) === 0) : [];
}

function scopeSelectionEnabled(scope = appState.runPayload?.source_scope) {
  return scope?.selection_enabled !== false;
}

function scopeOutlineKey(scope = appState.runPayload?.source_scope) {
  return `${appState.currentRunId}:${scope?.outline_artifact_id || ""}`;
}

function currentScopeDraft() {
  const scope = appState.runPayload?.source_scope;
  if (!scope?.outline_artifact_id) return null;
  const key = scopeOutlineKey(scope);
  if (!appState.scopeDrafts.has(key)) {
    appState.scopeDrafts.set(key, {
      mode: scope.selection?.mode === "selected" ? "selected" : "full",
      section_ids: [...(scope.selection?.section_ids || [])],
    });
  }
  if (isResumingScope() && scope.selection) {
    // A committed choice is immutable. Recovery only repeats that same choice.
    appState.scopeDrafts.set(key, {
      mode: scope.selection.mode,
      section_ids: [...scope.selection.section_ids],
    });
  } else if (!scopeSelectionEnabled(scope)) {
    appState.scopeDrafts.set(key, { mode: "full", section_ids: [] });
  } else {
    const allowedIds = new Set(scopeSections(scope).map((section) => section.section_id));
    const draft = appState.scopeDrafts.get(key);
    draft.section_ids = draft.section_ids.filter((id) => allowedIds.has(id));
  }
  return appState.scopeDrafts.get(key);
}

function showScopeChoices() {
  if (!appState.currentRunId || !isAwaitingScope() || !appState.runPayload?.source_scope?.outline_artifact_id) return;
  appState.scopeRunId = appState.currentRunId;
  appState.scopeOutlineArtifactId = appState.runPayload.source_scope.outline_artifact_id;
  $("#source-scope-error").textContent = "";
  renderScopeChoices();
  openDialog($("#source-scope-dialog"));
}

function scopeSectionDetails(section, modality) {
  if (Number.isInteger(section.direct_char_count)) {
    return t("{count} parsed characters including subsections", {
      count: section.direct_char_count + (section.descendant_char_count || 0),
    });
  }
  if (modality === "pdf") {
    if (Number.isInteger(section.page_start) && Number.isInteger(section.page_end)) {
      return t("Pages {start}–{end}", { start: section.page_start, end: section.page_end });
    }
    if (Number.isInteger(section.page_start)) return t("Starts on page {page}", { page: section.page_start });
    return t("Page boundaries will be resolved during parsing.");
  }
  if (Number.isInteger(section.known_page_count)) {
    return t("{count} known linked pages; more may be found.", { count: section.known_page_count });
  }
  return t("Page links will be expanded after selection.");
}

function updateScopeControls() {
  const draft = currentScopeDraft();
  if (!draft) return;
  const selected = draft.mode === "selected";
  const locked = isResumingScope() || appState.scopeRequestPending;
  $$("#source-scope-sections input[name='scope_section_id']").forEach((input) => { input.disabled = !selected || locked; });
  $("#source-scope-sections").setAttribute("aria-disabled", String(!selected || locked));
  $("#source-scope-selection-count").textContent = t("{count} chapters selected", { count: draft.section_ids.length });
  $("#source-scope-selection-count").hidden = !selected;
  $("#source-scope-continue").disabled = appState.scopeRequestPending || Boolean(appState.runPayload?.state?.active_task);
  $("#source-scope-continue").textContent = t(isResumingScope() ? "Continue extraction" : "Confirm scope and continue");
  $("#scope-mode-full").disabled = locked;
  $("#scope-mode-selected").disabled = locked || !scopeSelectionEnabled() || !scopeSections().length || appState.runPayload?.source_scope?.outline_available === false;
}

function scopeCoverageNotice(scope) {
  const coverage = scope.acquisition_coverage || {};
  const gaps = Array.isArray(coverage.navigation_gaps) ? coverage.navigation_gaps : [];
  const warnings = Array.isArray(scope.warnings) ? scope.warnings : [];
  // Diagnostics remain in the saved outline. The chooser only needs their impact.
  const limited = coverage.truncated === true
    || gaps.some((gap) => ["budget_exhausted", "depth_exhausted"].includes(gap?.status))
    || warnings.some((warning) => typeof warning === "string"
      && /Guided HTML acquisition (budget_exhausted|depth_exhausted):/.test(warning));
  if (limited) return t("This material reached the collection limit. Some linked pages were not included. You can continue with the available chapters.");
  if (coverage.acquisition_complete === false || gaps.length
    || (Array.isArray(coverage.unfetched_discovered_urls) && coverage.unfetched_discovered_urls.length)
    || warnings.some((warning) => typeof warning === "string" && warning.trim())) {
    return t("Some material could not be fully collected or processed. You can continue with the content available for this run.");
  }
  return "";
}

function renderScopeChoices() {
  const scope = appState.runPayload?.source_scope;
  const draft = currentScopeDraft();
  if (!scope || !draft) return;
  const sections = scopeSections(scope);
  const selectable = scopeSelectionEnabled(scope) && sections.length > 0 && scope.outline_available !== false;
  if (!selectable && !isResumingScope()) { draft.mode = "full"; draft.section_ids = []; }
  $("#source-scope-summary").textContent = scope.title || t("Source material");
  $("#scope-mode-full").checked = draft.mode === "full";
  $("#scope-mode-selected").checked = draft.mode === "selected";
  $("#source-scope-unavailable").hidden = selectable || !scopeSelectionEnabled(scope);
  const coverageNotice = scopeCoverageNotice(scope);
  $("#source-scope-warnings").textContent = coverageNotice;
  $("#source-scope-warnings").hidden = !coverageNotice;
  $("#source-scope-sections").innerHTML = sections.map((section) => `<label class="scope-section-option">
    <input type="checkbox" name="scope_section_id" value="${escapeHtml(section.section_id)}" ${draft.section_ids.includes(section.section_id) ? "checked" : ""}>
    <span><strong>${escapeHtml(section.title)}</strong><small>${escapeHtml(scopeSectionDetails(section, scope.modality))}</small></span>
  </label>`).join("");
  $("#source-scope-budget").hidden = scope.modality !== "html";
  $("#source-scope-budget").textContent = scope.structure_review_complete
    ? t("{count} pages of material parsed.", { count: scope.acquisition_coverage?.page_count ?? 0 })
    : t("This run can collect up to {count} linked pages in addition to the starting page.", {
    count: Number.isInteger(scope.max_child_pages) ? scope.max_child_pages : 80,
  });
  const requiresKey = Boolean(appState.runPayload?.state?.requires_api_key);
  $("#scope-api-key-field").hidden = !requiresKey;
  $("#scope-api-key").required = requiresKey;
  if (requiresKey) $("#scope-api-key").value = $("#run-form input[name='api_key']").value;
  updateScopeControls();
}

async function submitScopeSelection(event) {
  event.preventDefault();
  const scope = appState.runPayload?.source_scope;
  const draft = currentScopeDraft();
  if (appState.scopeRequestPending || appState.runPayload?.state?.active_task || !isAwaitingScope() || !scope || !draft
    || appState.scopeRunId !== appState.currentRunId || appState.scopeOutlineArtifactId !== scope.outline_artifact_id) return;
  const allowedIds = new Set(isResumingScope()
    ? (scope.selection?.section_ids || [])
    : scopeSections(scope).map((section) => section.section_id));
  const sectionIds = [...new Set(draft.section_ids)];
  if (draft.mode === "selected" && (!sectionIds.length || sectionIds.some((id) => !allowedIds.has(id)))) {
    $("#source-scope-error").textContent = t("Choose at least one chapter, or select the whole material.");
    return;
  }
  const runId = appState.currentRunId;
  appState.scopeRequestPending = true;
  $("#source-scope-error").textContent = "";
  updateScopeControls();
  try {
    const response = await fetch(`/api/runs/${encodeURIComponent(runId)}/select-scope`, {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ outline_artifact_id: scope.outline_artifact_id, mode: draft.mode,
        section_ids: draft.mode === "selected" ? sectionIds : [], api_key: continuationApiKey() }),
    });
    const payload = await response.json();
    if (!response.ok) throw new Error(formatApiError(payload.detail));
    if (appState.currentRunId !== runId) return;
    closeDialog($("#source-scope-dialog"));
    clearTimeout(appState.pollTimer);
    await pollRun();
  } catch (error) {
    if (appState.currentRunId !== runId) return;
    $("#source-scope-error").textContent = error.message;
    if (appState.runPayload?.state?.requires_api_key) $("#model-configuration").open = true;
  } finally {
    appState.scopeRequestPending = false;
    updateScopeControls();
  }
}

function renderScopeSelectionState(state) {
  const waiting = isAwaitingScope(state);
  $("#scope-selection-notice").hidden = !waiting;
  $("#scope-selection-notice-text").textContent = t(isResumingScope(state)
    ? "Your extraction scope is saved. Continue using the same selection."
    : "Parsing and structure review are complete. Choose the chapters to use for knowledge generation.");
  $("#open-source-scope").textContent = t(isResumingScope(state) ? "Continue extraction" : "Choose chapters");
  if (!waiting && $("#source-scope-dialog").open) closeDialog($("#source-scope-dialog"));
  const key = scopeOutlineKey();
  if (waiting && appState.runPayload?.source_scope?.outline_artifact_id
    && !appState.shownScopeOutlineIds.has(key) && !$("#source-mode-dialog").open) {
    appState.shownScopeOutlineIds.add(key);
    showScopeChoices();
  }
  if (waiting && $("#source-scope-dialog").open) updateScopeControls();
}

function bindForm() {
  $("#continue-to-sources").addEventListener("click", continueToSources);
  $("#continue-to-model").addEventListener("click", continueToModelConfiguration);
  const fileInput = $("#file-input");
  const dropzone = $("#dropzone");
  dropzone.addEventListener("click", () => fileInput.click());
  fileInput.addEventListener("change", () => addFiles(fileInput.files));
  dropzone.addEventListener("dragover", (event) => {
    event.preventDefault();
    dropzone.classList.add("dragging");
  });
  dropzone.addEventListener("dragleave", () => dropzone.classList.remove("dragging"));
  dropzone.addEventListener("drop", (event) => {
    event.preventDefault();
    dropzone.classList.remove("dragging");
    addFiles(event.dataTransfer.files);
  });
  $("#replace-file").addEventListener("click", () => fileInput.click());
  $$("[data-source-input]").forEach((button) => button.addEventListener("click", () => switchManualSourceKind(button.dataset.sourceInput)));
  $("#run-form").addEventListener("submit", submitRun);
  const clearCorrectedField = (event) => {
    if (String(event.target.value || "").trim() && event.target.validity?.valid !== false) {
      clearRunFormIssue(event.target);
    }
  };
  $("#run-form").addEventListener("input", clearCorrectedField);
  $("#run-form").addEventListener("change", clearCorrectedField);
  $("#cancel-run-button").addEventListener("click", cancelCurrentRun);
  $("#rerun-reference-button").addEventListener("click", rerunReferenceFromT5);
  $("#refresh-runs").addEventListener("click", refreshRuns);
  const baseUrlInput = $("#api-fields input[name='api_base_url']");
  const protocolSelect = $("#api-fields select[name='api_protocol']");
  ["api_base_url", "api_model", "api_protocol", "reasoning_effort"].forEach((name) => {
    const edited = () => { appState.editedModelFields.add(name); renderModelConfigurationSummary(); };
    $(`#api-fields [name='${name}']`).addEventListener("input", edited);
    $(`#api-fields [name='${name}']`).addEventListener("change", edited);
  });
  const syncProtocol = () => {
    if (appState.editedModelFields.has("api_protocol")) return;
    try {
      protocolSelect.value = new URL(baseUrlInput.value).pathname.replace(/\/+$/, "").endsWith("/responses")
        ? "responses" : "";
    } catch { protocolSelect.value = ""; }
  };
  baseUrlInput.addEventListener("input", syncProtocol);
  baseUrlInput.addEventListener("change", syncProtocol);
  syncProtocol();
  renderModelConfigurationSummary();
}

async function loadModelDefaults() {
  try {
    const response = await fetch("/api/model-defaults");
    if (!response.ok) throw new Error();
    const defaults = await response.json();
    const names = { api_base_url: "base_url", api_model: "model", api_protocol: "api_protocol", reasoning_effort: "reasoning_effort" };
    for (const [name, key] of Object.entries(names)) {
      if (appState.editedModelFields.has(name)) continue;
      // Editing the endpoint while this request was pending also changes which
      // protocol may be inferred. Keep the user's selection in that case.
      if (name === "api_protocol" && appState.editedModelFields.has("api_base_url")) continue;
      if (typeof defaults[key] === "string") {
        const field = $(`#api-fields [name='${name}']`);
        field.value = defaults[key];
        if (field.value.trim()) clearRunFormIssue(field);
      }
    }
  } catch {
    // Users can enter their provider settings if the defaults request failed.
  } finally {
    renderModelConfigurationSummary();
  }
}

function bindTabs() {
  $$(".tab").forEach((button) => {
    button.addEventListener("click", () => activateTab(button.dataset.tab, true));
  });
  const requestedTab = new URLSearchParams(window.location.search).get("tab");
  if (requestedTab === "knowledge") {
    activateTab("task-trajectory", true);
  } else if ($$(".tab").some((button) => button.dataset.tab === requestedTab)) {
    activateTab(requestedTab, false);
  } else if (requestedTab) {
    activateTab("sources", true);
  }
  const nav = $(".tabs");
  if (typeof ResizeObserver === "function") {
    const observer = new ResizeObserver(updateTabIndicator);
    observer.observe(nav);
    $$(".tab").forEach((button) => observer.observe(button));
  }
  window.addEventListener("resize", updateTabIndicator, { passive: true });
  document.fonts?.ready.then(updateTabIndicator);
  updateTabIndicator();
}

function updateTabIndicator() {
  const nav = $(".tabs");
  const selected = $(".tab.active");
  if (!nav || !selected?.offsetWidth) return;
  nav.style.setProperty("--tab-indicator-x", `${selected.offsetLeft}px`);
  nav.style.setProperty("--tab-indicator-width", `${selected.offsetWidth}px`);
  nav.classList.add("has-tab-indicator");
}

function activateTab(name, updateUrl) {
  $$(".tab").forEach((item) => {
    const active = item.dataset.tab === name;
    item.classList.toggle("active", active);
    if (active) item.setAttribute("aria-current", "page");
    else item.removeAttribute("aria-current");
  });
  $$(".tab-panel").forEach((panel) => panel.classList.toggle("active", panel.id === `tab-${name}`));
  updateTabIndicator();
  if (updateUrl) {
    const url = new URL(window.location.href);
    url.searchParams.set("tab", name);
    window.history.replaceState({}, "", url);
  }
}

function bindFilters() {
  $("#evidence-filter").addEventListener("input", renderEvidence);
  $("#evidence-modality").addEventListener("change", renderEvidence);
  $$("[data-evidence-view]").forEach((button) => {
    button.addEventListener("click", () => {
      appState.evidenceView = button.dataset.evidenceView;
      $$("[data-evidence-view]").forEach((item) => {
        const active = item === button;
        item.classList.toggle("active", active);
        item.setAttribute("aria-pressed", String(active));
      });
      renderEvidence();
    });
  });
  $("#audit-filters").addEventListener("click", (event) => {
    const button = event.target.closest("button[data-level]");
    if (!button) return;
    appState.auditFilter = button.dataset.level;
    $$("#audit-filters button").forEach((item) => item.classList.toggle("active", item === button));
    renderAudit();
  });
}

function bindReferenceCatalogDialog() {
  const dialog = $("#reference-catalog-dialog");
  const closeButton = $("#reference-catalog-dialog-close");
  const filter = $("#reference-catalog-filter");
  closeButton.addEventListener("click", () => dialog.close());
  dialog.addEventListener("click", (event) => {
    if (event.target === dialog) dialog.close();
  });
  dialog.addEventListener("close", () => {
    appState.openReferenceCatalog = null;
    filter.value = "";
  });
  filter.addEventListener("input", () => {
    if (appState.openReferenceCatalog) {
      renderReferenceCatalogEntries(appState.openReferenceCatalog, filter.value);
    }
  });
}

function bindPipelineErrorDialog() {
  const dialog = $("#pipeline-error-dialog");
  $("#pipeline-error-dialog-close").addEventListener("click", () => dialog.close());
  dialog.addEventListener("click", (event) => {
    if (event.target === dialog) dialog.close();
  });
}

function showPipelineError(title, message, details = {}) {
  const dialog = $("#pipeline-error-dialog");
  $("#pipeline-error-dialog-title").textContent = t(title || "Pipeline Run Failed");
  $("#pipeline-error-dialog-message").textContent = t(message || "The pipeline encountered an unrecovered error.");
  const detailEntries = Object.entries(details).filter(([, value]) => value !== undefined && value !== null && value !== "");
  $("#pipeline-error-dialog-details").innerHTML = detailEntries.map(([label, value]) => (
    `<dt>${escapeHtml(t(label))}</dt><dd>${escapeHtml(String(value))}</dd>`
  )).join("");
  if (typeof dialog.showModal === "function") {
    if (!dialog.open) dialog.showModal();
  } else {
    dialog.setAttribute("open", "");
  }
  refreshIcons();
}

function runFailureTitle(error = {}) {
  const category = String(error.category || "");
  const stage = String(error.details?.stage || "");
  const message = String(error.message || "");
  if (category === "no_usable_sources") return "No Usable Sources";
  if (category === "quota_exhausted") return "API Quota Exhausted";
  if (category === "api_timeout") return "API Request Timed Out";
  if (category === "rate_limited") return "API Rate Limit Exceeded";
  if (category === "authentication_error") return "API Authentication Failed";
  if (category === "permission_denied") return "Insufficient API Permissions";
  if (category === "api_connection_error") return "Cannot Connect to API Service";
  if (category === "invalid_api_response") return "Invalid API Response";
  if (category === "agent_stage_failed" || category.endsWith("_agent_failed")) {
    const failedStage = stage || (message.match(/^([a-z_]+)\s+(?:API\s+)?Agent\b/i)?.[1] || "");
    if (failedStage === "trajectory_inference") return "Trajectory Agent Produced No Valid Candidates";
    if (failedStage === "skill_induction") return "Skill Agent Produced No Valid Candidates";
    if (failedStage === "task_discovery") return "Task Discovery Agent Produced No Valid Candidates";
    if (failedStage === "reference_build") return "Reference Agent Produced No Valid Candidates";
    if (failedStage === "knowledge_alignment") return "Coverage Gap Audit Agent Failed";
    return "Agent Stage Failed";
  }
  if (category === "react_step_budget_exhausted") return "Agent Reached the ReAct Step Limit";
  if (category.startsWith("api_") || category === "api_agent_error") return "API Agent Failed";
  return "Pipeline Run Failed";
}

function showRunFailureIfNeeded() {
  const state = appState.runPayload?.state || {};
  if (state.status !== "failed" || !appState.currentRunId) return;
  if (appState.shownFailureRunIds.has(appState.currentRunId)) return;
  appState.shownFailureRunIds.add(appState.currentRunId);
  const error = state.error || {};
  const noUsableSources = error.category === "no_usable_sources";
  const sourceDetails = error.details || {};
  const details = {
    "Run ID": appState.currentRunId,
    "Error Category": error.category,
    "HTTP Status": error.status_code,
    "Exception Type": error.type,
  };
  if (noUsableSources) {
    details["Total Sources"] = sourceDetails.input_count;
    details["Excluded Sources"] = sourceDetails.excluded_count;
    details["Acquisition Failures"] = sourceDetails.failed_count;
    if (sourceDetails.reason_counts && typeof sourceDetails.reason_counts === "object") {
      details["Reason Counts"] = Object.entries(sourceDetails.reason_counts)
        .map(([reason, count]) => `${reason}: ${count}`)
        .join("；");
    }
  }
  showPipelineError(
    runFailureTitle(error),
    error.message || (
      noUsableSources
        ? t("All sources were excluded or failed during acquisition. The pipeline has stopped.")
        : t("The pipeline has stopped, but the backend did not provide detailed error information.")
    ),
    details,
  );
}

function addFiles(fileList) {
  const incoming = Array.from(fileList);
  if (incoming.length > 1 || incoming.length + appState.urls.filter((item) => item.url.trim()).length > 1) {
    $("#form-error").textContent = t("Use only one PDF, HTML file, or manual URL for this run.");
    $("#file-input").value = "";
    return;
  }
  incoming.forEach((file) => {
    if (!/\.(pdf|html?)$/i.test(file.name)) {
      $("#form-error").textContent = t("Choose a PDF or HTML file.");
      return;
    }
    const key = `${file.name}:${file.size}:${file.lastModified}`;
    appState.files = [{ key, file, version: "unknown" }];
    $("#form-error").textContent = "";
  });
  $("#file-input").value = "";
  renderInputSources();
}

function addUrlRow(url = "", version = "unknown") {
  if (appState.urls.length) return;
  appState.urls.push({ id: crypto.randomUUID(), url, version });
  renderInputSources();
}

function renderInputSources() {
  const fileList = $("#file-list");
  fileList.innerHTML = appState.files.map((item) => `
    <div class="source-entry" data-key="${escapeHtml(item.key)}">
      <div class="source-file-head">
        <span class="source-file-type" data-format="${/\.pdf$/i.test(item.file.name) ? "pdf" : "html"}" aria-hidden="true">${/\.pdf$/i.test(item.file.name) ? "PDF" : "HTML"}</span>
        <div class="source-file-meta"><span class="source-name" title="${escapeHtml(item.file.name)}">${escapeHtml(item.file.name)}</span><small class="source-file-size">${escapeHtml(t(/\.pdf$/i.test(item.file.name) ? "PDF document" : "HTML document"))}${Number.isFinite(item.file.size) ? ` · ${formatSourceFileSize(item.file.size)}` : ""}</small></div>
        <button class="icon-button remove-file" type="button" title="${t("Remove")}" aria-label="${escapeHtml(t("Remove {name}", { name: item.file.name }))}"><i data-lucide="x" aria-hidden="true"></i></button>
      </div>
      <label class="source-version-field"><span>${t("Source version")} <small>${t("Optional")}</small></span><input class="file-version" value="${escapeHtml(sourceVersionInputValue(item.version))}" placeholder="${t("e.g. 1.9")}" aria-label="${escapeHtml(t("Version of {name}", { name: item.file.name }))}"></label>
    </div>
  `).join("");
  fileList.querySelectorAll(".source-entry").forEach((row) => {
    const item = appState.files.find((candidate) => candidate.key === row.dataset.key);
    row.querySelector(".file-version").addEventListener("input", (event) => { item.version = event.target.value.trim() || "unknown"; });
    row.querySelector(".remove-file").addEventListener("click", () => {
      appState.files = appState.files.filter((candidate) => candidate.key !== row.dataset.key);
      renderInputSources();
    });
  });

  const urlList = $("#url-list");
  urlList.innerHTML = appState.urls.map((item) => `
    <div class="url-entry" data-id="${item.id}">
      <label class="source-url-field"><span>${t("Manual link")}</span><span class="source-url-control"><input class="url-value" value="${escapeHtml(item.url)}" placeholder="${t("https://... or file:///C:/...")}" aria-label="${t("Source URL")}"><button class="icon-button remove-url" type="button" title="${t("Clear link")}" aria-label="${t("Clear link")}" ${item.url.trim() ? "" : "hidden"}><i data-lucide="x" aria-hidden="true"></i></button></span></label>
      <label class="source-version-field"><span>${t("Source version")} <small>${t("Optional")}</small></span><input class="url-version" value="${escapeHtml(sourceVersionInputValue(item.version))}" placeholder="${t("e.g. 1.9")}" aria-label="${t("URL source version")}"></label>
    </div>
  `).join("");
  urlList.querySelectorAll(".url-entry").forEach((row) => {
    const item = appState.urls.find((candidate) => candidate.id === row.dataset.id);
    row.querySelector(".url-value").addEventListener("input", (event) => { item.url = event.target.value; row.querySelector(".remove-url").hidden = !item.url.trim(); updateSourceCount(); });
    row.querySelector(".url-version").addEventListener("input", (event) => { item.version = event.target.value.trim() || "unknown"; });
    row.querySelector(".remove-url").addEventListener("click", () => {
      appState.urls = appState.urls.filter((candidate) => candidate.id !== row.dataset.id);
      if (!appState.urls.length) addUrlRow(); else renderInputSources();
    });
  });
  renderManualSourceMode();
  updateSourceCount();
  if (appState.files.length + appState.urls.filter((item) => item.url.trim()).length === 1) {
    for (const field of appState.formIssues.keys()) {
      if (field.id === "dropzone" || field.classList.contains("url-value")) clearRunFormIssue(field);
    }
  }
  refreshIcons();
}

function updateSourceCount() {
  const count = appState.files.length + appState.urls.filter((item) => item.url.trim()).length;
  $("#source-count").textContent = t(count ? "Material added" : "Add material");
  $("#source-count").setAttribute("data-state", count ? "ready" : "empty");
  $("#source-guide").hidden = !appState.sourceModeChosen || appState.sourceMode !== "manual" || count > 0;
  $("#dropzone").disabled = appState.urls.some((item) => item.url.trim());
  $$("#url-list .url-value").forEach((input) => { input.disabled = appState.files.length > 0; });
  if (appState.setupStage === "model" && appState.sourceMode === "manual" && count !== 1) appState.setupStage = "sources";
  renderSetupFlow();
}

function sourceVersionInputValue(version) {
  const value = String(version || "");
  return value.trim().toLowerCase() === "unknown" ? "" : value;
}

function formatSourceFileSize(size) {
  if (size < 1024) return `${size} B`;
  if (size < 1024 * 1024) return `${Math.round(size / 1024)} KB`;
  return `${(size / (1024 * 1024)).toFixed(1)} MB`;
}

function clearRunFormIssue(field) {
  const issue = appState.formIssues.get(field);
  if (!issue) return;
  field.classList.remove("is-invalid");
  field.removeAttribute("aria-invalid");
  const descriptions = (field.getAttribute("aria-describedby") || "").split(/\s+/).filter((id) => id && id !== issue.hint.id);
  if (descriptions.length) field.setAttribute("aria-describedby", descriptions.join(" "));
  else field.removeAttribute("aria-describedby");
  issue.wrapper.classList.remove("validation-attention");
  issue.hint.remove();
  appState.formIssues.delete(field);
  if (!appState.formIssues.size) $("#form-error").textContent = "";
  renderModelConfigurationSummary();
}

function showRunFormIssues(issues) {
  for (const field of appState.formIssues.keys()) clearRunFormIssue(field);
  const configuration = $("#model-configuration");
  if (issues.some((issue) => configuration.contains(issue.field))) {
    configuration.hidden = false;
    configuration.open = true;
  }
  issues.forEach(({ field, message }, index) => {
    const wrapper = field.closest(".field, .source-url-field") || field;
    const hint = document.createElement("span");
    hint.id = `run-field-error-${field.name || field.id || index}`;
    hint.className = "field-validation-message";
    hint.textContent = message;
    if (wrapper === field) field.insertAdjacentElement("afterend", hint);
    else wrapper.appendChild(hint);
    field.classList.add("is-invalid");
    field.setAttribute("aria-invalid", "true");
    field.setAttribute("aria-describedby", [field.getAttribute("aria-describedby"), hint.id].filter(Boolean).join(" "));
    appState.formIssues.set(field, { wrapper, hint });
    wrapper.classList.remove("validation-attention");
    void wrapper.offsetWidth;
    wrapper.classList.add("validation-attention");
  });
  $("#form-error").textContent = issues.length === 1 ? issues[0].message : t("Please complete the highlighted required fields.");
  renderModelConfigurationSummary();
  const first = issues[0]?.field;
  if (first) {
    const reducedMotion = window.matchMedia?.("(prefers-reduced-motion: reduce)").matches;
    first.scrollIntoView({ behavior: reducedMotion ? "auto" : "smooth", block: "center" });
    first.focus({ preventScroll: true });
  }
}

function collectRunFormIssues() {
  const labels = { software_name: "Software", target_version: "Target Version", platform: "Platform", api_base_url: "Base URL", api_model: "Model" };
  const issues = [];
  $$("#run-form [required]").forEach((field) => {
    if (field.disabled) return;
    const label = t(labels[field.name] || field.name);
    if (!String(field.value || "").trim()) {
      issues.push({ field, message: t("Please fill in {field}.", { field: label }) });
    } else if (field.validity?.valid === false) {
      issues.push({ field, message: t("Please check {field}.", { field: label }) });
    }
  });
  const baseUrl = $("#api-fields [name='api_base_url']");
  if (baseUrl.value.trim() && !issues.some((issue) => issue.field === baseUrl)) {
    try {
      const url = new URL(baseUrl.value.trim());
      if (!["http:", "https:"].includes(url.protocol) || url.username || url.password) throw new Error();
    } catch { issues.push({ field: baseUrl, message: t("Please enter a valid Base URL (http:// or https://).") }); }
  }
  issues.push(...collectSourceMaterialIssues());
  return issues;
}

function runFormServerIssues(detail) {
  // The server resolves any process-configured credential. Keeping this check
  // authoritative there lets a blank form Key work when one is already supplied.
  if (typeof detail === "string" && detail.includes("API Agent requires both API Key and model name")) {
    const key = $("#api-fields [name='api_key']");
    if (!key.value.trim()) return [{ field: key, message: t("Please enter your API Key before generating.") }];
  }
  return [];
}

async function submitRun(event) {
  event.preventDefault();
  if (appState.setupStage === "target") { continueToSources(); return; }
  if (appState.setupStage === "sources") { continueToModelConfiguration(); return; }
  const errorNode = $("#form-error");
  errorNode.textContent = "";
  const issues = collectRunFormIssues();
  if (issues.length) {
    showRunFormIssues(issues);
    return;
  }
  for (const field of appState.formIssues.keys()) clearRunFormIssue(field);
  const activeUrls = appState.urls.filter((item) => item.url.trim());
  const form = new FormData(event.target);
  form.set("source_mode", appState.sourceMode);
  form.set("output_language", appState.language);
  form.set("generation_coverage", generationCoveragePlan().value);
  const manual = appState.sourceMode === "manual";
  form.set("file_versions_json", JSON.stringify(manual ? appState.files.map((item) => item.version || "unknown") : []));
  form.set("urls_json", JSON.stringify(manual ? activeUrls.map((item) => ({ url: item.url.trim(), user_declared_version: item.version || "unknown" })) : []));
  form.delete("files");
  if (manual) appState.files.forEach((item) => form.append("files", item.file, item.file.name));

  const button = $("#run-button");
  button.disabled = true;
  button.querySelector("span").textContent = t("Registering");
  try {
    const response = await fetch("/api/runs", { method: "POST", body: form });
    const payload = await response.json();
    if (!response.ok) {
      const serverIssues = runFormServerIssues(payload.detail);
      if (serverIssues.length) { showRunFormIssues(serverIssues); return; }
      throw new Error(formatApiError(payload.detail));
    }
    await selectRun(payload.run_id);
    await refreshRuns();
  } catch (error) {
    errorNode.textContent = error.message;
    showPipelineError(t("Unable to Start Pipeline"), error.message);
  } finally {
    button.disabled = false;
    button.querySelector("span").textContent = t("Generate");
  }
}

async function checkHealth() {
  const node = $("#service-status");
  try {
    const response = await fetch("/api/health");
    if (!response.ok) throw new Error();
    node.className = "service-status online";
    appState.serviceHealth = "Service available";
    node.lastElementChild.textContent = t(appState.serviceHealth);
  } catch {
    node.className = "service-status offline";
    appState.serviceHealth = "Service unavailable";
    node.lastElementChild.textContent = t(appState.serviceHealth);
  }
}

async function refreshRuns() {
  try {
    const response = await fetch("/api/runs?limit=60");
    const payload = await response.json();
    const list = $("#recent-list");
    list.innerHTML = (payload.runs || []).map((run) => {
      const status = getRunDisplayStatus(run);
      return `
      <button class="recent-run ${run.run_id === appState.currentRunId ? "active" : ""}" type="button" data-run-id="${escapeHtml(run.run_id)}">
        <span><strong>${escapeHtml(run.target_profile?.software_name || run.run_id)}</strong><small>${escapeHtml(run.run_id)} · ${escapeHtml(getRunStageLabel(run, status))}</small></span>
        <span class="status-badge ${statusClass(status)}">${escapeHtml(statusLabel(status))}</span>
      </button>
    `;
    }).join("") || `<div class="empty-list">${t("No runs yet")}</div>`;
    list.querySelectorAll(".recent-run").forEach((button) => button.addEventListener("click", () => selectRun(button.dataset.runId)));
  } catch {
    // Health indicator already communicates connection state.
  }
}

async function selectRun(runId) {
  if ($("#source-candidates-dialog").open) closeDialog($("#source-candidates-dialog"));
  if ($("#source-scope-dialog").open) closeDialog($("#source-scope-dialog"));
  appState.currentRunId = runId;
  appState.shownCandidateRunIds.delete(runId);
  appState.shownScopeOutlineIds.forEach((key) => { if (key.startsWith(`${runId}:`)) appState.shownScopeOutlineIds.delete(key); });
  document.body.classList.add("has-run");
  appState.events = [];
  appState.eventOffset = 0;
  clearTimeout(appState.pollTimer);
  const keepCurrentView = Boolean(appState.runPayload);
  $("#empty-state").hidden = true;
  $("#run-loading").hidden = keepCurrentView;
  $("#run-view").hidden = !keepCurrentView;
  const url = new URL(window.location.href);
  url.searchParams.set("run", runId);
  window.history.replaceState({}, "", url);
  await pollRun();
  refreshRuns();
}

async function pollRun() {
  if (!appState.currentRunId) return;
  const runId = appState.currentRunId;
  const eventOffset = appState.eventOffset;
  try {
    const [runResponse, eventsResponse] = await Promise.all([
      fetch(`/api/runs/${encodeURIComponent(runId)}`),
      fetch(`/api/runs/${encodeURIComponent(runId)}/events?after=${eventOffset}&limit=500`),
    ]);
    if (!runResponse.ok) throw new Error(t("Run not found"));
    const payload = await runResponse.json();
    if (appState.currentRunId !== runId) return;
    appState.runPayload = payload;
    let hasMoreEvents = false;
    if (eventsResponse.ok) {
      const eventPayload = await eventsResponse.json();
      if (appState.currentRunId !== runId) return;
      appState.events.push(...(eventPayload.events || []));
      appState.eventOffset = eventPayload.next_offset;
      hasMoreEvents = (eventPayload.events || []).length >= 500 && appState.eventOffset > eventOffset;
    }
    renderRun();
    showRunFailureIfNeeded();
    const status = appState.runPayload.state.status;
    if (hasMoreEvents || ["queued", "running", "cancelling", "waiting_for_input"].includes(status)) {
      clearTimeout(appState.pollTimer);
      appState.pollTimer = setTimeout(pollRun, 900);
    } else {
      await refreshRuns();
    }
  } catch (error) {
    $("#run-loading").hidden = true;
    if (!appState.runPayload) {
      $("#run-view").hidden = true;
      $("#empty-state").hidden = false;
    }
    $("#form-error").textContent = error.message;
  }
}

async function cancelCurrentRun() {
  if (!appState.currentRunId) return;
  const button = $("#cancel-run-button");
  button.disabled = true;
  button.querySelector("span").textContent = t("Stopping");
  try {
    const response = await fetch(`/api/runs/${encodeURIComponent(appState.currentRunId)}/cancel`, {
      method: "POST",
    });
    const payload = await response.json();
    if (!response.ok) throw new Error(formatApiError(payload.detail));
    await pollRun();
  } catch (error) {
    $("#form-error").textContent = error.message;
  } finally {
    button.querySelector("span").textContent = t("Stop Run");
  }
}

async function rerunReferenceFromT5() {
  if (!appState.currentRunId) return;
  const button = $("#rerun-reference-button");
  const errorNode = $("#form-error");
  errorNode.textContent = "";
  const form = new FormData($("#run-form"));
  for (const key of Array.from(form.keys())) {
    if (!["api_protocol", "api_base_url", "api_model", "reasoning_effort", "api_key"].includes(key)) {
      form.delete(key);
    }
  }
  form.set("output_language", appState.language);
  form.set("generation_coverage", generationCoveragePlan().value);
  button.disabled = true;
  button.querySelector("span").textContent = t("Starting Pipeline");
  try {
    const sourceRunId = appState.currentRunId;
    const response = await fetch(`/api/runs/${encodeURIComponent(sourceRunId)}/rerun-reference`, {
      method: "POST",
      body: form,
    });
    const payload = await response.json();
    if (!response.ok) throw new Error(formatApiError(payload.detail));
    await selectRun(payload.run_id);
    await refreshRuns();
  } catch (error) {
    errorNode.textContent = error.message;
    showPipelineError(t("Unable to Start Pipeline from T5"), error.message);
  } finally {
    button.disabled = false;
    button.querySelector("span").textContent = t("Run Full Pipeline from T5");
  }
}

function getRunDisplayStatus(state, result) {
  const status = ["queued", "running", "cancelling", "waiting_for_input", "paused", "failed", "cancelled"].includes(state.status)
    ? state.status : result?.outcome || state.outcome || state.status || "queued";
  if (!["succeeded", "completed", "degraded"].includes(status)) return status;
  // Result arrays and state counts contain validated artifacts, not proposals.
  // Keep acquisition/trajectory gaps in the backend and audit data.
  const skills = result?.skills ?? state.validated_skills;
  const skillCount = Array.isArray(skills) ? getDisplayableSkills(skills).length : state.skill_count || 0;
  const referenceCount = Array.isArray(result?.references) ? result.references.length : state.reference_count || 0;
  return skillCount > 0 && referenceCount > 0 ? "succeeded" : "degraded";
}

function getRunStageLabel(state, status) {
  const phase = ["succeeded", "completed"].includes(status) ? "completed"
    : status === "degraded" ? "degraded" : state.phase || state.stage || "registered";
  return stageName(phase);
}

function renderRun() {
  restoreGenerationCoverage(appState.runPayload?.state || {});
  renderSetupFlow();
  $("#empty-state").hidden = true;
  $("#run-loading").hidden = true;
  $("#run-view").hidden = false;
  updateTabIndicator();
  const state = appState.runPayload?.state || {};
  const result = appState.runPayload?.result;
  const status = getRunDisplayStatus(state, result);
  $("#run-title").textContent = state.target_profile?.software_name || t("Run Results");
  const badge = $("#run-status");
  badge.textContent = statusLabel(status);
  badge.className = `status-badge ${statusClass(status)}`;
  const stageProgress = getPipelineProgress(state);
  const progress = stageProgress.percent;
  $("#progress-fill").style.setProperty("--progress-numeric-width", `${progress}%`);
  $("#progress-fill").style.setProperty("--progress-stage-width", stageProgress.trackInset
    ? `max(0px, calc(${stageProgress.trackPercent}% - ${stageProgress.trackInset}px))`
    : `${stageProgress.trackPercent}%`);
  $("#progress-label").textContent = `${progress}%`;
  $("#progress-fill").classList.toggle("is-active", status === "running" && progress < 100);
  const stageLabel = getRunStageLabel(state, status);
  $("#stage-label").textContent = stageLabel;
  $(".progress-band").dataset.status = status;
  $("#progress-status").textContent = isAwaitingSource(state) || isAwaitingScope(state) ? t("Waiting for your selection") : statusLabel(status);
  $(".progress-track").setAttribute("aria-valuenow", String(progress));
  $(".progress-track").setAttribute("aria-valuetext", `${progress}% · ${stageLabel}`);
  renderStageRail(state, stageProgress);
  const cancelButton = $("#cancel-run-button");
  const activelyExecuting = Boolean(state.active_task) || ["queued", "running", "cancelling"].includes(state.status);
  const canCancel = activelyExecuting || state.status === "paused" || isAwaitingSource(state) || isAwaitingScope(state);
  cancelButton.hidden = !canCancel;
  cancelButton.disabled = state.status === "cancelling";
  cancelButton.querySelector("span").textContent = t(state.status === "cancelling" ? "Stopping" : "Stop Run");
  const rerunReferenceButton = $("#rerun-reference-button");
  rerunReferenceButton.hidden = !state.t5_stage_complete || activelyExecuting;
  rerunReferenceButton.disabled = state.status === "cancelling";
  renderSources();
  renderEvidence();
  renderReferences();
  renderTasksAndTrajectories();
  renderSkills();
  renderAudit();
  renderSourceSelectionState(state);
  renderScopeSelectionState(state);
  refreshIcons();
}

function getPipelineProgress(state) {
  // Group backend phases for display without changing the pipeline itself.
  const groups = [
    ["created", "registered", "acquiring", "acquisition", "previewing_scope", "awaiting_scope_selection", "scope_selected", "normalizing", "modal_preprocessing", "source_review", "source_validation", "version_alignment", "corpus_snapshot", "corpus_frozen"],
    ["reference_build", "reference_building", "validate_references", "references_ready"],
    ["task_and_reference", "task_discovery", "discovering_task_trajectories", "discovering_knowledge", "tasks_ready", "validate_tasks", "aligning_knowledge", "knowledge_alignment", "knowledge_snapshot", "knowledge_frozen", "inferring_trajectories", "trajectory_inference", "trajectory_validation"],
    ["inducing_skills", "skill_induction", "skill_validation", "validating", "committing", "registry_commit", "completed", "degraded"],
  ];
  const discovery = state.source_mode === "discovery";
  const positions = discovery ? [20, 25, 35, 50, 75] : [10, 20, 40, 70];
  if (discovery) {
    groups[0] = groups[0].filter((phase) => !["created", "registered"].includes(phase));
    groups.unshift(["created", "registered", "discovering_sources", "awaiting_source_selection", "source_selected"]);
  }
  const terminalPhases = ["completed", "degraded", "failed", "cancelled"];
  const knownPhase = (phase) => terminalPhases.includes(phase) || groups.some((phases) => phases.includes(phase));
  const current = knownPhase(state.phase) ? state.phase : state.stage;
  const currentIndex = groups.findIndex((phases) => phases.includes(current));
  const interrupted = ["failed", "cancelled"].includes(state.status);
  const finished = ["completed", "degraded"].includes(current) && !interrupted;
  const stopped = terminalPhases.includes(current) || interrupted;
  const checkpoints = [
    state.t5_stage_complete || current === "corpus_frozen",
    state.t6_reference_complete || current === "references_ready",
    state.t7_trajectory_complete,
    finished,
  ];
  if (discovery) checkpoints.unshift(current === "source_selected" || Boolean(state.selected_candidate_id));
  const stages = groups.map((_, index) => {
    const done = finished || index < currentIndex || Boolean(checkpoints[index]);
    const active = !stopped && !done && index === currentIndex;
    const paused = active && (state.status === "paused" || state.status === "waiting_for_input" || isResumingScope(state) || ["awaiting_source_selection", "awaiting_scope_selection"].includes(current));
    const label = done ? "Completed" : active
      ? isResumingScope(state) ? "Waiting to continue" :
      ["awaiting_source_selection", "awaiting_scope_selection"].includes(current) ? "Waiting for your selection" :
      paused ? "Paused" : state.status === "cancelling" ? "Stopping" : state.status === "queued" ? "Queued" :
      ["validating", "committing", "registry_commit"].includes(current) ? "Finalizing" : "In progress"
      : "Pending";
    return { done, active, paused, label: t(label) };
  });
  const activeIndex = stages.findIndex((stage) => stage.active);
  const firstIncomplete = stages.findIndex((stage) => !stage.done);
  const completedCount = firstIncomplete < 0 ? stages.length : firstIncomplete;
  // Active stages keep their display value; checkpoints advance to the next interval.
  const rawPercent = finished ? 100 : activeIndex < 0
    ? completedCount === 0 ? 0 : (positions[completedCount - 1] + (positions[completedCount] ?? 100)) / 2
    : ["created", "registered"].includes(current)
      ? 0 : positions[activeIndex];
  const percent = Number(rawPercent.toFixed(1));
  // The rail is evenly spaced; its visual fill follows stages, independently of the numeric scale.
  const trackPercent = percent === 0 ? 0 : finished ? 100
    : (activeIndex >= 0 ? activeIndex + 0.5 : completedCount) / stages.length * 100;
  const trackInset = activeIndex >= 0 && percent > 0 ? 16 : 0;
  return { stages, percent, positions, trackPercent, trackInset };
}

function renderStageRail(state, progress = getPipelineProgress(state)) {
  const rail = $("#stage-rail");
  const definitions = [
    ["source_evidence", "Source & Evidence Build"], ["reference_build", "Reference Build"],
    ["task_trajectory", "Task & Trajectory Generation"], ["skill_induction", "Skill Generation"],
  ];
  if (state.source_mode === "discovery") definitions.unshift(["source_discovery", "Find Material"]);
  rail.dataset.sourceMode = state.source_mode === "discovery" ? "discovery" : "manual";
  rail.style.setProperty("--stage-count", definitions.length);
  if (rail.children.length !== definitions.length) {
    rail.innerHTML = definitions.map(([key, title]) => `<div class="stage-step" data-stage="${key}" role="listitem"><div class="stage-copy"><span class="stage-title">${escapeHtml(t(title))}</span><span class="stage-status"></span></div></div>`).join("");
  }
  $$("#stage-rail > .stage-step").forEach((node, index) => {
    node.querySelector(".stage-title").textContent = t(definitions[index][1]);
    const stage = progress.stages[index];
    node.classList.toggle("done", stage.done);
    node.classList.toggle("current", stage.active);
    node.classList.toggle("paused", stage.paused);
    node.querySelector(".stage-status").textContent = stage.label;
    node.setAttribute("aria-label", `${node.querySelector(".stage-title").textContent}: ${stage.label}`);
    if (stage.active) node.setAttribute("aria-current", "step");
    else node.removeAttribute("aria-current");
  });
}

function displaySources(payload = appState.runPayload) {
  const results = payload?.result?.snapshot?.source_results || [];
  if (results.length) return results;
  if (Array.isArray(payload?.sources)) return payload.sources;
  // Older running backends already expose the selected discovery candidate.
  const candidate = (payload?.source_discovery?.candidates || [])
    .find((item) => item.candidate_id === payload?.state?.selected_candidate_id);
  if (!candidate) return [];
  return [{
    display_name: candidate.title || candidate.url, modality: candidate.modality, is_preview: true,
    manifest: { origin_url: candidate.url, resolved_version: candidate.document_version,
      source_origin: "websearch", source_priority: "user_selected", source_metadata: candidate },
  }];
}

function sourcePreviewStatus(source, state = appState.runPayload?.state || {}) {
  if (!source.is_preview) return "";
  if (["failed", "excluded"].includes(source.status)) return source.status;
  if (["failed", "cancelled", "cancelling", "paused"].includes(state.status)) return state.status;
  const phase = state.phase || state.stage;
  if (["acquiring", "acquisition"].includes(phase)) return "acquiring";
  if (["normalizing", "modal_preprocessing", "source_review", "source_validation", "version_alignment", "corpus_snapshot"].includes(phase)) return "processing";
  return "selected";
}

function renderSources() {
  const sources = displaySources();
  const tbody = $("#source-table-body");
  $("#source-waiting").hidden = sources.length > 0;
  tbody.innerHTML = sources.map((source) => {
    const manifest = source.manifest || {};
    const sourceUrl = safeSourceUrl(manifest.origin_url) || safeSourceUrl(manifest.final_url);
    const relation = manifest.source_metadata?.version_relation || source.evidence?.[0]?.locator?.version_relation || versionRelation(manifest.resolved_version);
    const applicability = manifest.source_metadata?.evidence_applicability || source.evidence?.[0]?.locator?.evidence_applicability || "";
    const previewStatus = sourcePreviewStatus(source);
    const previewLabel = ({ selected: "Source selected", acquiring: "Acquiring source material", processing: "Processing source material" })[previewStatus];
    const previewBadge = previewStatus ? `<div class="provenance-badges"><span class="status-badge ${statusClass(previewStatus === "acquiring" ? "processing" : previewStatus)}">${escapeHtml(previewLabel ? t(previewLabel) : statusLabel(previewStatus))}</span></div>` : "";
    return `
      <tr>
        <td><div class="source-title"><strong>${escapeHtml(source.display_name)}</strong>${source.source_id ? `<small>${escapeHtml(source.source_id)}</small>` : ""}${sourceUrl ? `<a class="source-origin-link" href="${escapeHtml(sourceUrl)}" target="_blank" rel="noopener noreferrer">${escapeHtml(sourceUrl)}</a>` : ""}${previewBadge}</div></td>
        <td>${escapeHtml(source.modality || t("unknown"))}</td>
        <td>${escapeHtml(manifest.resolved_version || t("unknown"))}${manifest.version_conflict ? `<br><small class="confidence">${t("Version conflict")}</small>` : ""}<div class="provenance-badges">${renderVersionBadge(relation, applicability)}</div></td>
        <td>${renderSourceBadge(manifest.source_origin, manifest.source_priority)}</td>
      </tr>
    `;
  }).join("");
}

function renderEvidence() {
  const snapshot = appState.runPayload?.result?.snapshot || {};
  const trees = getEvidenceTrees(snapshot);
  const evidence = snapshot.evidence || [];
  $("#evidence-tree-count").textContent = trees.reduce((total, tree) => total + (tree.nodes || []).length, 0);
  $("#evidence-flat-count").textContent = evidence.length;
  const treeMode = appState.evidenceView === "tree";
  $("#evidence-tree-list").hidden = !treeMode;
  $("#evidence-list").hidden = treeMode;
  $("#evidence-filter").placeholder = treeMode
    ? t("Filter tree nodes, content, or sources")
    : t("Filter excerpts, titles, or sources");
  const loading = shouldShowResultSkeleton("evidence");
  $("#tab-evidence")?.setAttribute?.("aria-busy", String(loading));
  if (loading) {
    const markup = renderResultSkeleton("evidence");
    if (treeMode) updateEvidenceTreeList(markup);
    else updateResultListMarkup($("#evidence-list"), markup);
    return;
  }
  resultSkeletonMarkup.delete($("#evidence-list"));
  if (treeMode) {
    renderEvidenceTrees(snapshot, trees);
  } else {
    renderFlatEvidence(snapshot);
  }
}

function renderFlatEvidence(snapshot) {
  const evidence = snapshot?.evidence || [];
  const sources = snapshot?.source_results || [];
  const sourceById = Object.fromEntries(sources.map((item) => [item.source_id, item]));
  const sourceOrder = Object.fromEntries(sources.map((item, index) => [item.source_id, index]));
  const evidenceOrder = new Map(evidence.map((item, index) => [item.evidence_id, index]));
  const query = $("#evidence-filter")?.value.trim().toLocaleLowerCase() || "";
  const modality = $("#evidence-modality")?.value || "all";
  const filtered = evidence.filter((item) => {
    const source = sourceById[item.source_id] || {};
    if (modality !== "all" && source.modality !== modality) return false;
    const imageSearch = (item.image_refs || []).map((image) => `${image.artifact_id} ${image.role}`).join(" ");
    const locatorSearch = JSON.stringify(item.locator || {});
    const haystack = `${item.title} ${item.content} ${source.display_name || ""} ${imageSearch} ${locatorSearch}`.toLocaleLowerCase();
    return !query || haystack.includes(query);
  }).sort((left, right) => {
    const sourceDifference = (sourceOrder[left.source_id] ?? Number.MAX_SAFE_INTEGER)
      - (sourceOrder[right.source_id] ?? Number.MAX_SAFE_INTEGER);
    if (sourceDifference) return sourceDifference;
    const leftPage = Number(left.locator?.page_start ?? left.locator?.page ?? Number.MAX_SAFE_INTEGER);
    const rightPage = Number(right.locator?.page_start ?? right.locator?.page ?? Number.MAX_SAFE_INTEGER);
    return leftPage - rightPage
      || (evidenceOrder.get(left.evidence_id) ?? 0) - (evidenceOrder.get(right.evidence_id) ?? 0);
  });
  const list = $("#evidence-list");
  if (!filtered.length) {
    list.innerHTML = `<div class="empty-list">${t(evidence.length ? "No matching Evidence" : "No Evidence yet")}</div>`;
    return;
  }
  list.innerHTML = filtered.map((item) => {
    const source = sourceById[item.source_id] || {};
    const locator = formatLocator(item.locator);
    const sectionPath = item.section_path || item.locator?.section_path || "";
    const hierarchy = sectionPath && sectionPath !== item.title
      ? `<div class="evidence-path">${escapeHtml(sectionPath)}</div>`
      : "";
    const imageCount = (item.image_refs || []).length;
    const evidenceLabel = item.content && imageCount
      ? t("Text + images")
      : imageCount ? t("Image Evidence") : t("Exact text");
    const provenance = renderEvidenceProvenance(item);
    const images = (item.image_refs || []).map((image) => {
      const imageUrl = `/api/runs/${encodeURIComponent(appState.currentRunId)}/artifacts/${encodeURIComponent(image.artifact_id)}`;
      return `
        <figure class="evidence-image">
          <a href="${imageUrl}" target="_blank" rel="noopener">
            <img src="${imageUrl}" loading="lazy" alt="${escapeHtml(t("Evidence image on page {page}", { page: image.page }))}">
          </a>
          <figcaption>${escapeHtml(image.artifact_id)} · ${escapeHtml(t("Page {page}", { page: image.page }))} · ${escapeHtml(image.role)}</figcaption>
        </figure>
      `;
    }).join("");
    return `
      <article class="evidence-item">
        <div class="evidence-top">
          <div><h3>${escapeHtml(item.title)}</h3>${hierarchy}<div class="evidence-meta">${escapeHtml(source.display_name || item.source_id)} · ${escapeHtml(locator)} · ${escapeHtml(item.extraction_method)}</div>${provenance}</div>
          <span class="confidence">${evidenceLabel}</span>
        </div>
        ${item.content ? `<p class="evidence-content">${escapeHtml(item.content)}</p>` : ""}
        ${images ? `<div class="evidence-images">${images}</div>` : ""}
      </article>
    `;
  }).join("");
}

function getEvidenceTrees(snapshot) {
  if (Array.isArray(snapshot?.evidence_trees) && snapshot.evidence_trees.length) {
    return snapshot.evidence_trees;
  }
  return (snapshot?.source_results || [])
    .map((source) => source.evidence_tree)
    .filter(Boolean);
}

function renderEvidenceTrees(snapshot, trees) {
  const sources = snapshot?.source_results || [];
  const sourceById = Object.fromEntries(sources.map((source) => [source.source_id, source]));
  const query = $("#evidence-filter")?.value.trim().toLocaleLowerCase() || "";
  const modality = $("#evidence-modality")?.value || "all";
  const rendered = [];

  trees.forEach((tree) => {
    const source = sourceById[tree.source_id] || {};
    if (modality !== "all" && source.modality !== modality) return;
    const nodes = tree.nodes || [];
    const nodeById = Object.fromEntries(nodes.map((node) => [node.evidence_node_id, node]));
    const sourceMatches = `${source.display_name || ""} ${tree.source_id || ""}`
      .toLocaleLowerCase()
      .includes(query);
    const matchingIds = new Set();
    if (!query || sourceMatches) {
      nodes.forEach((node) => matchingIds.add(node.evidence_node_id));
    } else {
      nodes.forEach((node) => {
        const locators = (node.source_refs || []).map((ref) => JSON.stringify(ref.locator || {})).join(" ");
        const haystack = `${node.title || ""} ${node.node_key || ""} ${node.node_type || ""} ${node.content || ""} ${locators}`
          .toLocaleLowerCase();
        if (haystack.includes(query)) matchingIds.add(node.evidence_node_id);
      });
    }
    if (!matchingIds.size) return;

    const visibleIds = new Set(matchingIds);
    matchingIds.forEach((nodeId) => {
      let current = nodeById[nodeId];
      while (current?.parent_node_id && !visibleIds.has(current.parent_node_id)) {
        visibleIds.add(current.parent_node_id);
        current = nodeById[current.parent_node_id];
      }
    });
    const root = nodeById[tree.root_node_id];
    if (!root || !visibleIds.has(root.evidence_node_id)) return;
    rendered.push(`
      <section class="evidence-tree-source">
        <header class="evidence-tree-source-header">
          <div>
            <h3>${escapeHtml(source.display_name || tree.source_id || t("Evidence source"))}</h3>
            <p>${escapeHtml(tree.source_id || "")} · ${escapeHtml(source.modality || t("unknown"))}</p>
          </div>
          <div class="evidence-tree-summary">
            <span>${t("{count} nodes", { count: nodes.length })}</span>
            <span>${t("{count} leaves", { count: tree.leaf_count || 0 })}</span>
            <span>${t("{count} SourceParts", { count: tree.referenced_part_count || 0 })}</span>
          </div>
        </header>
        <div class="evidence-tree">
          ${renderEvidenceTreeNode(root, nodeById, visibleIds, 0, query, tree.source_id)}
        </div>
      </section>
    `);
  });

  updateEvidenceTreeList(rendered.join("") || `<div class="empty-list">${t(trees.length ? "No matching EvidenceTree nodes" : "No EvidenceTree yet; use Flat Evidence for legacy runs")}</div>`);
}

function updateEvidenceTreeList(markup) {
  const list = $("#evidence-tree-list");
  if (appState.evidenceTreeRunId !== appState.currentRunId) {
    appState.evidenceTreeRunId = appState.currentRunId;
    appState.evidenceTreeMarkup = null;
    appState.evidenceTreeOpenState.clear();
  } else {
    list.querySelectorAll("details[data-evidence-detail-key]").forEach((detail) => {
      appState.evidenceTreeOpenState.set(detail.dataset.evidenceDetailKey, detail.open);
    });
  }
  // Polling unchanged evidence must not replace the nodes the user is reading.
  if (appState.evidenceTreeMarkup === markup) return;
  list.innerHTML = markup;
  list.querySelectorAll("details[data-evidence-detail-key]").forEach((detail) => {
    if (appState.evidenceTreeOpenState.has(detail.dataset.evidenceDetailKey)) {
      detail.open = appState.evidenceTreeOpenState.get(detail.dataset.evidenceDetailKey);
    }
  });
  appState.evidenceTreeMarkup = markup;
}

function renderEvidenceTreeNode(node, nodeById, visibleIds, depth, query, sourceId) {
  const children = (node.child_node_ids || [])
    .map((nodeId) => nodeById[nodeId])
    .filter((child) => child && visibleIds.has(child.evidence_node_id));
  const refs = node.source_refs || [];
  const open = Boolean(query) || depth <= 1;
  // Search expansion is temporary; clearing the query restores the ordinary view.
  const detailKey = JSON.stringify([sourceId, node.evidence_node_id, query]);
  const contentBlock = node.content ? `
    <details class="evidence-node-material" data-evidence-detail-key="${escapeHtml(`${detailKey}:material`)}">
      <summary>${t("Exact content · {count} SourcePart(s)", { count: refs.length })}</summary>
      <pre>${escapeHtml(node.content)}</pre>
      ${renderEvidenceTreeRefs(refs, detailKey)}
    </details>
  ` : "";
  return `
    <details class="evidence-tree-node depth-${Math.min(depth, 4)}" data-node-key="${escapeHtml(node.node_key || "")}" data-evidence-detail-key="${escapeHtml(`${detailKey}:node`)}" ${open ? "open" : ""}>
      <summary>
        <span class="evidence-tree-node-title">${escapeHtml(node.title || node.node_key || t("Untitled node"))}</span>
        <span class="evidence-tree-node-facts">
          <span>${escapeHtml(node.node_type || "other")}</span>
          <span>${t("{count} children", { count: children.length })}</span>
          <span>${t("{count} refs", { count: refs.length })}</span>
          <span>${t("{count} chars", { count: Number(node.direct_char_count || 0).toLocaleString() })}</span>
        </span>
      </summary>
      <div class="evidence-tree-node-body">
        ${contentBlock}
        ${children.length ? `<div class="evidence-tree-children">${children.map((child) => (
          renderEvidenceTreeNode(child, nodeById, visibleIds, depth + 1, query, sourceId)
        )).join("")}</div>` : ""}
      </div>
    </details>
  `;
}

function renderEvidenceTreeRefs(refs, detailKey) {
  if (!refs.length) return "";
  return `
    <details class="evidence-node-refs" data-evidence-detail-key="${escapeHtml(`${detailKey}:refs`)}">
      <summary>${t("Source provenance ({count})", { count: refs.length })}</summary>
      <ol>${refs.map((ref) => `
        <li>
          <code>${escapeHtml(ref.source_part_id || "unknown")}</code>
          <span>${Number(ref.start_offset || 0).toLocaleString()}–${Number(ref.end_offset || 0).toLocaleString()}</span>
          <span>${escapeHtml(formatLocator(ref.locator || {}))}</span>
        </li>
      `).join("")}</ol>
    </details>
  `;
}

function renderReferences() {
  const result = appState.runPayload?.result || {};
  const references = result.references || [];
  $("#reference-count").textContent = references.length;
  const loading = shouldShowResultSkeleton("reference");
  $("#tab-reference")?.setAttribute?.("aria-busy", String(loading));
  if (loading) {
    updateResultListMarkup($("#reference-list"), renderResultSkeleton("reference"));
    return;
  }
  resultSkeletonMarkup.delete($("#reference-list"));
  renderReferenceListWithDetails(references);
}

function renderTasksAndTrajectories() {
  const result = appState.runPayload?.result || {};
  const tasks = result.tasks || [];
  const trajectories = result.trajectories || [];
  $("#task-count").textContent = tasks.length;
  $("#trajectory-count").textContent = trajectories.length;
  const tasksLoading = shouldShowResultSkeleton("tasks");
  const trajectoriesLoading = shouldShowResultSkeleton("trajectories");
  $("#tab-task-trajectory")?.setAttribute?.("aria-busy", String(tasksLoading || trajectoriesLoading));
  const taskList = $("#task-list");
  const trajectoryList = $("#trajectory-list");
  taskList.setAttribute("aria-busy", String(tasksLoading));
  trajectoryList.setAttribute("aria-busy", String(trajectoriesLoading));
  if (tasksLoading) {
    updateResultListMarkup(taskList, renderResultSkeleton("tasks"));
  } else {
    resultSkeletonMarkup.delete(taskList);
    renderTaskList(tasks);
  }
  if (trajectoriesLoading) {
    updateResultListMarkup(trajectoryList, renderResultSkeleton("trajectories"));
  } else {
    resultSkeletonMarkup.delete(trajectoryList);
    renderTrajectoryList(trajectories, tasks);
  }
}

function renderReferenceList(references) {
  const list = $("#reference-list");
  if (!references.length) {
    list.innerHTML = `<div class="empty-list compact-empty">${t("No submitted artifacts yet")}</div>`;
    return;
  }
  list.innerHTML = references.map((item) => {
    const entries = item.entries || [];
    const formatted = Boolean(referencePresentationText(item));
    const isCatalog = item.reference_mode === "source_catalog";
    const policy = item.injection_policy || "on_demand";
    const label = formatted ? t("Formatted reference") : isCatalog ? t("Source-preserved catalog · {count} entries", { count: entries.length }) : (item.category || "reference");
    const previews = entries.slice(0, 5).map((entry) => referenceEntryTitle(entry)).filter(Boolean);
    const remaining = entries.length - previews.length;
    return `
      <article class="knowledge-item reference-item ${isCatalog ? "reference-catalog" : ""}">
        <div class="knowledge-item-top"><h4>${escapeHtml(referenceTextOnly(item.title) || "Reference")}</h4><span class="confidence">${escapeHtml(label)}</span></div>
        ${renderReferenceText(item)}
        ${isCatalog && !formatted ? `<div class="reference-entry-index"><span>${escapeHtml(t("Injection policy: {policy}", { policy }))}</span>${previews.map((title) => `<span>${escapeHtml(title)}</span>`).join("")}${remaining > 0 ? `<span>${t("+{count} entries", { count: remaining })}</span>` : ""}</div>` : ""}
        <div class="knowledge-item-meta"><span>${t("{count} citations", { count: (item.citations || []).length })}</span></div>
        ${renderCitationBadges(item.citations)}
      </article>
    `;
  }).join("");
}

function renderReferenceListWithDetails(references) {
  const list = $("#reference-list");
  if (!references.length) {
    list.innerHTML = `<div class="empty-list compact-empty">${t("No submitted artifacts yet")}</div>`;
    return;
  }
  list.innerHTML = references.map((item, index) => {
    const entries = item.entries || [];
    const formatted = Boolean(referencePresentationText(item));
    const isCatalog = item.reference_mode === "source_catalog";
    const policy = item.injection_policy || "on_demand";
    const label = formatted ? t("Formatted reference") : isCatalog ? t("Source-preserved catalog · {count} entries", { count: entries.length }) : (item.category || "reference");
    const previews = entries.slice(0, 5).map((entry) => referenceEntryTitle(entry)).filter(Boolean);
    const remaining = entries.length - previews.length;
    const catalogId = item.reference_id || `catalog-${index}`;
    return `
      <article class="knowledge-item reference-item ${isCatalog ? "reference-catalog" : ""}">
        <div class="knowledge-item-top"><h4>${escapeHtml(referenceTextOnly(item.title) || "Reference")}</h4><span class="confidence">${escapeHtml(label)}</span></div>
        ${renderReferenceText(item)}
        ${isCatalog || formatted ? `
          ${!formatted ? `<div class="reference-entry-index"><span>${escapeHtml(t("Injection policy: {policy}", { policy }))}</span>${previews.map((title) => `<span>${escapeHtml(title)}</span>`).join("")}${remaining > 0 ? `<span>${t("+{count} entries", { count: remaining })}</span>` : ""}</div>` : ""}
          <div class="reference-catalog-actions">
            <button class="reference-catalog-open" type="button" data-reference-catalog-id="${escapeHtml(catalogId)}" aria-label="${escapeHtml(t("View all entries in {title}", { title: referenceTextOnly(item.title) || "Reference" }))}">
              <i data-lucide="expand" aria-hidden="true"></i><span>${t("View Full Reference")}</span>
            </button>
          </div>
        ` : ""}
        <div class="knowledge-item-meta"><span>${t("{count} citations", { count: (item.citations || []).length })}</span></div>
        ${renderCitationBadges(item.citations)}
      </article>
    `;
  }).join("");

  list.querySelectorAll("[data-reference-catalog-id]").forEach((button) => {
    button.addEventListener("click", () => {
      const referenceId = button.dataset.referenceCatalogId;
      const reference = references.find((item, index) => (
        (item.reference_id || `catalog-${index}`) === referenceId
      ));
      if (reference) openReferenceCatalog(reference);
    });
  });
  refreshIcons();
}

function referenceDescription(reference) {
  return referenceTextOnly(reference.description || reference.content || "");
}

function referencePresentationText(reference) {
  const presentation = reference.presentation;
  return presentation?.mode === "agent_formatted" && typeof presentation.content === "string"
    ? presentation.content : "";
}

function renderReferenceText(reference) {
  const description = referenceDescription(reference);
  if (referencePresentationText(reference)) return `<p>${escapeHtml(description)}</p>`;
  const content = referenceTextOnly(reference.content || "");
  return `<p>${escapeHtml(description)}</p>${(content && content !== description)
    ? `<div class="reference-knowledge-content">${renderReferenceContent(content)}</div>`
    : ""}`;
}

function referenceTextOnly(text) {
  let value = String(text || "");
  const labels = new Set(Array.from(value.matchAll(/!\[([^\]\n]*)\]\[([^\]\n]*)\]/g), (match) => match[2] || match[1]));
  value = value.replace(/!\[[^\]\n]*\]\[[^\]\n]*\]/g, "");
  value = value.split("\n").filter((line) => {
    const definition = line.match(/^ {0,3}\[([^\]]+)\]:/);
    return !definition || !labels.has(definition[1]);
  }).join("\n");
  return value.replace(/!\[[^\]\n]*\]\((?:\\.|[^()\\]|\([^()]*\))*\)/g, "")
    .replace(/<img\b[^>]*>/gi, "").replace(/\[image\s+[^\]\n]+\]/gi, "");
}

function referenceEntryTitle(entry) {
  return referenceTextOnly(entry.display_title || entry.title || "").trim()
    || referenceTextOnly(entry.content || "").split(/(?<!\\)\|/).find((cell) => cell.trim())?.trim()
    || "";
}

function referenceTableCells(line) {
  if (!/^\s*\|.*\|\s*$/.test(line)) return null;
  const cells = [];
  let cell = "", code = false, escaped = false;
  for (const char of line.trim().slice(1, -1)) {
    if (char === "`" && !escaped) code = !code;
    if (char === "|" && !escaped && !code) { cells.push(cell.trim()); cell = ""; }
    else cell += char;
    escaped = char === "\\" && !escaped;
  }
  cells.push(cell.trim());
  return cells;
}

function renderReferenceContent(text) {
  const inline = escapeHtml;
  const lines = referenceTextOnly(text).split("\n");
  const html = [];
  for (let index = 0; index < lines.length;) {
    if (!referenceTableCells(lines[index])) { html.push(inline(lines[index++])); continue; }
    const rows = [];
    while (index < lines.length && referenceTableCells(lines[index])) rows.push(referenceTableCells(lines[index++]));
    const hasHeader = rows.length > 1 && rows[1].every((cell) => /^:?-+:?$/.test(cell));
    html.push(`<div class="reference-table-wrap"><table class="reference-table">${rows.map((cells, row) => {
      if (hasHeader && row === 1) return "";
      const tag = hasHeader && row === 0 ? "th" : "td";
      return `<tr>${cells.map((cell) => `<${tag}>${inline(cell)}</${tag}>`).join("")}</tr>`;
    }).join("")}</table></div>`);
  }
  return html.join("\n");
}

function renderReferencePresentation(text) {
  // This intentionally supports a small Markdown subset and never accepts HTML,
  // link destinations, style attributes or image elements from model output.
  const lines = String(text || "").split(/\r?\n/);
  const decodeCellEntities = value => value.replace(/&(amp|lt|gt);/g,
    (_, entity) => ({amp: "&", lt: "<", gt: ">"})[entity]);
  const inline = (value, tableCell = false) => {
    const parts = [];
    let plain = "", index = 0;
    const escape = text => escapeHtml(tableCell ? decodeCellEntities(text) : text);
    while (index < value.length) {
      const char = value[index];
      if (tableCell && char === "\\" && /[\\|`]/.test(value[index + 1] || "")) {
        plain += value[index + 1]; index += 2; continue;
      }
      if (char === "`") {
        let end = index + 1;
        while (value[end] === "`") end++;
        const marker = value.slice(index, end);
        const close = value.indexOf(marker, end);
        if (close > end) {
          parts.push(escape(plain)); plain = "";
          const code = value.slice(end, close);
          parts.push(`<code>${escape(tableCell ? code.replace(/\\([\\|`])/g, "$1") : code)}</code>`);
          index = close + marker.length; continue;
        }
      }
      plain += char; index++;
    }
    parts.push(escape(plain));
    return parts.join("");
  };
  const tableInline = value => value.split("<br>").map(piece => inline(piece, true)).join("<br>");
  const listItem = line => {
    const match = line.match(/^([ \t]*)(?:([-+*])|(-?\d+)[.)])(?:[ \t]+(.*))?$/);
    return match ? {
      indent: match[1].replace(/\t/g, "    ").length,
      ordered: match[3] !== undefined,
      ordinal: match[3], text: match[4] || "",
    } : null;
  };
  const renderList = start => {
    // Keep a stack rather than recursive calls so deeply nested source lists
    // cannot exhaust the browser's JavaScript call stack.
    const parts = [], levels = [];
    const closeLevel = () => {
      const level = levels.pop();
      if (level.itemOpen) parts.push("</li>");
      parts.push(level.ordered ? "</ol>" : "</ul>");
    };
    const openLevel = item => {
      const ordinal = item.ordered ? BigInt(item.ordinal) : null;
      parts.push(item.ordered ? `<ol${ordinal !== 1n ? ` start="${ordinal}"` : ""}>` : "<ul>");
      levels.push({indent: item.indent, ordered: item.ordered, nextOrdinal: ordinal, itemOpen: false});
    };
    let index = start;
    while (index < lines.length) {
      const item = listItem(lines[index]);
      if (!item) break;
      while (levels.length && item.indent < levels[levels.length - 1].indent) closeLevel();
      let level = levels[levels.length - 1];
      if (level && item.indent === level.indent && item.ordered !== level.ordered) {
        closeLevel(); level = levels[levels.length - 1];
      }
      if (!level || item.indent > level.indent) openLevel(item);
      level = levels[levels.length - 1];
      if (level.itemOpen) parts.push("</li>");
      let value = "";
      if (item.ordered) {
        const ordinal = BigInt(item.ordinal);
        if (ordinal !== level.nextOrdinal) value = ` value="${ordinal}"`;
        level.nextOrdinal = ordinal + 1n;
      }
      parts.push(`<li${value}>${inline(item.text)}`);
      level.itemOpen = true;
      index++;
    }
    while (levels.length) closeLevel();
    return {html: parts.join(""), next: index};
  };
  const html = [];
  for (let index = 0; index < lines.length;) {
    const line = lines[index];
    if (!line.trim()) { index++; continue; }
    const fence = line.match(/^ {0,3}(`{3,}|~{3,})[^`~]*$/);
    if (fence) {
      const code = [];
      const close = new RegExp(`^ {0,3}${fence[1][0]}{${fence[1].length},}\\s*$`);
      index++;
      while (index < lines.length && !close.test(lines[index])) code.push(lines[index++]);
      if (index < lines.length) index++;
      html.push(`<pre><code>${escapeHtml(code.join("\n"))}</code></pre>`);
      continue;
    }
    const heading = line.match(/^(#{1,6})\s+(.+)$/);
    if (heading) {
      const level = Math.min(6, heading[1].length + 1);
      html.push(`<h${level}>${inline(heading[2])}</h${level}>`);
      index++; continue;
    }
    const headerCells = referenceTableCells(line);
    const separatorCells = referenceTableCells(lines[index + 1] || "");
    if (headerCells && separatorCells && headerCells.length === separatorCells.length
        && separatorCells.every(cell => /^:?-+:?$/.test(cell))) {
      const rows = [];
      while (index < lines.length && referenceTableCells(lines[index])) rows.push(referenceTableCells(lines[index++]));
      const hasHeader = rows.length > 1 && rows[1].every(cell => /^:?-+:?$/.test(cell));
      html.push(`<div class="reference-table-wrap"><table class="reference-table">${rows.map((cells, row) => {
        if (hasHeader && row === 1) return "";
        const tag = hasHeader && row === 0 ? "th" : "td";
        return `<tr>${cells.map(cell => `<${tag}>${tableInline(cell)}</${tag}>`).join("")}</tr>`;
      }).join("")}</table></div>`);
      continue;
    }
    if (listItem(line)) {
      const list = renderList(index);
      html.push(list.html); index = list.next; continue;
    }
    html.push(`<p>${inline(line)}</p>`); index++;
  }
  return html.join("\n");
}

function openReferenceCatalog(reference) {
  const dialog = $("#reference-catalog-dialog");
  const entries = reference.entries || [];
  appState.openReferenceCatalog = reference;
  $("#reference-catalog-dialog-title").textContent = referenceTextOnly(reference.title) || "Reference";
  $("#reference-catalog-dialog-description").textContent = [
    referenceDescription(reference) || t("Source-preserved entries."),
    ...(!referencePresentationText(reference) ? [
      t("Injection policy: {policy}", { policy: reference.injection_policy || "on_demand" }),
      t("{count} complete entries", { count: entries.length }),
    ] : []),
  ].join(" · ");
  $("#reference-catalog-filter").value = "";
  renderReferenceCatalogEntries(reference, "");
  if (typeof dialog.showModal === "function") {
    if (!dialog.open) dialog.showModal();
  } else {
    dialog.setAttribute("open", "");
  }
  refreshIcons();
}

function renderReferenceCatalogEntries(reference, query) {
  const entries = reference.entries || [];
  const normalizedQuery = String(query || "").trim().toLocaleLowerCase();
  const presentation = referencePresentationText(reference);
  const filtered = entries.filter((entry) => {
    const haystack = [
      entry.title,
      entry.content,
      entry.source_evidence_id,
      formatLocator(entry.locator),
    ].join(" ").toLocaleLowerCase();
    return !normalizedQuery || haystack.includes(normalizedQuery);
  });
  $("#reference-catalog-result-count").textContent = presentation ? t("Formatted reference") : normalizedQuery
    ? t("Showing {count} / {total} entries", { count: filtered.length, total: entries.length })
    : t("{count} entries total", { count: entries.length });
  const list = $("#reference-catalog-entry-list");
  const content = referenceTextOnly(reference.content || "");
  const knowledgeContent = (content && content !== referenceDescription(reference))
    ? `<article class="reference-catalog-entry"><h3>${t("Knowledge content")}</h3><div class="reference-catalog-entry-content">${renderReferenceContent(content)}</div></article>`
    : "";
  if (!filtered.length && !presentation) {
    list.innerHTML = knowledgeContent + `<div class="reference-catalog-empty">${t("No matching full Reference entries.")}</div>`;
    return;
  }
  const originalEntries = filtered.map((entry, index) => {
    const citations = entry.citations || [];
    const locator = formatLocator(entry.locator);
    return `
      <article class="reference-catalog-entry">
        <div class="reference-catalog-entry-top">
          <div><span class="reference-catalog-entry-number">${index + 1}</span><h3>${escapeHtml(referenceEntryTitle(entry) || t("Untitled entry"))}</h3></div>
          <span class="reference-catalog-entry-locator">${escapeHtml(locator)}</span>
        </div>
        <div class="reference-catalog-entry-content">${renderReferenceContent(entry.content || "")}</div>
        <div class="reference-catalog-entry-meta">
          <span>${escapeHtml(t("Evidence: {id}", { id: entry.source_evidence_id || "unknown" }))}</span>
          <span>${t("{count} citations", { count: citations.length })}</span>
        </div>
        ${renderEvidenceBadgeById(entry.source_evidence_id)}
        ${citations.length ? `
          <details class="reference-catalog-citations">
            <summary>${t("View Citation")}</summary>
            <ul>${citations.map((citation) => `
              <li>
                <strong>${escapeHtml(citation.evidence_id || "Evidence")}</strong>
                <div class="reference-citation-quote">${renderReferenceContent(citation.quote || "")}</div>
              </li>
            `).join("")}</ul>
          </details>
        ` : ""}
      </article>
    `;
  }).join("");
  if (presentation) {
    const matches = !normalizedQuery || presentation.toLocaleLowerCase().includes(normalizedQuery);
    list.innerHTML = (matches
      ? `<article class="reference-catalog-entry reference-formatted-content">${renderReferencePresentation(presentation)}</article>`
      : `<div class="reference-catalog-empty">${t("No matching full Reference entries.")}</div>`)
      + `<details class="reference-source-details"><summary>${t("View source-preserved entries")}</summary>${knowledgeContent}${originalEntries}</details>`;
  } else list.innerHTML = knowledgeContent + originalEntries;
}

function renderTaskList(tasks) {
  const list = $("#task-list");
  if (!tasks.length) {
    list.innerHTML = `<div class="empty-list compact-empty">${t("No submitted artifacts yet")}</div>`;
    return;
  }
  list.innerHTML = tasks.map((item) => `
    <article class="knowledge-item">
      <div class="knowledge-item-top"><h4>${escapeHtml(item.task || item.title || "")}</h4><span class="confidence">${escapeHtml(item.category || "")}</span></div>
      <div class="knowledge-item-meta"><span>${t("{count} citations", { count: (item.citations || []).length })}</span></div>
      ${renderCitationBadges(item.citations)}
    </article>
  `).join("");
}

function renderTrajectoryList(items, tasks) {
  const list = $("#trajectory-list");
  const tasksById = new Map(tasks.map((task) => [task.task_id, task]));
  if (!items.length) {
    list.innerHTML = `<div class="empty-list compact-empty">${t("No submitted artifacts yet")}</div>`;
    return;
  }
  list.innerHTML = items.map((item) => `
    <article class="knowledge-item">
      <div class="knowledge-item-top"><h4>${escapeHtml(tasksById.get(item.task_id)?.task || item.task_id || item.trajectory_id)}</h4><span class="confidence">${t("{count} steps", { count: (item.steps || []).length })}</span></div>
      <div class="knowledge-item-meta"><span>${t("{count} citations", { count: (item.citations || []).length })}</span></div>
      ${renderCitationBadges(item.citations)}
      ${item.steps ? renderTrajectorySteps(item) : ""}
    </article>
  `).join("");
}

function renderTrajectorySteps(item) {
  return `<details class="trajectory-details"><summary>${t("View trajectory steps · {count} steps", { count: (item.steps || []).length })}</summary>
    <ol class="skill-procedure">${(item.steps || []).map((step) => `<li>
      <span>${escapeHtml(step.instruction)}</span>
      ${step.expected_observation ? `<span class="skill-observation">${t("Observation:")} ${escapeHtml(step.expected_observation)}</span>` : ""}
      ${renderCitationBadges(step.citations, 5)}
    </li>`).join("")}</ol>
  </details>`;
}

function renderSkills() {
  const sourceSkills = appState.runPayload?.result?.skills || [];
  const skills = getDisplayableSkills(sourceSkills);
  const loading = shouldShowResultSkeleton("skills");
  $("#tab-skills")?.setAttribute?.("aria-busy", String(loading));
  if (loading) {
    updateSkillsList(renderResultSkeleton("skills"));
    return;
  }
  if (!skills.length) {
    const message = sourceSkills.length
      ? "This run contains only legacy placeholder artifacts, which are not shown as Skills."
      : "No deliverable, source-grounded Skills are available yet.";
    updateSkillsList(`<div class="empty-list">${t(message)}</div>`);
    return;
  }
  updateSkillsList(skills.map((skill) => `
    <article class="skill-item">
      <details class="skill-details" data-skill-key="${escapeHtml(skill.skill_id || JSON.stringify([skill.title, skill.description, skill.trajectory_ids || []]))}">
        <summary class="skill-summary">
          <div class="skill-item-top"><div><h3>${escapeHtml(skill.title)}</h3><p>${escapeHtml(skill.description)}</p></div></div>
          <span class="skill-toggle" aria-hidden="true"></span>
        </summary>
        <div class="skill-details-body">
      <section class="skill-section skill-applicability"><h4>${t("Applicability")}</h4><p>${escapeHtml(skill.applicability)}</p></section>
      ${renderSkillPreconditions(skill.preconditions)}
      <section class="skill-section"><h4>${t("Procedure")}</h4><ol class="skill-procedure">${(skill.procedure || []).map((step) => `
        <li>
          <span>${escapeHtml(step.instruction)}</span>
          <small>${t("{count} citations", { count: (step.citations || []).length })}</small>
          ${renderCitationBadges(step.citations, 5)}
        </li>`).join("")}</ol></section>
      <div class="skill-item-meta"><span>${t("{count} trajectories", { count: (skill.trajectory_ids || []).length })}</span></div>
        </div>
      </details>
    </article>
  `).join(""));
}

function updateSkillsList(markup) {
  const list = $("#skills-list");
  if (appState.skillsRunId !== appState.currentRunId) {
    appState.skillsRunId = appState.currentRunId;
    appState.skillsMarkup = null;
    appState.skillOpenState.clear();
  } else {
    list.querySelectorAll("details[data-skill-key]").forEach((detail) => {
      appState.skillOpenState.set(detail.dataset.skillKey, getSkillMotionOpenState(detail));
    });
  }
  // Preserve both the user's expansion choices and unchanged DOM during polling.
  if (appState.skillsMarkup === markup) return;
  list.innerHTML = markup;
  list.querySelectorAll("details[data-skill-key]").forEach((detail) => {
    if (appState.skillOpenState.has(detail.dataset.skillKey)) {
      detail.open = appState.skillOpenState.get(detail.dataset.skillKey);
    }
  });
  appState.skillsMarkup = markup;
}

function getDisplayableSkills(skills) {
  return skills.filter((skill) => !isLegacyPlaceholderSkill(skill));
}

function renderSkillPreconditions(preconditions) {
  let text = "";
  if (typeof preconditions === "string") {
    text = preconditions;
  } else if (Array.isArray(preconditions)) {
    // Historical results stored requirements as objects with a criterion field.
    text = preconditions.map((item) => {
      if (typeof item === "string") return item;
      return typeof item?.criterion === "string" ? item.criterion : "";
    }).filter((item) => item.trim()).join("\n");
  }
  if (!text.trim()) return "";
  return `<section class="skill-section skill-preconditions"><h4>${t("Preconditions")}</h4><p>${escapeHtml(text).replace(/\n/g, "<br>")}</p></section>`;
}

function isLegacyPlaceholderSkill(skill) {
  const title = String(skill.title || "").trim();
  const body = `${skill.description || ""} ${skill.applicability || ""}`.toLocaleLowerCase();
  const isDocumentHeading = /^(?:chapter|section|appendix)\b|^\u7b2c\s*[\u4e00\u4e8c\u4e09\u56db\u4e94\u516d\u4e03\u516b\u4e5d\u5341\u767e\u53430-9]+\s*[\u7ae0\u8282\u7bc7]|^\d+(?:\.\d+)+(?:\s|$)/i.test(title);
  const isDocumentDescriptor = /\b(?:manual|handbook|documentation|introduction|acknowledg(?:e)?ments?|license|contents)\b|\u624b\u518c|\u7ffb\u8bd1|\u7b80\u4ecb|\u81f4\u8c22|\u8bb8\u53ef|\u76ee\u5f55/i.test(title);
  const isGeneric = [
    "use the documented operation or concept in the cited source material",
    "use when the cited source material applies",
    "follow the documented operation identified by the cited evidence",
  ].some((fragment) => body.includes(fragment));
  return isDocumentHeading || isDocumentDescriptor || isGeneric;
}

function renderAudit() {
  const list = $("#audit-list");
  const filtered = appState.events.filter((event) => {
    if (appState.auditFilter === "all") return true;
    if (appState.auditFilter === "error") return event.level === "error" || event.level === "warning";
    return event.event_type.startsWith(`${appState.auditFilter}_`) || event.event_type.includes(appState.auditFilter);
  });
  list.innerHTML = filtered.map((event) => {
    const group = event.event_type.includes("tool_") ? "tool" : event.event_type.includes("model_") ? "model" : "";
    const summary = auditSummary(event);
    return `
      <div class="audit-event ${group} ${event.level === "error" ? "error" : ""}">
        <span class="audit-time">${formatTime(event.timestamp)}</span>
        <span class="audit-type">${escapeHtml(event.event_type)}</span>
        <details class="audit-payload"><summary>${escapeHtml(summary)}</summary><pre>${escapeHtml(JSON.stringify(event.payload || {}, null, 2))}</pre>${renderValidationReportLink(event)}</details>
      </div>
    `;
  }).join("") || `<div class="empty-list">${t("No matching events")}</div>`;
  if ($("#audit-follow")?.checked) list.scrollTop = list.scrollHeight;
}

function auditSummary(event) {
  const payload = event.payload || {};
  const identity = event.agent || event.source_id || "run";
  if (event.event_type === "source_validation_completed") {
    return `${identity} · ${t("Evidence: {accepted}/{total} accepted; {rejected} rejected", {
      accepted: payload.evidence_count, total: payload.input_evidence_count,
      rejected: payload.rejected_evidence_count,
    })}`;
  }
  if (event.event_type === "model_waiting") {
    return `${identity} · ${t("Model session waiting: {seconds}s elapsed, {idle}s since last activity", {
      seconds: Math.round(payload.elapsed_seconds || 0),
      idle: Math.round(payload.seconds_since_event || 0),
    })}`;
  }
  if (event.event_type === "model_retry_scheduled") {
    return `${identity} · ${t("Model retry {retry}/{max} in {seconds}s: {reason}", {
      retry: payload.retry_number, max: payload.max_retries ?? "?",
      seconds: payload.delay_seconds, reason: auditModelMessage(payload.message),
    })}`;
  }
  if (event.event_type === "model_retry_started") {
    return `${identity} · ${t("Model retry {retry} started", {retry: payload.retry_number})}`;
  }
  if (event.event_type === "model_retry_stopped") {
    return `${identity} · ${t("Model retry stopped: {reason}", {reason: auditModelMessage(payload.message)})}`;
  }
  if (event.event_type.startsWith("model_") && payload.message) {
    return `${identity} · ${auditModelMessage(payload.message)}`;
  }
  if (payload.tool) {
    const description = payload.arguments?.description || payload.arguments?.command;
    return [identity, payload.tool, payload.status, description?.slice(0, 100)].filter(Boolean).join(" · ");
  }
  if (payload.status) return `${identity} · ${payload.status}`;
  if (payload.message) return `${identity} · ${payload.message}`;
  if (payload.modality) return `${identity} · ${payload.modality}`;
  return identity;
}

function auditModelMessage(value) {
  return String(typeof value === "string" ? value : value?.preview || "").slice(0, 500);
}

function renderValidationReportLink(event) {
  const artifactId = event.payload?.validation_artifact_id;
  const runId = event.run_id || appState.currentRunId;
  if (event.event_type !== "source_validation_completed" || !runId ||
      typeof artifactId !== "string" || !/^art_[0-9a-f]{24}$/.test(artifactId)) return "";
  const url = `/api/runs/${encodeURIComponent(runId)}/artifacts/${encodeURIComponent(artifactId)}`;
  return `<a class="audit-artifact" href="${escapeHtml(url)}" target="_blank" rel="noopener">${t("View validation report")}</a>`;
}

function getEvidenceById() {
  const evidence = appState.runPayload?.result?.snapshot?.evidence || [];
  return Object.fromEntries(evidence.map((item) => [item.evidence_id, item]));
}

function renderCitationBadges(citations = [], limit = 4) {
  const evidenceById = getEvidenceById();
  const badges = citations
    .slice(0, limit)
    .map((citation) => renderEvidenceBadge(evidenceById[citation.evidence_id], citation.evidence_id))
    .join("");
  const remaining = citations.length - limit;
  if (!badges && !remaining) return "";
  return `<div class="citation-badges">${badges}${remaining > 0 ? `<span class="provenance-badge muted">+${remaining}</span>` : ""}</div>`;
}

function renderEvidenceBadgeById(evidenceId) {
  if (!evidenceId) return "";
  const evidence = getEvidenceById()[evidenceId];
  return `<div class="citation-badges">${renderEvidenceBadge(evidence, evidenceId)}</div>`;
}

function renderEvidenceBadge(evidence, fallbackId = "Evidence") {
  if (!evidence) return `<span class="provenance-badge muted">${escapeHtml(fallbackId)}</span>`;
  const locator = evidence.locator || {};
  const origin = sourceOriginLabel(locator.source_origin || "user");
  const version = locator.source_resolved_version || evidence.resolved_version || "unknown";
  const relation = relationLabel(locator.version_relation);
  const applicability = applicabilityLabel(locator.evidence_applicability);
  const text = [fallbackId, origin, version, relation || applicability].filter(Boolean).join(" · ");
  const cssClass = provenanceClass(locator.source_origin, locator.evidence_applicability, locator.version_relation);
  return `<span class="provenance-badge ${cssClass}" title="${escapeHtml(text)}">${escapeHtml(text)}</span>`;
}

function renderEvidenceProvenance(evidence) {
  const locator = evidence.locator || {};
  if (!locator.source_origin && !locator.version_relation && !locator.evidence_applicability) return "";
  return `<div class="provenance-badges">${renderSourceBadge(locator.source_origin, locator.source_priority)}${renderVersionBadge(locator.version_relation, locator.evidence_applicability, locator.source_resolved_version)}</div>`;
}

function renderSourceBadge(origin = "user", priority = "") {
  const label = sourceOriginLabel(origin);
  const priorityLabel = priority ? ` · ${sourcePriorityLabel(priority)}` : "";
  const cssClass = origin === "websearch" ? "web" : "user";
  return `<span class="provenance-badge ${cssClass}">${escapeHtml(label + priorityLabel)}</span>`;
}

function renderVersionBadge(relation = "", applicability = "", version = "") {
  if (!relation && !applicability && !version) return "";
  const label = [version, relationLabel(relation), applicabilityLabel(applicability)].filter(Boolean).join(" · ");
  const cssClass = provenanceClass("", applicability, relation);
  return `<span class="provenance-badge ${cssClass}">${escapeHtml(label || t("Version information"))}</span>`;
}

function sourceOriginLabel(origin) {
  return t(origin === "websearch" ? "Online source" : "User upload");
}

function sourcePriorityLabel(priority) {
  return t(({
    official_docs: "Official documentation",
    release_notes: "Release notes",
    github_repository: "GitHub",
    github_or_wiki: "GitHub",
    github_release: "GitHub release",
    github_tag: "GitHub tag",
    github_docs: "GitHub documentation",
    manual_or_help: "Help / manual",
    tutorial: "Tutorial",
    community: "Community",
    unclassified: "Unclassified",
  })[priority] || priority || "");
}

function relationLabel(relation) {
  return t(({
    exact: "Target version",
    exact_match: "Target version",
    close: "Nearest version",
    nearby: "Nearest version",
    compatible: "Nearest version",
    target_match: "Target version",
    nearest: "Nearest version",
    older: "Older version",
    newer: "Newer version",
    unknown: "Unknown version",
    target_unknown: "Unknown target version",
  })[relation] || relation || "");
}

function applicabilityLabel(value) {
  return t(({
    preferred: "Preferred",
    preferred_nearest: "Preferred nearest",
    superseded_when_conflict: "Superseded on conflict",
    background: "Background",
    version_unknown: "Use with caution",
    candidate: "Candidate",
  })[value] || value || "");
}

function provenanceClass(origin, applicability, relation) {
  if (origin === "websearch" || applicability === "preferred" || applicability === "preferred_nearest") return "preferred";
  if (applicability === "superseded_when_conflict" || relation === "older") return "superseded";
  if (relation === "unknown" || applicability === "version_unknown") return "unknown";
  return "muted";
}

function versionRelation(sourceVersion) {
  const targetVersion = appState.runPayload?.state?.target_profile?.target_version;
  if (!targetVersion || targetVersion === "unknown") return "";
  if (!sourceVersion || sourceVersion === "unknown") return "unknown";
  return String(sourceVersion).trim() === String(targetVersion).trim() ? "target_match" : "older";
}

function stageName(stage) {
  return t(({
    discovering_sources: "Searching for manuals",
    awaiting_source_selection: "Awaiting manual selection",
    source_selected: "Manual selected",
    previewing_scope: "Preparing reviewed chapter choices",
    awaiting_scope_selection: "Choose extraction scope",
    scope_selected: "Extraction scope selected",
    created: "Registered",
    registered: "Registered",
    acquiring: "Acquiring source material",
    acquisition: "Acquiring source material",
    normalizing: "Parsing source material and building evidence",
    modal_preprocessing: "Modal Agent processing",
    source_review: "Reviewing source evidence",
    source_validation: "Validating source branches",
    version_alignment: "Aligning source versions",
    corpus_snapshot: "Creating CorpusSnapshot",
    corpus_frozen: "Source evidence snapshot ready",
    task_and_reference: "Discovering Tasks and References",
    task_discovery: "Discovering Tasks",
    reference_build: "Building References",
    reference_building: "Building References from EvidenceTree units",
    references_ready: "References ready (paused before Task discovery)",
    discovering_task_trajectories: "Discovering Tasks and Trajectories",
    discovering_knowledge: "Discovering Tasks from EvidenceTree units",
    tasks_ready: "Tasks and References ready for KnowledgeSnapshot",
    aligning_knowledge: "Freezing KnowledgeSnapshot",
    knowledge_frozen: "KnowledgeSnapshot frozen",
    knowledge_alignment: "Auditing knowledge coverage gaps",
    knowledge_snapshot: "Freezing KnowledgeSnapshot",
    inferring_trajectories: "Inferring operation trajectories",
    trajectory_inference: "Inferring operation trajectories",
    trajectory_validation: "Validating operation trajectories",
    inducing_skills: "Inducing Skills",
    skill_induction: "Inducing Skills",
    skill_validation: "Validating Skills",
    validating: "Validating final candidates",
    committing: "Committing Skill Registry",
    registry_commit: "Committing Skill Registry",
    cancelling: "Stopping run",
    cancelled: "Stopped",
    completed: "Run completed",
    degraded: "Run completed with gaps",
    failed: "Run failed",
  })[stage] || stage || "Waiting");
}

function statusClass(status) {
  if (["succeeded", "completed", "references_ready", "tasks_ready"].includes(status)) return "completed";
  if (["running", "processing", "queued", "cancelling"].includes(status)) return "running";
  if (status === "cancelled") return "cancelled";
  if (status === "degraded") return "degraded";
  if (["failed", "excluded"].includes(status)) return status;
  return "pending";
}

function statusLabel(status) {
  return t(({ queued: "Queued", running: "Running", cancelling: "Stopping", paused: "Paused", references_ready: "Paused", tasks_ready: "Paused", succeeded: "Completed", completed: "Completed", degraded: "With gaps", failed: "Failed", cancelled: "Stopped", waiting_for_input: "Waiting for your selection", processing: "processing", excluded: "excluded" })[status] || status || "Waiting");
}

function formatLocator(locator = {}) {
  const range = locator.char_start !== undefined ? ` · ${t("Characters {start}-{end}", { start: locator.char_start, end: locator.char_end })}` : "";
  if (locator.page_start !== undefined) {
    const pages = locator.page_start === locator.page_end
      ? t("Page {page}", { page: locator.page_start })
      : t("Pages {start}-{end}", { start: locator.page_start, end: locator.page_end });
    return `${pages}${locator.heading ? ` · ${locator.heading}` : ""}${range}`;
  }
  if (locator.page) return `${t("Page {page}", { page: locator.page })}${range}`;
  if (locator.heading) return `${locator.heading}${locator.chunk > 1 ? ` · ${locator.chunk}` : ""}${range}`;
  if (locator.url) return `${locator.url}${range}`;
  if (locator.time_start !== undefined) return `${locator.time_start}s${range}`;
  return JSON.stringify(locator);
}

function formatTime(value) {
  try { return new Date(value).toLocaleTimeString(appState.language === "Chinese" ? "zh-CN" : "en-GB", { hour12: false }); }
  catch { return value || ""; }
}

function formatApiError(detail) {
  const codes = {
    api_key_required: "Your model API Key is required to continue.",
    missing_api_key: "Your model API Key is required to continue.",
    search_not_configured: "The search service is not configured.",
    source_discovery_not_configured: "The search service is not configured.",
    source_selection_not_ready: "This run is not waiting for a source selection.",
    invalid_candidate: "The selected candidate is no longer available.",
  };
  if (typeof detail === "string") return t(detail);
  if (Array.isArray(detail)) return detail.map((item) => t(item.msg || JSON.stringify(item))).join("；");
  if (detail?.code && codes[detail.code]) return t(codes[detail.code]);
  if (detail?.message) return t(detail.message);
  return detail ? JSON.stringify(detail) : t("Request failed");
}

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function refreshIcons() {
  if (window.lucide) window.lucide.createIcons({ attrs: { "stroke-width": 1.8 } });
}

// Translate product text only. Source excerpts, generated artifacts, and technical logs stay intact.
const UI_ZH = Object.freeze({
  "Generation Coverage": "生成覆盖度",
  "Higher coverage increases API costs and takes longer.": "覆盖度越高，API 费用越高，耗时越长。",
  "This run's coverage is saved. It stays unchanged while the run is in progress.": "本次任务的覆盖度已保存，运行期间保持不变。",
  "Low": "低",
  "Medium": "中",
  "High": "高",
  "Ultra": "Ultra",
  "Next: choose material": "下一步：选择资料",
  "Next: configure model": "下一步：配置模型",
  "Back to Target Environment": "返回填写目标环境",
  "The pipeline will search for manuals after you configure your model.": "完成模型配置后，流程会自动搜索资料。",
  "Please enter a valid manual URL.": "请填写有效的资料链接。",
  "Extraction scope": "知识生成范围",
  "Choose chapters": "选择章节",
  "Which chapters do you need?": "你希望提取哪些章节？",
  "Choose the whole material or first-level chapters. Each selected chapter includes all its subsections.": "可以选择整份资料，也可以选择一个或多个一级目录。所选一级目录包含其全部子章节。",
  "Whole material": "整份资料",
  "Extract knowledge across all chapters.": "提取资料中所有章节的知识。",
  "Selected chapters": "指定章节",
  "Choose one or more first-level chapters below.": "在下方选择一个或多个一级目录。",
  "Chapters": "章节",
  "Source material": "资料",
  "A reliable chapter outline could not be identified. You can continue with the whole material.": "未能识别出可靠的章节目录。你可以继续提取整份资料。",
  "Confirm scope and continue": "确认范围并继续",
  "Continue extraction": "继续生成知识",
  "Waiting to continue": "等待继续",
  "Your extraction scope is saved. Continue using the same selection.": "知识生成范围已保存，可以使用原来的选择继续。",
  "Parsing and structure review are complete. Choose the chapters to use for knowledge generation.": "资料抓取、解析与结构审核已完成，请选择用于后续知识生成的章节。",
  "Preparing reviewed chapter choices": "正在整理审核后的章节选项",
  "{count} parsed characters including subsections": "已解析 {count} 个字符（含子章节）",
  "{count} pages of material parsed.": "本次已解析 {count} 页资料。",
  "This material reached the collection limit. Some linked pages were not included. You can continue with the available chapters.": "资料采集已达到本次处理上限，部分关联页面尚未纳入。您可以基于已获取的章节继续生成。",
  "Some material could not be fully collected or processed. You can continue with the content available for this run.": "部分资料未能完整采集或解析。您可以基于本次已获取的内容继续生成。",
  "Pages {start}–{end}": "第 {start}–{end} 页",
  "Starts on page {page}": "起始页：第 {page} 页",
  "Page boundaries will be resolved during parsing.": "解析时将确定页码范围。",
  "{count} known linked pages; more may be found.": "已识别 {count} 个关联页面，后续可能发现更多。",
  "Page links will be expanded after selection.": "选择范围后将继续补充关联页面。",
  "{count} chapters selected": "已选择 {count} 个章节",
  "This run can collect up to {count} linked pages in addition to the starting page.": "本次最多采集 {count} 个关联页面，起始页另计。",
  "Choose at least one chapter, or select the whole material.": "请至少选择一个章节，或者选择整份资料。",
  "The outline is ready. Choose the whole material or the chapters you need before extraction starts.": "章节目录已准备好，请选择整份资料或需要的章节，再开始提取。",
  "Reading the document outline": "正在识别章节目录",
  "Choose extraction scope": "等待选择提取范围",
  "Extraction scope selected": "已确认提取范围",
  "Source image": "原文图片",
  "Image unavailable": "图片暂不可用",
  "Needs attention": "请补全信息",
  "Please complete the highlighted required fields.": "请补全标红的必填项。",
  "Please fill in {field}.": "请填写{field}。",
  "Please check {field}.": "请检查{field}的填写内容。",
  "Please enter your API Key before generating.": "请先填写 API Key，再点击生成。",
  "Please enter a valid Base URL (http:// or https://).": "请填写以 http:// 或 https:// 开头的有效 Base URL。",
  "Continue with saved model: {model} at {url}": "本次任务继续使用已保存的模型：{model}，接口地址：{url}",
  "Model API Key is required to continue this Run after a server restart": "服务重启后，请重新填写模型 API Key 以继续本次任务。",
  "Unknown source candidate": "所选候选资料已不存在，请重新选择。",
  "This Run is not waiting for source selection": "本次任务当前未等待选择资料。",
  "A different source has already been selected": "本次任务已经选择了其他资料。",
  "The search task is still finishing; please retry": "搜索正在保存结果，请稍后重试。",
  "This Run is not available for another search": "本次任务当前无法重新搜索。",
  "Please provide exactly one PDF or HTML source": "请提供一份 PDF 或 HTML 资料。",
  "Automatic discovery does not accept uploaded sources": "自动搜索模式无需上传资料。",
  "Output language must be Chinese or English": "请选择中文或英文。",
  "Connecting": "连接中",
  "Service available": "服务可用",
  "Service unavailable": "服务不可用",
  "Page language": "页面语言",
  "Target Environment": "目标环境",
  "Software": "软件",
  "Target Version": "目标版本",
  "Select platform": "选择平台",
  "Platform": "平台",
  "General": "通用",
  "Sources": "资料",
  "Add material": "添加资料",
  "Material added": "已添加资料",
  "Add one manual to ground your Skills.": "添加一份手册，作为 Skill 的资料依据。",
  "Choose a file or paste a link to get started.": "选择文件或粘贴链接即可开始。",
  "Material input method": "资料输入方式",
  "Upload file": "上传文件",
  "Paste link": "粘贴链接",
  "Drop your manual here": "将手册拖到这里",
  "PDF or HTML · Click to browse": "PDF 或 HTML · 点击选择文件",
  "Replace file": "更换文件",
  "Manual link": "手册链接",
  "Paste a manual URL or a file:/// path.": "粘贴手册网址或 file:/// 本地路径。",
  "Clear link": "清除链接",
  "Source version": "资料版本",
  "Optional": "可选",
  "e.g. 1.9": "例如 1.9",
  "PDF document": "PDF 文档",
  "HTML document": "HTML 文档",
  "Choose a model": "选择模型",
  "Configured": "已配置",
  "Set up": "待配置",
  "0 items": "0 份资料",
  "Select PDF or HTML": "选择 PDF 或 HTML",
  "URL / Local file:///": "URL / 本地 file:///",
  "Add URL": "添加 URL",
  "Model Configuration": "模型配置",
  "Base URL": "接口地址",
  "Protocol": "协议",
  "Model": "模型",
  "Reasoning Effort": "推理强度",
  "Automatic": "自动选择",
  "Default": "默认值",
  "Model identifier from your provider": "填写服务商提供的模型名称",
  "none": "无",
  "low": "低",
  "medium": "中",
  "high": "高",
  "xhigh": "很高",
  "max": "最高",
  "API Key": "API Key",
  "Model API Key": "模型 API Key",
  "Used in memory for this run only": "仅在内存中供本次任务使用",
  "Generate": "生成",
  "Registering": "创建任务中",
  "Starting Pipeline": "启动流程中",
  "Recent Runs": "最近任务",
  "Refresh": "刷新",
  "Refresh recent runs": "刷新最近任务",
  "No Run Selected": "尚未选择任务",
  "Loading run…": "正在加载任务…",
  "Run Results": "任务结果",
  "queued": "排队中",
  "Run Full Pipeline from T5": "从 T5 运行完整流程",
  "Create a new Run from this frozen T5 snapshot and continue through Reference, Tasks, Trajectories, and Skills": "基于此 T5 冻结快照创建新任务，继续生成 Reference、任务、轨迹和 Skill",
  "Stop Run": "停止任务",
  "Pipeline progress": "流程进度",
  "Registered": "已创建",
  "Queued": "排队中",
  "Overall pipeline progress": "整体流程进度",
  "Pipeline stages": "流程阶段",
  "Source & Evidence Build": "资料与证据构建",
  "Reference Build": "Reference 构建",
  "Task & Trajectory Generation": "任务与轨迹生成",
  "Skill Generation": "Skill 生成",
  "Finalizing": "正在收尾",
  "Find Material": "查找资料",
  "Pending": "待处理",
  "Completed": "已完成",
  "Paused": "已暂停",
  "Stopping": "停止中",
  "In progress": "进行中",
  "Running": "运行中",
  "With gaps": "存在缺口",
  "Failed": "失败",
  "Stopped": "已停止",
  "Waiting": "等待中",
  "Run result views": "任务结果视图",
  "Evidence": "证据",
  "Reference": "Reference",
  "Knowledge content": "知识正文",
  "Task and Trajectory": "任务与轨迹",
  "Skills": "Skills",
  "Audit Trail": "审计日志",
  "Source": "资料",
  "Modality": "类型",
  "Version": "版本",
  "Origin": "来源",
  "Status": "状态",
  "Coverage": "覆盖情况",
  "Excerpts": "摘录",
  "Waiting for source results": "等待资料处理结果",
  "Source selected": "资料已选中",
  "Processing source material": "正在处理资料",
  "Evidence view": "证据视图",
  "Evidence Tree": "证据树",
  "Flat Evidence": "平铺证据",
  "Filter tree nodes, content, or sources": "筛选树节点、内容或资料",
  "Filter excerpts, titles, or sources": "筛选摘录、标题或资料",
  "Filter by modality": "按类型筛选",
  "All modalities": "所有类型",
  "References": "References",
  "Tasks": "任务",
  "Trajectories": "轨迹",
  "All": "全部",
  "Tools": "工具",
  "Errors": "错误",
  "Evidence: {accepted}/{total} accepted; {rejected} rejected": "Evidence：接受 {accepted}/{total} 条；拒绝 {rejected} 条",
  "View validation report": "查看校验报告",
  "Auto-follow": "自动滚动",
  "Source-preserved Catalog": "保留原文的目录",
  "Close full Reference": "关闭完整 Reference",
  "Search entry name, content, or source locator": "搜索条目名称、内容或资料位置",
  "Pipeline Stopped": "流程已停止",
  "Pipeline Run Failed": "流程运行失败",
  "Dismiss": "关闭",
  "Close": "关闭",
  "Change": "更改",
  "Your own material": "自备资料",
  "Search for material": "自动查找资料",
  "Have you prepared PDF or HTML material?": "你是否已准备好 PDF 或 HTML 资料？",
  "Choose how to provide one manual for this run.": "请选择本次任务的资料来源，每次使用一份手册。",
  "I have prepared material": "我准备好了",
  "Upload one PDF or HTML file, or enter a manual URL.": "上传一份 PDF 或 HTML 文件，或填写手册 URL。",
  "I have not prepared material": "我还没有准备",
  "The pipeline will search for manuals. Configure your target and model to begin.": "流程将自动查找手册。请填写目标环境和模型配置。",
  "Upload your PDF or HTML here.": "你可以在这里上传 PDF 或 HTML 资料。",
  "Configure your model below. The pipeline will search for PDF or HTML manuals after you click Generate.": "请填写下方模型配置，点击“生成”后，流程会自动搜索 PDF 或 HTML 手册。",
  "The pipeline will search for PDF or HTML manuals. Fill in the target environment and model configuration, then click Generate.": "流程会自动搜索 PDF 或 HTML 手册。请填写目标环境和模型配置，然后点击“生成”。",
  "Use one PDF, one HTML file, or one HTML manual root URL.": "使用一份 PDF、一份 HTML 文件，或一个 HTML 手册根目录 URL。",
  "Manual search": "资料搜索",
  "Choose one manual to continue": "选择一份手册以继续",
  "Choose one manual to continue.": "请选择一份手册以继续。",
  "Choose material": "选择资料",
  "Use my own material": "自行上传资料",
  "Search again": "重新搜索",
  "Use this material and continue": "使用此资料并继续",
  "Search time limit reached. Saved manuals are available below; version verification may be incomplete.": "\u641c\u7d22\u65f6\u95f4\u5df2\u5230\u3002\u4ee5\u4e0b\u662f\u5df2\u4fdd\u5b58\u7684\u624b\u518c\uff0c\u7248\u672c\u6838\u67e5\u53ef\u80fd\u5c1a\u672a\u5b8c\u6210\u3002",
  "Search ended. Previously verified and saved manuals are available below.": "\u641c\u7d22\u5df2\u7ed3\u675f\u3002\u4ee5\u4e0b\u662f\u5df2\u6838\u5b9e\u5e76\u4fdd\u5b58\u7684\u624b\u518c\u3002",
  "Compare the candidates with your target version {version}. Choose one manual. PDF files will be downloaded; HTML manuals will be crawled.": "请与目标版本 {version} 对照后选择一份手册。PDF 将下载，HTML 手册将抓取。",
  "No usable manual was found. Search again or provide your own PDF or HTML material.": "未找到可用手册。你可以重新搜索，或自行提供 PDF 或 HTML 资料。",
  "Search could not finish because the search service failed. Search again or provide your own PDF or HTML material.": "搜索服务失败，未能完成资料检索。你可以重新搜索，或自行提供 PDF 或 HTML 资料。",
  "Sorry, the configured search service did not provide verifiable web search. Please check the search configuration.": "很抱歉，当前搜索服务未能提供可核实的联网搜索结果，请检查搜索配置。",
  "Sorry, the search service rejected the API credentials or access permissions. Please check the search configuration.": "很抱歉，搜索服务未通过 API 凭据或访问权限验证，请检查搜索配置。",
  "Sorry, the search service has no available quota. Please check its quota or billing settings.": "很抱歉，搜索服务的可用额度不足，请检查额度或计费设置。",
  "Sorry, the search service is rate limited. Please wait before retrying.": "很抱歉，搜索服务触发了请求频率限制，请稍后重试。",
  "Sorry, the search configuration could not be used. Please check the search settings.": "很抱歉，当前搜索配置无法使用，请检查搜索设置。",
  "Sorry, the search service failed, so this search could not be completed.": "很抱歉，搜索服务失败，未能完成本次资料检索。",
  "Sorry, some inspected documentation pages required sign-in or authorization, and no usable manual was saved.": "很抱歉，检查过的部分文档页面要求登录或授权，本次尚未保存可用手册。",
  "Sorry, some inspected documentation pages refused access, and no usable manual was saved.": "很抱歉，检查过的部分文档页面拒绝访问，本次尚未保存可用手册。",
  "Sorry, some inspected documentation links returned 404, and no usable manual was saved.": "很抱歉，检查过的部分文档链接返回 404，本次尚未保存可用手册。",
  "Sorry, the inspected documentation could not be retrieved, and no usable manual was saved.": "很抱歉，未能获取所检查的文档内容，本次尚未保存可用手册。",
  "Sorry, no usable manual was found within this search.": "很抱歉，本次搜索未找到可用手册。",
  "Sorry, the search time limit was reached before a usable manual was saved.": "很抱歉，搜索时间已到，尚未保存可用手册。",
  "Sorry, search was interrupted before a usable manual was saved.": "很抱歉，搜索中断，尚未保存可用手册。",
  "You can search again, upload a PDF or HTML manual, or provide a directly accessible official manual link.": "你可以重新搜索、自行上传 PDF 或 HTML 手册，或提供可直接访问的官方手册链接。",
  "Some inspected material links could not be accessed.": "部分已检查的资料链接无法访问。",
  "Search time limit reached before a usable manual was saved. Search again or provide your own material.": "搜索时间已到，尚未保存可用手册。你可以重新搜索，或自行提供资料。",
  "Search was interrupted before a usable manual was saved. Search again or provide your own material.": "搜索中断，尚未保存可用手册。你可以重新搜索，或自行提供资料。",
  "Some searches failed. Previously verified and saved manuals are available below.": "部分搜索失败，已核实并保存的手册仍可在下方选择。",
  "Official": "官方",
  "Third party": "第三方",
  "Unknown source": "来源未确认",
  "Document version": "资料实际版本",
  "Version match": "版本匹配关系",
  "Version evidence": "版本依据",
  "Manual language": "手册语言",
  "Manual root assessment": "手册根目录判断",
  "Material scope": "资料覆盖范围",
  "Included topics": "包含的教学内容",
  "Outside this collection": "本资料集以外的内容",
  "Scope limitations": "范围核验的限制",
  "Recommendation": "推荐说明",
  "Chinese": "中文",
  "English": "英文",
  "unknown": "未知",
  "Unknown version": "版本未知",
  "Re-enter your model API Key": "请重新填写模型 API Key",
  "The server needs your model API Key again. Enter it in Model Configuration before continuing.": "服务器需要你重新提供模型 API Key，请在模型配置中填写后继续。",
  "Enter your model API Key in Model Configuration before continuing.": "请在模型配置中填写模型 API Key 后继续。",
  "Waiting for your selection": "等待你选择资料",
  "Search is complete. Choose one manual to continue the pipeline.": "搜索完成。请选择一份手册，继续后续流程。",
  "Search is complete. No usable manual was found. Search again or provide your own material.": "搜索完成，未找到可用手册。你可以重新搜索或自行提供资料。",
  "Use only one PDF, HTML file, or manual URL for this run.": "本次任务只能使用一份 PDF、HTML 文件或一个手册 URL。",
  "Choose a PDF or HTML file.": "请选择 PDF 或 HTML 文件。",
  "Please provide exactly one PDF, HTML file, or manual URL.": "请提供一份 PDF、HTML 文件或一个手册 URL。",
  "Remove": "移除",
  "Remove {name}": "移除 {name}",
  "Version of {name}": "{name} 的版本",
  "Source URL": "资料 URL",
  "URL source version": "URL 资料版本",
  "Remove URL": "移除 URL",
  "https://... or file:///C:/...": "https://... 或 file:///C:/...",
  "{count} item(s)": "{count} 份资料",
  "No runs yet": "暂无任务",
  "Run not found": "未找到任务",
  "Unable to Start Pipeline": "无法启动流程",
  "Unable to Start Pipeline from T5": "无法从 T5 启动流程",
  "Request failed": "请求失败",
  "The pipeline encountered an unrecovered error.": "流程发生了无法自动恢复的错误。",
  "No Usable Sources": "没有可用资料",
  "API Quota Exhausted": "API 配额已用尽",
  "API Request Timed Out": "API 请求超时",
  "API Rate Limit Exceeded": "API 请求超出速率限制",
  "API Authentication Failed": "API 身份验证失败",
  "Insufficient API Permissions": "API 权限不足",
  "Cannot Connect to API Service": "无法连接 API 服务",
  "Invalid API Response": "API 响应无效",
  "Trajectory Agent Produced No Valid Candidates": "轨迹 Agent 未生成有效候选",
  "Skill Agent Produced No Valid Candidates": "Skill Agent 未生成有效候选",
  "Task Discovery Agent Produced No Valid Candidates": "任务发现 Agent 未生成有效候选",
  "Reference Agent Produced No Valid Candidates": "Reference Agent 未生成有效候选",
  "Coverage Gap Audit Agent Failed": "覆盖缺口审计 Agent 失败",
  "Agent Stage Failed": "Agent 阶段失败",
  "Agent Reached the ReAct Step Limit": "Agent 达到 ReAct 步数上限",
  "API Agent Failed": "API Agent 失败",
  "Run ID": "任务 ID",
  "Error Category": "错误类别",
  "HTTP Status": "HTTP 状态",
  "Exception Type": "异常类型",
  "Total Sources": "资料总数",
  "Excluded Sources": "排除资料数",
  "Acquisition Failures": "采集失败数",
  "Reason Counts": "原因统计",
  "All sources were excluded or failed during acquisition. The pipeline has stopped.": "全部资料均被排除或采集失败，流程已停止。",
  "The pipeline has stopped, but the backend did not provide detailed error information.": "流程已停止，后端未提供详细错误信息。",
  "Version conflict": "版本冲突",
  "Discovered {count} · Semantic units {units}": "发现 {count} · 语义单元 {units}",
  "Evidence-bearing excerpts {count} · Skipped {skipped}": "含证据摘录 {count} · 跳过 {skipped}",
  "No matching Evidence": "没有匹配的证据",
  "No Evidence yet": "暂无证据",
  "Preparing Evidence…": "正在准备证据…",
  "Preparing References…": "正在准备 Reference…",
  "Preparing Tasks…": "正在准备任务…",
  "Preparing Trajectories…": "正在准备轨迹…",
  "Preparing Skills…": "正在准备 Skill…",
  "Results will appear here as the pipeline advances.": "流程推进后，结果将在这里显示。",
  "Text + images": "文本与图片",
  "Image Evidence": "图片证据",
  "Exact text": "原文文本",
  "Evidence image on page {page}": "第 {page} 页的证据图片",
  "Page {page}": "第 {page} 页",
  "Pages {start}-{end}": "第 {start}–{end} 页",
  "Characters {start}-{end}": "字符 {start}–{end}",
  "Evidence source": "证据来源",
  "{count} nodes": "{count} 个节点",
  "{count} leaves": "{count} 个叶节点",
  "{count} SourceParts": "{count} 个 SourcePart",
  "No matching EvidenceTree nodes": "没有匹配的证据树节点",
  "No EvidenceTree yet; use Flat Evidence for legacy runs": "暂无证据树；旧任务可使用平铺证据",
  "Exact content · {count} SourcePart(s)": "原文内容 · {count} 个 SourcePart",
  "Untitled node": "未命名节点",
  "{count} children": "{count} 个子节点",
  "{count} refs": "{count} 个引用",
  "{count} chars": "{count} 个字符",
  "Source provenance ({count})": "资料来源（{count}）",
  "No submitted artifacts yet": "暂无提交的产物",
  "Source-preserved catalog · {count} entries": "保留原文的目录 · {count} 个条目",
  "Injection policy: {policy}": "注入策略：{policy}",
  "+{count} entries": "+{count} 个条目",
  "{count} citations": "{count} 个引用",
  "View all entries in {title}": "查看 {title} 的全部条目",
  "View Full Reference": "查看完整 Reference",
  "Source-preserved entries.": "保留手册原文的条目。",
  "Formatted reference": "排版后的 Reference",
  "View source-preserved entries": "查看原始条目与来源",
  "{count} complete entries": "{count} 个完整条目",
  "Showing {count} / {total} entries": "显示 {count} / {total} 个条目",
  "{count} entries total": "共 {count} 个条目",
  "No matching full Reference entries.": "没有匹配的完整 Reference 条目。",
  "Untitled entry": "未命名条目",
  "Evidence: {id}": "证据：{id}",
  "View Citation": "查看引用",
  "View trajectory steps · {count} steps": "查看轨迹步骤 · {count} 步",
  "Observation:": "预期观察：",
  "This run contains only legacy placeholder artifacts, which are not shown as Skills.": "此任务仅包含旧版占位产物，不展示为 Skill。",
  "No deliverable, source-grounded Skills are available yet.": "暂无基于资料证据的可交付 Skill。",
  "Applicability": "适用范围",
  "Procedure": "操作步骤",
  "Preconditions": "前置条件",
  "Caveats": "注意事项",
  "{count} trajectories": "{count} 条轨迹",
  "{count} references": "{count} 个 Reference",
  "{count} steps": "{count} 步",
  "No matching events": "没有匹配的事件",
  "Model session waiting: {seconds}s elapsed, {idle}s since last activity": "模型会话等待：已用时 {seconds} 秒，{idle} 秒无新活动",
  "Model retry {retry}/{max} in {seconds}s: {reason}": "模型重试 {retry}/{max}，{seconds} 秒后重试：{reason}",
  "Model retry {retry} started": "模型第 {retry} 次重试开始",
  "Model retry stopped: {reason}": "模型已停止重试：{reason}",
  "Version information": "版本信息",
  "Online source": "在线资料",
  "User upload": "用户上传",
  "Official documentation": "官方文档",
  "Release notes": "发行说明",
  "GitHub release": "GitHub 发行版",
  "GitHub tag": "GitHub 标签",
  "GitHub documentation": "GitHub 文档",
  "Help / manual": "帮助 / 手册",
  "Tutorial": "教程",
  "Community": "社区",
  "Unclassified": "未分类",
  "Target version": "目标版本一致",
  "Nearest version": "接近版本",
  "Older version": "较旧版本",
  "Newer version": "较新版本",
  "Unknown target version": "目标版本未知",
  "Preferred": "优先采用",
  "Preferred nearest": "优先采用近版本",
  "Superseded on conflict": "冲突时被替代",
  "Background": "背景资料",
  "Use with caution": "谨慎使用",
  "Candidate": "候选",
  "Searching for manuals": "搜索手册中",
  "Awaiting manual selection": "等待选择手册",
  "Manual selected": "已选择手册",
  "Acquiring source material": "采集资料",
  "Parsing source material and building evidence": "解析资料并构建证据",
  "Modal Agent processing": "模态 Agent 处理中",
  "Reviewing source evidence": "审查资料证据",
  "Validating source branches": "校验资料分支",
  "Aligning source versions": "对齐资料版本",
  "Creating CorpusSnapshot": "创建资料快照",
  "Source evidence snapshot ready": "资料证据快照已就绪",
  "Discovering Tasks and References": "发现任务与 Reference",
  "Discovering Tasks": "发现任务",
  "Building References": "构建 Reference",
  "Building References from EvidenceTree units": "基于证据树单元构建 Reference",
  "References ready (paused before Task discovery)": "Reference 已就绪（任务发现前暂停）",
  "Discovering Tasks and Trajectories": "发现任务与轨迹",
  "Discovering Tasks from EvidenceTree units": "基于证据树单元发现任务",
  "Tasks and References ready for KnowledgeSnapshot": "任务与 Reference 已就绪，可创建知识快照",
  "Freezing KnowledgeSnapshot": "冻结知识快照",
  "KnowledgeSnapshot frozen": "知识快照已冻结",
  "Auditing knowledge coverage gaps": "审计知识覆盖缺口",
  "Inferring operation trajectories": "推导操作轨迹",
  "Validating operation trajectories": "校验操作轨迹",
  "Inducing Skills": "推导 Skills",
  "Validating Skills": "校验 Skills",
  "Validating final candidates": "校验最终候选",
  "Committing Skill Registry": "写入 Skill 注册表",
  "Stopping run": "停止任务中",
  "Run completed": "任务已完成",
  "Run completed with gaps": "任务已完成，存在缺口",
  "Run failed": "任务失败",
  "succeeded": "已成功",
  "completed": "已完成",
  "degraded": "存在缺口",
  "failed": "失败",
  "cancelled": "已停止",
  "excluded": "已排除",
  "processing": "处理中",
  "running": "运行中",
  "paused": "已暂停",
  "waiting_for_input": "等待选择",
  "cancelling": "停止中",
  "Your model API Key is required to continue.": "继续前需要提供模型 API Key。",
  "The search service is not configured.": "搜索服务尚未配置。",
  "This run is not waiting for a source selection.": "此任务当前未在等待选择资料。",
  "The selected candidate is no longer available.": "所选候选资料已不可用。"
});

// Call once after #skills-list exists. Polling only updates content; motion starts
// on an explicit summary activation, so unchanged cards never animate again.
const skillMotionAnimations = new WeakMap();
const skillMotionBoundLists = new WeakSet();

function getSkillMotionOpenState(detail) {
  return skillMotionAnimations.get(detail)?.open ?? detail.open;
}

function finishSkillMotion(detail, motion) {
  if (skillMotionAnimations.get(detail) !== motion) return;
  skillMotionAnimations.delete(detail);
  motion.animation?.cancel();
  detail.open = motion.open;
  detail.style.height = motion.height;
  detail.style.overflow = motion.overflow;
  delete detail.dataset.skillMotionOpen;
}

function bindSkillMotion() {
  const list = document.querySelector("#skills-list");
  if (!list || skillMotionBoundLists.has(list)) return;
  skillMotionBoundLists.add(list);
  const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)");

  list.addEventListener("click", (event) => {
    const summary = event.target.closest?.("summary.skill-summary");
    const detail = summary?.parentElement;
    if (!summary || !list.contains(summary) || !detail?.matches("details.skill-details")) return;
    if (event.defaultPrevented || event.target.closest("a, button, input, select, textarea, [contenteditable='true']")) return;
    // Native <summary> handles pointer, Enter and Space without extra key handlers.
    // Leave that behavior untouched when animation is unavailable or unwanted.
    if (reducedMotion.matches || typeof detail.animate !== "function") return;

    const previous = skillMotionAnimations.get(detail);
    const open = !(previous?.open ?? detail.open);
    const start = detail.getBoundingClientRect().height;
    const height = previous?.height ?? detail.style.height;
    const overflow = previous?.overflow ?? detail.style.overflow;
    previous?.animation?.cancel();
    skillMotionAnimations.delete(detail);

    event.preventDefault();
    detail.dataset.skillMotionOpen = String(open);
    // Keep the body present while it folds shut; the getter above exposes the
    // user's target to polling, even during this brief native-open interval.
    detail.open = true;
    detail.style.height = "auto";
    const computed = getComputedStyle(detail);
    const frame = ["borderTopWidth", "borderBottomWidth", "paddingTop", "paddingBottom"]
      .reduce((sum, property) => sum + (parseFloat(computed[property]) || 0), 0);
    const end = open ? detail.getBoundingClientRect().height : summary.getBoundingClientRect().height + frame;
    detail.style.height = `${start}px`;
    detail.style.overflow = "hidden";

    const motion = { open, height, overflow, animation: null };
    skillMotionAnimations.set(detail, motion);
    try {
      motion.animation = detail.animate([{ height: `${start}px` }, { height: `${end}px` }], {
        duration: Math.min(300, 180 + Math.abs(end - start) * 0.08),
        easing: "cubic-bezier(.22, .68, .2, 1)",
        fill: "both",
      });
      motion.animation.finished.then(
        () => finishSkillMotion(detail, motion),
        () => { /* A newer user activation can cancel and reverse the motion. */ },
      );
    } catch {
      // Native details remains fully usable if the animation API cannot run.
      finishSkillMotion(detail, motion);
    }
  });

  reducedMotion.addEventListener?.("change", () => {
    if (!reducedMotion.matches) return;
    list.querySelectorAll("details.skill-details").forEach((detail) => {
      const motion = skillMotionAnimations.get(detail);
      if (motion) finishSkillMotion(detail, motion);
    });
  });
}

function shouldShowResultSkeleton(section) {
  const state = appState.runPayload?.state;
  if (!state || !["queued", "running"].includes(state.status)) return false;
  // A stopped run must show its actual empty result, even if its status is stale.
  if (["completed", "degraded", "failed", "cancelled"].includes(state.phase)
    || ["completed", "degraded", "failed", "cancelled"].includes(state.stage)) return false;
  if (isAwaitingSource(state) || isAwaitingScope(state) || state.phase === "previewing_scope") return false;

  const result = appState.runPayload?.result || {};
  const snapshot = result.snapshot || {};
  const offset = state.source_mode === "discovery" ? 1 : 0;
  let stageIndex;
  if (section === "evidence") {
    // Do not cover an existing alternate Evidence view or a filtered empty state.
    if ((snapshot.evidence || []).length || getEvidenceTrees(snapshot).some((tree) => (tree.nodes || []).length)) return false;
    if ($("#evidence-filter")?.value?.trim() || ($("#evidence-modality")?.value || "all") !== "all") return false;
    stageIndex = offset;
  } else if (section === "reference") {
    if ((result.references || []).length) return false;
    stageIndex = offset + 1;
  } else if (section === "tasks" || section === "trajectories") {
    // The two columns can receive artifacts at different times.
    if ((result[section] || []).length) return false;
    stageIndex = offset + 2;
  } else if (section === "skills") {
    // Preserve the existing explanation for legacy placeholder artifacts as well.
    if ((result.skills || []).length) return false;
    stageIndex = offset + 3;
  } else {
    return false;
  }

  const { stages } = getPipelineProgress(state);
  // An unknown phase is not evidence that the backend is still producing data.
  if (!stages.some((stage) => stage.active && !stage.paused)) return false;
  return Boolean(stages[stageIndex] && !stages[stageIndex].done);
}

function renderResultSkeleton(section, message = "", count = 3) {
  const messages = {
    evidence: "Preparing Evidence…",
    reference: "Preparing References…",
    tasks: "Preparing Tasks…",
    trajectories: "Preparing Trajectories…",
    skills: "Preparing Skills…",
  };
  const variant = Object.hasOwn(messages, section) ? section : "skills";
  const label = escapeHtml(t(message || messages[variant]));
  const cardCount = Math.max(1, Math.min(4, Math.floor(Number(count) || 3)));
  // Markup is deterministic. Callers should retain it while the section is loading.
  return `<div class="result-skeleton result-skeleton-${variant}" role="status" aria-live="polite" aria-atomic="true">
    <div class="result-skeleton-caption"><span class="result-skeleton-dot" aria-hidden="true"></span><span>${label}</span></div>
    <p class="result-skeleton-hint">${escapeHtml(t("Results will appear here as the pipeline advances."))}</p>
    <div class="result-skeleton-cards" aria-hidden="true">${Array.from({ length: cardCount }, (_, index) => `
      <div class="result-skeleton-card" style="--skeleton-delay: ${index * 120}ms">
        <div class="skeleton-card-heading"><span class="skeleton-shape skeleton-square"></span><span class="skeleton-shape skeleton-title"></span><span class="skeleton-shape skeleton-chip"></span></div>
        <span class="skeleton-shape skeleton-line"></span>
        <span class="skeleton-shape skeleton-line skeleton-line-short"></span>
        <div class="skeleton-card-footer"><span class="skeleton-shape skeleton-chip"></span><span class="skeleton-shape skeleton-chip skeleton-chip-wide"></span></div>
      </div>`).join("")}</div>
  </div>`;
}

// Use for lists without their own stable update helper (Reference, Tasks, Trajectories and Flat Evidence).
// Evidence trees and Skills should use updateEvidenceTreeList / updateSkillsList instead.
const resultSkeletonMarkup = new WeakMap();
function updateResultListMarkup(list, markup) {
  // The generated shape contains no user-editable state. Unchanged DOM keeps its
  // animation timeline, focus, and scroll position intact during the 900 ms poll.
  if (resultSkeletonMarkup.get(list) !== markup) {
    list.innerHTML = markup;
    resultSkeletonMarkup.set(list, markup);
  }
}
