/* ============================================================
   Service Desk — Ticket Assistant (SPA)
   Vanilla JS, no build step. Talks to the FastAPI backend:
     POST /classify/start, POST /classify/answer,
     GET /api/faq, POST /api/tickets
   ============================================================ */
(() => {
  'use strict';

  // ---------- Label metadata (mirrors config/taxonomy.yaml ids) ----------
  const LAYER_CAPTIONS = { layer1: 'Type', layer2: 'Domain' };
  const LABELS = {
    incident:        { name: 'Incident',        color: 'var(--c-incident)' },
    service_request: { name: 'Service Request', color: 'var(--c-service-request)' },
    erp:             { name: 'ERP',             color: 'var(--c-erp)' },
    staff:           { name: 'Staff',           color: 'var(--c-staff)' },
  };
  const labelName = (id) =>
    (LABELS[id] && LABELS[id].name) ||
    (id ? id.replace(/_/g, ' ').replace(/\b\w/g, (c) => c.toUpperCase()) : '—');
  const labelColor = (id) => (LABELS[id] && LABELS[id].color) || 'var(--text-3)';

  // ---------- State ----------
  const LS_IDENTITY = 'tta.identity';
  const LS_THEME = 'tta.theme';
  const state = {
    identity: null,           // {employeeId, firstName, lastName}
    faq: { categories: [], items: [] },
    activeCategory: null,
    query: '',
    ticket: { summary: '', description: '', templateName: null },
    sessionId: null,
    result: null,             // final classification result
    placeholderWarned: false,
  };

  const $ = (id) => document.getElementById(id);
  const esc = (s) =>
    String(s ?? '').replace(/[&<>"']/g, (c) =>
      ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));

  // ---------- Persian/Arabic-aware normalization (mirror of src/faq.py) ----------
  const CHAR_MAP = { 'ي': 'ی', 'ك': 'ک', 'ة': 'ه', 'أ': 'ا', 'إ': 'ا', 'ؤ': 'و', '‌': ' ' };
  for (let i = 0; i < 10; i++) {
    CHAR_MAP[String.fromCharCode(0x06f0 + i)] = String(i); // ۰-۹
    CHAR_MAP[String.fromCharCode(0x0660 + i)] = String(i); // ٠-٩
  }
  const normalize = (s) =>
    String(s || '').replace(/./g, (c) => CHAR_MAP[c] ?? c).toLowerCase();

  // ---------- API helper ----------
  async function api(path, body) {
    const res = await fetch(path, {
      method: body === undefined ? 'GET' : 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: body === undefined ? undefined : JSON.stringify(body),
    });
    if (!res.ok) {
      let detail = '';
      try { detail = (await res.json()).detail || ''; } catch { /* ignore */ }
      const message = typeof detail === 'string' ? detail : (detail.message || '');
      const err = new Error(message || `Request failed (${res.status})`);
      err.status = res.status;
      err.code = typeof detail === 'object' ? detail.code : undefined;
      throw err;
    }
    return res.json();
  }

  // ---------- Toasts ----------
  function toast(message, retryFn) {
    const region = $('toast-region');
    const el = document.createElement('div');
    el.className = 'toast';
    el.setAttribute('role', 'alert');
    el.innerHTML = `<span class="toast-msg">${esc(message)}</span>`;
    if (retryFn) {
      const btn = document.createElement('button');
      btn.className = 'btn btn-secondary btn-sm';
      btn.textContent = 'Retry';
      btn.addEventListener('click', () => { el.remove(); retryFn(); });
      el.appendChild(btn);
    }
    region.appendChild(el);
    setTimeout(() => el.remove(), 8000);
  }

  // ---------- Theme ----------
  function applyTheme(theme) {
    document.documentElement.dataset.theme = theme;
    localStorage.setItem(LS_THEME, theme);
  }
  function initTheme() {
    const saved = localStorage.getItem(LS_THEME);
    const prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
    applyTheme(saved || (prefersDark ? 'dark' : 'light'));
    $('theme-btn').addEventListener('click', () =>
      applyTheme(document.documentElement.dataset.theme === 'dark' ? 'light' : 'dark'));
  }

  // ---------- Brand logo (company logo with graceful fallback) ----------
  function initLogo() {
    const img = new Image();
    img.alt = '';
    img.onload = () => { $('brand-logo').replaceChildren(img); };
    img.onerror = () => { $('brand-logo').textContent = '🎫'; };
    img.src = '/api/logo';
  }

  // ---------- Views & stepper ----------
  const VIEWS = ['identity', 'home', 'compose', 'triage', 'done'];
  const STEP_OF_VIEW = { home: 1, compose: 1, triage: 2, done: 3 };

  function showView(name) {
    VIEWS.forEach((v) => { $(`view-${v}`).hidden = v !== name; });
    const step = STEP_OF_VIEW[name];
    $('stepper').hidden = !step;
    if (step) {
      document.querySelectorAll('.step').forEach((el) => {
        const n = Number(el.dataset.step);
        el.classList.toggle('active', n === step);
        el.classList.toggle('complete', n < step);
      });
    }
    const focusTargets = {
      identity: 'f-empid', home: 'faq-search', compose: 't-subject', triage: null, done: null,
    };
    const fid = focusTargets[name];
    if (fid) setTimeout(() => $(fid).focus(), 60);
    window.scrollTo({ top: 0 });
  }

  // ---------- Identity ----------
  function loadIdentity() {
    try {
      const raw = JSON.parse(localStorage.getItem(LS_IDENTITY) || 'null');
      if (raw && raw.employeeId && raw.firstName && raw.lastName) return raw;
    } catch { /* ignore */ }
    return null;
  }

  function renderUserChip() {
    const id = state.identity;
    const chip = $('user-chip');
    if (!id) { chip.hidden = true; return; }
    chip.hidden = false;
    $('user-avatar').textContent =
      (id.firstName[0] || '').toUpperCase() + (id.lastName[0] || '').toUpperCase();
    $('user-name').textContent = `${id.firstName} ${id.lastName}`;
    $('user-id').textContent = `ID ${id.employeeId}`;
    $('home-greeting').textContent = `Hi ${id.firstName} — how can we help?`;
  }

  function initIdentityForm() {
    const form = $('identity-form');
    form.addEventListener('submit', (e) => {
      e.preventDefault();
      const empid = $('f-empid').value.trim();
      const first = $('f-first').value.trim();
      const last = $('f-last').value.trim();
      const okId = /^\d{3,10}$/.test(empid);
      const okFirst = first.length >= 2;
      const okLast = last.length >= 2;
      setFieldError('f-empid', 'err-empid', !okId);
      setFieldError('f-first', 'err-first', !okFirst);
      setFieldError('f-last', 'err-last', !okLast);
      if (!(okId && okFirst && okLast)) return;
      state.identity = { employeeId: empid, firstName: first, lastName: last };
      localStorage.setItem(LS_IDENTITY, JSON.stringify(state.identity));
      renderUserChip();
      showView('home');
    });
    $('user-chip').addEventListener('click', () => {
      const id = state.identity || {};
      $('f-empid').value = id.employeeId || '';
      $('f-first').value = id.firstName || '';
      $('f-last').value = id.lastName || '';
      showView('identity');
    });
  }

  function setFieldError(inputId, errId, isInvalid) {
    $(errId).hidden = !isInvalid;
    $(inputId).closest('.field').classList.toggle('invalid', isInvalid);
    $(inputId).setAttribute('aria-invalid', String(isInvalid));
  }

  // ---------- FAQ ----------
  async function loadFaq() {
    try {
      const data = await api('/api/faq');
      state.faq.categories = data.categories || [];
      state.faq.items = (data.items || []).map((it) => ({
        ...it,
        blob: normalize([it.question, it.category, ...(it.keywords || [])].join(' ')),
      }));
    } catch {
      state.faq = { categories: [], items: [] };
    }
    renderChips();
    renderFaqList();
  }

  function renderChips() {
    const row = $('chip-row');
    row.replaceChildren();
    const mk = (label, value) => {
      const b = document.createElement('button');
      b.type = 'button';
      b.className = 'chip';
      b.setAttribute('role', 'tab');
      b.setAttribute('aria-selected', String(state.activeCategory === value));
      b.textContent = label;
      b.addEventListener('click', () => {
        state.activeCategory = value;
        renderChips();
        renderFaqList();
      });
      return b;
    };
    row.appendChild(mk('All', null));
    state.faq.categories.forEach((c) => row.appendChild(mk(c, c)));
  }

  // Highlight query terms: normalization is 1:1 per character, so match
  // positions on the normalized string map directly onto the original.
  function highlight(text, terms) {
    if (!terms.length) return esc(text);
    const norm = normalize(text);
    const ranges = [];
    terms.forEach((t) => {
      let from = 0;
      while (t) {
        const i = norm.indexOf(t, from);
        if (i === -1) break;
        ranges.push([i, i + t.length]);
        from = i + t.length;
      }
    });
    if (!ranges.length) return esc(text);
    ranges.sort((a, b) => a[0] - b[0]);
    let html = '';
    let pos = 0;
    ranges.forEach(([s, e]) => {
      if (s < pos) { e > pos && (html += `<mark>${esc(text.slice(pos, e))}</mark>`, pos = e); return; }
      html += esc(text.slice(pos, s)) + `<mark>${esc(text.slice(s, e))}</mark>`;
      pos = e;
    });
    return html + esc(text.slice(pos));
  }

  function renderFaqList() {
    const terms = normalize(state.query).split(/\s+/).filter(Boolean);
    const items = state.faq.items.filter((it) =>
      (!state.activeCategory || it.category === state.activeCategory) &&
      terms.every((t) => it.blob.includes(t)));

    const list = $('faq-list');
    list.replaceChildren();
    items.forEach((it) => {
      const li = document.createElement('li');
      const btn = document.createElement('button');
      btn.type = 'button';
      btn.className = 'faq-item';
      btn.innerHTML = `
        <span class="faq-item-main">
          <p class="faq-q" dir="auto">${highlight(it.question, terms)}</p>
          <span class="faq-cat">${esc(it.category)}</span>
        </span>
        <span class="faq-arrow" aria-hidden="true">
          <svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><path d="M5 12h14m-6-6 6 6-6 6"/></svg>
        </span>`;
      btn.addEventListener('click', () => openCompose(it));
      li.appendChild(btn);
      list.appendChild(li);
    });

    const n = items.length;
    $('faq-count').textContent = n
      ? `${n} common request${n === 1 ? '' : 's'}${state.query ? ' matching your search' : ''}`
      : '';
    $('faq-empty').hidden = n > 0;
  }

  function initSearch() {
    const input = $('faq-search');
    input.addEventListener('input', () => {
      state.query = input.value;
      $('search-clear').hidden = !input.value;
      renderFaqList();
    });
    $('search-clear').addEventListener('click', () => {
      input.value = '';
      state.query = '';
      $('search-clear').hidden = true;
      renderFaqList();
      input.focus();
    });
    document.addEventListener('keydown', (e) => {
      if (e.key === '/' && !$('view-home').hidden &&
          !/^(INPUT|TEXTAREA)$/.test(document.activeElement.tagName)) {
        e.preventDefault();
        input.focus();
      }
    });
  }

  // ---------- Compose ----------
  function openCompose(faqItem) {
    state.placeholderWarned = false;
    $('compose-error').hidden = true;
    resetAnalyzeButton();
    if (faqItem) {
      $('t-subject').value = faqItem.summary || faqItem.question;
      $('t-desc').value = faqItem.description || '';
      $('template-name').textContent = faqItem.question;
      $('template-banner').hidden = false;
    } else {
      $('t-subject').value = state.ticket.summary || '';
      $('t-desc').value = state.ticket.description || '';
      $('template-banner').hidden = true;
    }
    updateCounters();
    showView('compose');
  }

  function updateCounters() {
    $('subject-count').textContent = String($('t-subject').value.length);
    $('desc-count').textContent = String($('t-desc').value.length);
  }

  function resetAnalyzeButton() {
    setBusy('analyze-btn', false);
    $('analyze-btn').querySelector('.btn-label').textContent = 'Analyze & continue';
  }

  function setBusy(btnId, busy) {
    const btn = $(btnId);
    btn.disabled = busy;
    btn.querySelector('.btn-spinner').hidden = !busy;
  }

  function initCompose() {
    $('t-subject').addEventListener('input', updateCounters);
    $('t-desc').addEventListener('input', () => {
      updateCounters();
      state.placeholderWarned = false;
      resetAnalyzeButton();
      $('compose-error').hidden = true;
    });
    $('analyze-btn').addEventListener('click', onAnalyze);
    $('t-subject').addEventListener('keydown', (e) => {
      if (e.key === 'Enter') { e.preventDefault(); onAnalyze(); }
    });
  }

  function onAnalyze() {
    const summary = $('t-subject').value.trim();
    const description = $('t-desc').value.trim();
    const err = $('compose-error');
    if (!summary && !description) {
      err.textContent = 'Please fill in the subject or the description before continuing.';
      err.hidden = false;
      return;
    }
    const leftovers = (summary + ' ' + description).match(/\[[A-Z_ ]{2,}\]/g);
    if (leftovers && !state.placeholderWarned) {
      state.placeholderWarned = true;
      err.textContent =
        `Your ticket still contains template placeholders (${leftovers.slice(0, 3).join(', ')}). ` +
        'Replace them with your details, or press the button again to continue anyway.';
      err.hidden = false;
      $('analyze-btn').querySelector('.btn-label').textContent = 'Analyze anyway';
      return;
    }
    err.hidden = true;
    state.ticket.summary = summary;
    state.ticket.description = description;
    startTriage();
  }

  // ---------- Triage (chat) ----------
  function addMsg(role, html, note) {
    const chat = $('chat');
    const msg = document.createElement('div');
    msg.className = `msg ${role}`;
    msg.innerHTML = `
      <span class="msg-avatar" aria-hidden="true">${role === 'bot' ? '🤖' : '👤'}</span>
      <div style="min-width:0">
        <div class="msg-bubble">${html}</div>
        ${note ? `<p class="msg-note">${esc(note)}</p>` : ''}
      </div>`;
    chat.appendChild(msg);
    chat.scrollTop = chat.scrollHeight;
    return msg;
  }

  function addTyping() {
    return addMsg('bot', '<span class="typing"><i></i><i></i><i></i></span>');
  }

  async function startTriage() {
    setBusy('analyze-btn', true);
    showView('triage');
    $('chat').replaceChildren();
    $('reply-bar').hidden = true;
    $('confirm-bar').hidden = true;
    state.sessionId = null;
    state.result = null;

    const { summary, description } = state.ticket;
    addMsg('user',
      `${summary ? `<b>${esc(summary)}</b>` : ''}${summary && description ? '<br><br>' : ''}${esc(description)}`);
    const typing = addTyping();
    try {
      const resp = await api('/classify/start', { summary, description });
      typing.remove();
      handleClassifyResponse(resp);
    } catch (e) {
      typing.remove();
      onTriageError(e, startTriage);
    } finally {
      setBusy('analyze-btn', false);
    }
  }

  async function sendReply(answer) {
    addMsg('user', esc(answer));
    $('reply-bar').hidden = true;
    const typing = addTyping();
    try {
      const resp = await api('/classify/answer', { session_id: state.sessionId, answer });
      typing.remove();
      handleClassifyResponse(resp);
    } catch (e) {
      typing.remove();
      onTriageError(e, () => { $('reply-bar').hidden = false; $('reply-input').focus(); });
    }
  }

  function onTriageError(e, retryFn) {
    if (e.code === 'llm_unavailable' || e.status === 503) {
      // حالتِ degraded: دسته‌بندی در دسترس نیست، ولی ثبتِ تیکت مستقل است —
      // با needs_review=true ثبت می‌شود و تیم پشتیبانی دستی مسیریابی می‌کند.
      state.result = { labels: {}, needs_review: true };
      addMsg('bot', `
        ⚠️ The smart assistant is temporarily unavailable, but <b>you can still submit
        your ticket now</b> — a support agent will route it manually.
        <div class="result-card">
          <div class="note-warn">⚠️ <b>Manual routing.</b> Your ticket will be reviewed
          and categorized by the support team.</div>
        </div>`);
      $('confirm-bar').hidden = false;
      $('submit-ticket-btn').focus();
      toast('Assistant unavailable — you can submit without classification.', retryFn);
      return;
    }
    addMsg('bot', `⚠️ ${esc("We couldn't reach the assistant. Please try again.")}`);
    toast(e.message || 'Network error', retryFn);
    $('confirm-bar').hidden = true;
  }

  function handleClassifyResponse(resp) {
    state.sessionId = resp.session_id;
    if (resp.status === 'need_info') {
      addMsg('bot', esc(resp.question || ''),
        `Quick question ${(resp.questions_asked || 0) + 1} of 2 — this helps route your ticket correctly.`);
      $('reply-bar').hidden = false;
      $('reply-input').value = '';
      $('reply-input').focus();
      return;
    }
    state.result = resp.result || {};
    renderResultCard(state.result);
    $('confirm-bar').hidden = false;
    $('submit-ticket-btn').focus();
  }

  function renderResultCard(result) {
    const labels = result.labels || {};
    const badges = Object.entries(LAYER_CAPTIONS).map(([layer, cap]) => `
      <div class="badge" style="background:${labelColor(labels[layer])}">
        <div class="cap">${cap}</div>
        <div class="val">${esc(labelName(labels[layer]))}</div>
      </div>`).join('');
    const note = result.needs_review
      ? `<div class="note-warn">⚠️ <b>Will be double-checked.</b> The assistant wasn't fully sure,
           so a support agent will verify the routing after you submit. You can still submit now.</div>`
      : `<div class="note-ok">✅ Classified with high confidence — ready to submit.</div>`;
    addMsg('bot', `
      Here's how your ticket will be routed:
      <div class="result-card">
        <div class="badge-row">${badges}</div>
        ${note}
      </div>`);
  }

  // ---------- Submit ----------
  async function submitTicket() {
    const id = state.identity;
    const result = state.result || {};
    setBusy('submit-ticket-btn', true);
    try {
      const record = await api('/api/tickets', {
        employee_id: id.employeeId,
        first_name: id.firstName,
        last_name: id.lastName,
        summary: state.ticket.summary || result.suggested_summary || '',
        description: state.ticket.description,
        labels: result.labels || {},
        needs_review: Boolean(result.needs_review),
        session_id: state.sessionId,
      });
      renderDone(record);
      showView('done');
    } catch (e) {
      toast(e.message || 'Could not submit the ticket.', submitTicket);
    } finally {
      setBusy('submit-ticket-btn', false);
    }
  }

  function renderDone(record) {
    $('ref-number').textContent = record.reference;
    $('done-subject').textContent = record.summary || '—';
    const labels = record.labels || {};
    $('done-routing').innerHTML = Object.entries(LAYER_CAPTIONS).map(([layer]) =>
      `<span class="routing-pill" style="background:${labelColor(labels[layer])}">${esc(labelName(labels[layer]))}</span>`
    ).join('');
    $('done-by').textContent =
      `${record.first_name} ${record.last_name} (ID ${record.employee_id})`;
    $('done-time').textContent = new Date(record.submitted_at).toLocaleString();
    $('done-review-note').hidden = !record.needs_review;
  }

  function initDone() {
    $('copy-ref').addEventListener('click', async () => {
      try {
        await navigator.clipboard.writeText($('ref-number').textContent);
        $('copy-ref').textContent = 'Copied ✓';
        setTimeout(() => { $('copy-ref').textContent = 'Copy'; }, 2000);
      } catch {
        toast('Copy failed — please copy the reference manually.');
      }
    });
  }

  // ---------- Global actions ----------
  function startOver() {
    state.ticket = { summary: '', description: '', templateName: null };
    state.sessionId = null;
    state.result = null;
    $('faq-search').value = '';
    state.query = '';
    $('search-clear').hidden = true;
    state.activeCategory = null;
    renderChips();
    renderFaqList();
    showView('home');
  }

  function initActions() {
    document.querySelectorAll('[data-action]').forEach((el) => {
      el.addEventListener('click', () => {
        const action = el.dataset.action;
        if (action === 'compose-blank') openCompose(null);
        else if (action === 'back-home') showView('home');
        else if (action === 'edit-ticket') openCompose(null);
        else if (action === 'start-over') startOver();
      });
    });
    $('reply-bar').addEventListener('submit', (e) => {
      e.preventDefault();
      const v = $('reply-input').value.trim();
      if (v) sendReply(v);
    });
    $('submit-ticket-btn').addEventListener('click', submitTicket);
    document.addEventListener('keydown', (e) => {
      if (e.key === 'Escape' && !$('view-compose').hidden) showView('home');
    });
  }

  // ---------- Boot ----------
  function boot() {
    initTheme();
    initLogo();
    initIdentityForm();
    initSearch();
    initCompose();
    initDone();
    initActions();
    state.identity = loadIdentity();
    renderUserChip();
    loadFaq();
    showView(state.identity ? 'home' : 'identity');
  }

  boot();
})();
