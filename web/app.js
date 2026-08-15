/* ============================================================
   Service Desk — Ticket Assistant (SPA)
   Chat-first flow: bot greets → collects subject → collects
   description → classifies via backend.
   ============================================================ */
(() => {
  'use strict';
0000000000
  // ---------- Label metadata ----------
  const LAYER_CAPTIONS = { layer1: 'Type', layer2: 'Domain' };
  const LABELS = {
    incident:        { name: 'Incident',        color: 'var(--c-incident)' },
    service_request: { name: 'Service Request', color: 'var(--c-service-request)' },
    erp:             { name: 'ERP',             color: 'var(--c-erp)' },
    staff:           { name: 'Staff',           color: 'var(--c-staff)' },
  };
  const labelName  = (id) =>
    (LABELS[id] && LABELS[id].name) ||
    (id ? id.replace(/_/g, ' ').replace(/\b\w/g, (c) => c.toUpperCase()) : '—');
  const labelColor = (id) => (LABELS[id] && LABELS[id].color) || 'var(--text-3)';

  // ---------- Chat-flow state machine ----------
  // Stages: 'awaiting_subject' → 'awaiting_description' → 'classifying' → 'done'
  const chatFlow = {
    stage: 'awaiting_subject',
  };

  // ---------- App state ----------
  const LS_THEME = 'tta.theme';
  const state = {
    identity: { employeeId: '264790', firstName: 'Sina', lastName: 'BaniasadAzad' },
    faq: { categories: [], items: [] },
    activeCategory: null,
    query: '',
    ticket: { summary: '', description: '', templateName: null },
    sessionId: null,
    selectedFaqId: null,
    resultFeedback: null,
    result: null,
  };

  const $ = (id) => document.getElementById(id);
  const esc = (s) =>
    String(s ?? '').replace(/[&<>"']/g, (c) =>
      ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));

  // ---------- Normalization ----------
  const CHAR_MAP = { 'ي': 'ی', 'ك': 'ک', 'ة': 'ه', 'أ': 'ا', 'إ': 'ا', 'ؤ': 'و', '‌': ' ' };
  for (let i = 0; i < 10; i++) {
    CHAR_MAP[String.fromCharCode(0x06f0 + i)] = String(i);
    CHAR_MAP[String.fromCharCode(0x0660 + i)] = String(i);
  }
  const normalize = (s) =>
    String(s || '').replace(/./g, (c) => CHAR_MAP[c] ?? c).toLowerCase();

  // ---------- API ----------
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
    const prefersDark = window.matchMedia('(prefers-color-scheme: light)').matches;
    applyTheme(saved || 'light');
    $('theme-btn').addEventListener('click', () =>
      applyTheme(document.documentElement.dataset.theme === 'light' ? 'dark' : 'light'));
  }

  // ---------- Logo ----------
  function initLogo() {
    const img = new Image();
    img.alt = '';
    img.onload  = () => { $('brand-logo').replaceChildren(img); };
    img.onerror = () => { $('brand-logo').textContent = '🎫'; };
    img.src = '/api/logo';
  }

  // ---------- Views ----------
  function showView(name) {
    ['compose', 'done'].forEach((v) => {
      $(`view-${v}`).hidden = v !== name;
    });
    window.scrollTo({ top: 0 });
  }

  // ---------- Chat helpers ----------
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

  // ---------- Chat input ----------
  function setChatInputBusy(busy) {
    const input = $('chat-input');
    const btn   = $('chat-send');
    input.disabled = busy;
    btn.disabled   = busy;
    btn.querySelector('.btn-spinner').hidden = !busy;
  }

  function initChatInput() {
    const form  = $('chat-form');
    const input = $('chat-input');

    form.addEventListener('submit', (e) => {
      e.preventDefault();
      const value = input.value.trim();
      if (!value) return;
      input.value = '';
      autoResizeTextarea(input);
      handleChatInput(value);
    });

    // Ctrl/Cmd+Enter or Enter (no shift) sends; Shift+Enter inserts newline
    input.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        form.requestSubmit();
      }
    });

    input.addEventListener('input', () => autoResizeTextarea(input));
  }

  function autoResizeTextarea(el) {
    el.style.height = 'auto';
    el.style.height = Math.min(el.scrollHeight, 180) + 'px';
  }

  // ---------- Reset button visibility ----------
  function showResetButton() {
    $('chat-reset-btn').hidden = false;
  }
  function hideResetButton() {
    $('chat-reset-btn').hidden = true;
  }

  // ---------- Chat flow ----------
  async function handleChatInput(value) {
    addMsg('user', esc(value));
    showResetButton();      // once the user sends anything, show the reset button
    setChatInputBusy(true);

    if (chatFlow.stage === 'awaiting_subject') {
      state.ticket.summary = value;
      chatFlow.stage = 'awaiting_description';
      renderFaqList();      // re-render FAQ to dim items (stage changed)
      await simulateBotDelay(() =>
        addMsg('bot', 'لطفا توضیحات مشکل خود را بصورت واضح بیان کنید.')
      );
      setChatInputBusy(false);
      return;
    }

    if (chatFlow.stage === 'awaiting_description') {
      state.ticket.description = value;
      chatFlow.stage = 'classifying';
      await startClassification();
      return;
    }

    if (chatFlow.stage === 'need_info') {
      await sendTriageReply(value);
      return;
    }

    setChatInputBusy(false);
  }

  function simulateBotDelay(fn, ms = 600) {
    return new Promise((resolve) => {
      setTimeout(() => { fn(); resolve(); }, ms);
    });
  }

  function applyFaqTemplate(item) {
  const summary = item.summary || item.question || '';
  const description = item.description || '';

  state.ticket.summary = summary;
  state.ticket.description = description;
  state.ticket.templateName = item.id || null;
  state.selectedFaqId = item.id || null;

  const input = $('chat-input');

  if (chatFlow.stage === 'awaiting_subject') {
    addMsg('user', esc(summary));
    chatFlow.stage = 'awaiting_description';
    addMsg('bot', 'لطفا توضیحات مشکل خود را بصورت واضح بیان کنید');
  }

  input.value = description;
  autoResizeTextarea(input);
  input.focus();
}

  // ---------- Classification (triage) ----------
  async function startClassification() {
    const typing = addTyping();
    try {
      const resp = await api('/classify/start', {
        summary: state.ticket.summary,
        description: state.ticket.description,
      });
      typing.remove();
      handleClassifyResponse(resp);
    } catch (e) {
      typing.remove();
      onTriageError(e, startClassification);
    } finally {
      setChatInputBusy(false);
    }
  }

  async function sendTriageReply(answer) {
    const typing = addTyping();
    try {
      const resp = await api('/classify/answer', {
        session_id: state.sessionId,
        answer,
      });
      typing.remove();
      handleClassifyResponse(resp);
    } catch (e) {
      typing.remove();
      onTriageError(e, () => setChatInputBusy(false));
    } finally {
      setChatInputBusy(false);
    }
  }

  function onTriageError(e, retryFn) {
    if (e.code === 'llm_unavailable' || e.status === 503) {
      state.result = { labels: {}, needs_review: true };
      addMsg('bot', `
        ⚠️ The smart assistant is temporarily unavailable, but <b>you can still submit
        your ticket now</b> — a support agent will route it manually.
        <div class="result-card">
          <div class="note-warn">⚠️ <b>Manual routing.</b> Your ticket will be reviewed
          and categorized by the support team.</div>
        </div>`);
      $('confirm-bar').hidden = false;
      toast('Assistant unavailable — you can submit without classification.', retryFn);
      return;
    }
    addMsg('bot', `⚠️ ${esc("We couldn't reach the assistant. Please try again.")}`);
    toast(e.message || 'Network error', retryFn);
  }

  function handleClassifyResponse(resp) {
    state.sessionId = resp.session_id;
    if (resp.status === 'need_info') {
      chatFlow.stage = 'need_info';
      addMsg('bot', esc(resp.question || ''),
        `Quick question ${(resp.questions_asked || 0) + 1} of 2 — this helps route your ticket correctly.`);
      setChatInputBusy(false);
      return;
    }
    chatFlow.stage = 'classified';
    state.result = resp.result || {};
    renderResultCard(state.result);
    $('confirm-bar').hidden = false;
    $('submit-ticket-btn').focus();
    setChatInputBusy(false);
  }

  function renderResultCard(result) {
    const labels = result.labels || {};
    const type = labelName(labels.layer1);
    const domain = labelName(labels.layer2);

    const msg = addMsg('bot', `بر اساس توضیحات شما، من درخواست شما را به این شکل دسته‌بندی کردم:
            <span class="result-status">
      <div class="result-inline">
        <span class="result-pill result-pill-type" style="background:${labelColor(labels.layer1)}">
          <span class="result-key">Type:</span>
          <span class="result-value">${esc(type)}</span>
        </span>
        <span class="result-pill result-pill-domain" style="background:${labelColor(labels.layer2)}">
          <span class="result-key">Domain:</span>
          <span class="result-value">${esc(domain)}</span>
        </span>
      </div>
      <div class="feedback-row">
        <button class="feedback-mini" type="button" data-feedback="like" aria-label="Like this result" title="Like">
          <svg viewBox="0 0 24 24" aria-hidden="true">
            <path d="M7 11v9H4v-9zm3 9h7.2c1 0 1.8-.7 2-1.6l1.5-7.5c.1-.7 0-1.4-.4-1.9-.4-.6-1-.9-1.7-.9H13V5.5c0-1-.8-1.8-1.8-1.8-.5 0-1 .2-1.3.6L9.3 6.1c-.2.2-.3.5-.3.8v2.6c0 .7-.3 1.4-.8 1.9L10 20z" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linejoin="round"/>
          </svg>
        </button>
        <button class="feedback-mini" type="button" data-feedback="dislike" aria-label="Dislike this result" title="Dislike">
          <svg viewBox="0 0 24 24" aria-hidden="true">
            <path d="M17 13V4h3v9zm-3-9H6.8c-1 0-1.8.7-2 1.6L3.3 13.1c-.1.7 0 1.4.4 1.9.4.6 1 .9 1.7.9H11v2.6c0 1 .8 1.8 1.8 1.8.5 0 1-.2 1.3-.6l.6-.8c.2-.2.3-.5.3-.8v-2.6c0-.7.3-1.4.8-1.9L14 4z" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linejoin="round"/>
          </svg>
        </button>
      </div>
    `);

    msg.querySelectorAll('[data-feedback]').forEach((btn) => {
      btn.addEventListener('click', () => {
        state.resultFeedback = btn.dataset.feedback;
        msg.querySelectorAll('[data-feedback]').forEach((b) => {
          b.classList.toggle('active', b === btn);
        });
      });
    });
  }

  // ---------- Submit ----------
  async function submitTicket() {
    const id     = state.identity;
    const result = state.result || {};

    const btn = $('submit-ticket-btn');
    if (!btn) return;

    setBusy('submit-ticket-btn', true);
    try {
      const record = await api('/api/tickets', {
        employee_id:  id.employeeId,
        first_name:   id.firstName,
        last_name:    id.lastName,
        summary:      state.ticket.summary || result.suggested_summary || '',
        description:  state.ticket.description,
        labels:       result.labels || {},
        needs_review: Boolean(result.needs_review),
        session_id:   state.sessionId,
      });

      renderDone(record || {});
      showView('done');
    } catch (e) {
      console.error('submitTicket failed:', e);
      toast(e.message || 'Could not submit the ticket.', submitTicket);
    } finally {
      setBusy('submit-ticket-btn', false);
    }
  }

  function setBusy(btnId, busy) {
    const btn = $(btnId);
    btn.disabled = busy;
    btn.querySelector('.btn-spinner').hidden = !busy;
  }

  function renderDone(record) {
    $('ref-number').textContent  = record.reference;
    $('done-subject').textContent = record.summary || '—';
    const labels = record.labels || {};
    $('done-routing').innerHTML = Object.entries(LAYER_CAPTIONS).map(([layer]) =>
      `<span class="routing-pill" style="background:${labelColor(labels[layer])}">${esc(labelName(labels[layer]))}</span>`
    ).join('');
    $('done-by').textContent   = `${record.first_name} ${record.last_name} (ID ${record.employee_id})`;
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

  // ---------- Start over ----------
  function startOver() {
    state.ticket = { summary: '', description: '', templateName: null };
    state.selectedFaqId = null;
    state.sessionId = null;
    state.resultFeedback = null;
    state.result = null;
    chatFlow.stage = 'awaiting_subject';

    $('chat').replaceChildren();
    $('confirm-bar').hidden = true;
    $('chat-input').value = '';
    autoResizeTextarea($('chat-input'));
    setChatInputBusy(false);

    $('faq-search').value = '';
    state.query = '';
    $('search-clear').hidden = true;
    renderFaqList();

    showView('compose');
    botGreet();
  }

  function initActions() {
    document.querySelectorAll('[data-action]').forEach((el) => {
      el.addEventListener('click', () => {
        const action = el.dataset.action;
        if (action === 'start-over') startOver();
      });
    });
    $('submit-ticket-btn').addEventListener('click', submitTicket);
    $('chat-reset-btn').addEventListener('click', startOver);
  }

  // ---------- FAQ (sidebar, read-only hints) ----------
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
    renderFaqList();
  }

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
    let html = '', pos = 0;
    ranges.forEach(([s, e]) => {
      if (s < pos) { e > pos && (html += `<mark>${esc(text.slice(pos, e))}</mark>`, pos = e); return; }
      html += esc(text.slice(pos, s)) + `<mark>${esc(text.slice(s, e))}</mark>`;
      pos = e;
    });
    return html + esc(text.slice(pos));
  }

  function renderFaqList() {
    const terms = normalize(state.query).split(/\s+/).filter(Boolean);
    const items = terms.length === 0
      ? state.faq.items
      : state.faq.items.filter((it) => terms.every((t) => it.blob.includes(t)));

    const list = $('faq-list');
    list.replaceChildren();
    items.forEach((it) => {
      const li  = document.createElement('li');
      const btn = document.createElement('button');
      btn.type      = 'button';
      btn.className = 'faq-item';
      // Dim items that can't be used in the current stage
      btn.dataset.unavailable = chatFlow.stage !== 'awaiting_subject' ? 'true' : 'false';
      btn.innerHTML = `
        <span class="faq-item-main">
          <p class="faq-q" dir="auto">${highlight(it.question, terms)}</p>
        </span>
        <span class="faq-arrow">
          <svg viewBox="0 0 24 24" width="18" height="18" fill="none"
               stroke="currentColor" stroke-width="2" stroke-linecap="round">
            <path d="M5 12h14m-6-6 6 6-6 6"/>
          </svg>
        </span>`;
      btn.addEventListener('click', () => onFaqClick(it));
      li.appendChild(btn);
      list.appendChild(li);
    });
    $('faq-empty').hidden = items.length > 0;
    $('faq-count').textContent =
      items.length === 0 ? '' : `${items.length} common request${items.length > 1 ? 's' : ''}`;
  }

  /**
   * Called when the user clicks a FAQ item.
   *
   * Stage: awaiting_subject
   *   → Auto-send the summary as the user's subject message.
   *   → Bot replies asking for description.
   *   → Pre-fill the textarea with the FAQ description template so the
   *     user can edit it before sending.
   *
   * Any other stage: show a brief hint toast and do nothing else.
   */
  async function onFaqClick(faqItem) {
    if (chatFlow.stage !== 'awaiting_subject') {
      toast('Start a new conversation first to use an FAQ template.');
      return;
    }

    const subject = faqItem.summary || faqItem.question || '';

    state.ticket.summary = subject;
    state.ticket.description = faqItem.description || '';
    state.ticket.templateName = faqItem.id || null;
    state.selectedFaqId = faqItem.id || null;

    addMsg('user', esc(subject));
    showResetButton();
    chatFlow.stage = 'awaiting_description';
    renderFaqList();

    setChatInputBusy(true);
    await simulateBotDelay(() => {
      addMsg('bot', 'لطفا توضیحات مشکل خود را بصورت واضح بیان کنید.');
    });
    setChatInputBusy(false);

    const input = $('chat-input');
    input.value = faqItem.description || '';
    autoResizeTextarea(input);
    input.focus();

    const firstPlaceholder = input.value.indexOf('[');
    if (firstPlaceholder !== -1) {
      input.setSelectionRange(firstPlaceholder, firstPlaceholder);
    }
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
  }

  // ---------- Bot greeting ----------
  function botGreet() {
    setTimeout(() => {
      addMsg('bot',
        'سلام، من دستیار هوشمند سیستم Ticketing نقش اول کیفیت هستم.\n' +
        'لطفا موضوع مشکل خود را حداکثر در 5 کلمه بیان کنید.'
      );
    }, 300);
  }

  // ---------- Boot ----------
  function boot() {
    initTheme();
    initLogo();
    initSearch();
    initChatInput();
    initDone();
    initActions();

    loadFaq();
    showView('compose');
    botGreet();
  }

  boot();
})();
