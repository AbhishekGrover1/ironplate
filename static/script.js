/* Ironplate front end: keeps state in the browser, talks to /api/chat, renders the replies. */
(() => {
  'use strict';

  const STORAGE_KEY = 'ironplate:v1';
  const MAX_MESSAGES = 40;
  const KEEP_DAYS = 120;
  const REQUEST_TIMEOUT_MS = 90000;
  const WAKE_NOTICE_MS = 6000;
  const ACTIVITY = { sedentary: 'sedentary', light: 'lightly active', moderate: 'moderately active', active: 'very active', very_active: 'extra active' };

  const $ = (sel, root = document) => root.querySelector(sel);
  const feed = $('#feed');
  const form = $('#composer');
  const input = $('#input');
  const sendBtn = $('#send');
  const todayBox = $('#today');
  const calm = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  const finePointer = window.matchMedia('(pointer: fine)').matches;

  /* ---------- state ---------- */
  const blank = () => ({ profile: {}, pending: null, planSeed: 0, targets: null, weights: [], log: {}, messages: [] });
  let state = loadState();
  let busy = false;

  function loadState() {
    try {
      const raw = JSON.parse(localStorage.getItem(STORAGE_KEY));
      if (raw && typeof raw === 'object') return { ...blank(), ...raw };
    } catch (_) { /* corrupt or unavailable storage */ }
    return blank();
  }

  function saveState() {
    try { localStorage.setItem(STORAGE_KEY, JSON.stringify(state)); } catch (_) { /* storage full or disabled */ }
  }

  const pad = (n) => String(n).padStart(2, '0');
  const todayKey = () => {
    const d = new Date();
    return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())}`;
  };
  const int = (n) => Math.round(n).toLocaleString('en-US');
  const gram = (n) => {
    const v = Math.round(n * 10) / 10;
    return v >= 10 || Number.isInteger(v) ? String(Math.round(v)) : String(v);
  };
  const cap = (s) => s.charAt(0).toUpperCase() + s.slice(1);
  const signed = (n, digits = 2) => `${n > 0 ? '+' : n < 0 ? '-' : ''}${Math.abs(n).toFixed(digits)}`;

  function dayTotals(key = todayKey()) {
    const totals = { kcal: 0, protein: 0, carbs: 0, fat: 0 };
    for (const e of state.log[key] || []) {
      totals.kcal += e.kcal; totals.protein += e.protein; totals.carbs += e.carbs; totals.fat += e.fat;
    }
    return totals;
  }

  function upsertWeight(date, kg) {
    const i = state.weights.findIndex((w) => w.date === date);
    if (i >= 0) state.weights[i].kg = kg; else state.weights.push({ date, kg });
    state.weights.sort((a, b) => (a.date < b.date ? -1 : 1));
    state.weights = state.weights.slice(-400);
  }

  function pruneLog() {
    const cutoff = new Date(Date.now() - KEEP_DAYS * 86400000);
    const key = `${cutoff.getFullYear()}-${pad(cutoff.getMonth() + 1)}-${pad(cutoff.getDate())}`;
    for (const day of Object.keys(state.log)) if (day < key) delete state.log[day];
  }

  function removeEntries(ids) {
    const gone = new Set(ids);
    for (const day of Object.keys(state.log)) {
      state.log[day] = state.log[day].filter((e) => !gone.has(e.id));
      if (!state.log[day].length) delete state.log[day];
    }
  }

  function payload(message) {
    const today = todayKey();
    const intake = Object.entries(state.log)
      .filter(([day]) => day !== today)
      .map(([date, rows]) => ({ date, kcal: rows.reduce((sum, r) => sum + r.kcal, 0) }))
      .slice(-60);
    return {
      message,
      state: {
        profile: state.profile, pending: state.pending, plan_seed: state.planSeed, today,
        hour: new Date().getHours(), log_today: state.log[today] || [], weights: state.weights.slice(-120), intake,
      },
    };
  }

  function applyResponse(res) {
    state.profile = res.profile || {};
    state.pending = res.pending || null;
    state.planSeed = res.plan_seed || 0;
    state.targets = res.targets || null;
    for (const action of res.actions || []) {
      if (action.type === 'log_food') (state.log[action.date] = state.log[action.date] || []).push(...action.entries);
      else if (action.type === 'log_weight') upsertWeight(action.date, action.kg);
      else if (action.type === 'reset_all') { state.weights = []; state.log = {}; state.profile = {}; state.targets = null; }
    }
    pruneLog();
  }

  /* ---------- DOM helpers ---------- */
  function h(tag, props, ...kids) {
    const el = document.createElement(tag);
    for (const [key, value] of Object.entries(props || {})) {
      if (value == null || value === false) continue;
      if (key === 'class') el.className = value;
      else if (key === 'text') el.textContent = value;
      else if (key.startsWith('on') && typeof value === 'function') el.addEventListener(key.slice(2), value);
      else el.setAttribute(key, value === true ? '' : value);
    }
    for (const kid of kids.flat()) {
      if (kid == null || kid === false) continue;
      el.append(kid.nodeType ? kid : document.createTextNode(String(kid)));
    }
    return el;
  }

  const panel = (...kids) => h('div', { class: 'panel glow' }, ...kids);
  const stat = (label, value, sub) => h('div', { class: 'stat' }, h('span', { text: label }), h('b', { text: value }), sub ? h('span', { text: sub }) : null);
  const end = (main, sub) => h('div', { class: 'cell-end' }, h('span', { class: 'num', text: main }), sub ? h('div', { class: 'sub', text: sub }) : null);

  function bar(label, valueText, pct, cls = '', over = false) {
    const fill = h('div', { class: `fill ${cls}${over ? ' over' : ''}` });
    fill.dataset.w = String(Math.max(0, Math.min(100, pct)));
    return h('div', {},
      h('div', { class: 'bar-top' }, h('span', { text: label }), h('span', { class: 'num', text: valueText })),
      h('div', { class: 'track' }, fill));
  }

  function animateFills(root) {
    const fills = root.querySelectorAll('.fill[data-w]');
    requestAnimationFrame(() => requestAnimationFrame(() => fills.forEach((f) => { f.style.width = `${f.dataset.w}%`; })));
  }

  function trendChart(points, goal) {
    const W = 320, H = 112, padX = 30, padY = 14, NS = 'http://www.w3.org/2000/svg';
    const svg = document.createElementNS(NS, 'svg');
    svg.setAttribute('viewBox', `0 0 ${W} ${H}`);
    svg.setAttribute('class', 'chart');
    svg.setAttribute('role', 'img');
    svg.setAttribute('aria-label', 'Weight trend');
    const times = points.map((p) => Date.parse(`${p.date}T00:00:00Z`));
    const t0 = times[0], t1 = times[times.length - 1];
    const values = points.flatMap((p) => [p.kg, p.trend]);
    const mean = values.reduce((a, b) => a + b, 0) / values.length;
    const showGoal = goal && Math.abs(goal - mean) < 6;
    if (showGoal) values.push(goal);
    let lo = Math.min(...values), hi = Math.max(...values);
    if (hi - lo < 1) { lo -= 0.5; hi += 0.5; }
    const x = (t) => padX + ((t - t0) / Math.max(t1 - t0, 1)) * (W - padX - 8);
    const y = (v) => padY + (1 - (v - lo) / (hi - lo)) * (H - padY * 2);
    const mk = (name, attrs) => {
      const node = document.createElementNS(NS, name);
      for (const [k, v] of Object.entries(attrs)) node.setAttribute(k, v);
      return node;
    };
    if (showGoal) svg.append(mk('line', { class: 'chart-goal', x1: padX, x2: W - 8, y1: y(goal), y2: y(goal) }));
    points.forEach((p, i) => svg.append(mk('circle', { class: 'chart-dot', cx: x(times[i]).toFixed(1), cy: y(p.kg).toFixed(1), r: 2.2 })));
    svg.append(mk('path', { class: 'chart-line', d: points.map((p, i) => `${i ? 'L' : 'M'}${x(times[i]).toFixed(1)} ${y(p.trend).toFixed(1)}`).join(' ') }));
    for (const [v, dy] of [[hi, 3], [lo, 3]]) {
      const label = mk('text', { class: 'chart-label', x: padX - 6, y: y(v) + dy, 'text-anchor': 'end' });
      label.textContent = v.toFixed(1);
      svg.append(label);
    }
    return svg;
  }

  /* ---------- block renderers ---------- */
  const R = {
    targets(b) {
      const energy = Math.max(b.protein_g * 4 + b.carbs_g * 4 + b.fat_g * 9, 1);
      const change = b.weekly_change_kg;
      return panel(
        h('div', { class: 'hero' },
          h('div', { class: 'big' }, int(b.calories), h('small', { text: 'kcal a day' })),
          h('div', { class: 'sub', text: change === 0 ? 'hold weight' : `${signed(change)} kg a week` })),
        h('div', { class: 'bars' },
          bar('Protein', `${b.protein_g} g`, (b.protein_g * 4 / energy) * 100, 'protein'),
          bar('Carbs', `${b.carbs_g} g`, (b.carbs_g * 4 / energy) * 100, 'carbs'),
          bar('Fat', `${b.fat_g} g`, (b.fat_g * 9 / energy) * 100, 'fat')),
        h('div', { class: 'facts' },
          h('div', { class: 'fact' }, h('span', { text: 'Maintenance' }), h('b', { text: `${int(b.tdee)} kcal` })),
          h('div', { class: 'fact' }, h('span', { text: 'Resting (BMR)' }), h('b', { text: `${int(b.bmr)} kcal` })),
          h('div', { class: 'fact' }, h('span', { text: 'BMI' }), h('b', { text: `${b.bmi} ${b.bmi_label}` })),
          h('div', { class: 'fact' }, h('span', { text: 'Fibre' }), h('b', { text: `${b.fiber_g} g` })),
          h('div', { class: 'fact' }, h('span', { text: 'Activity' }), h('b', { text: ACTIVITY[b.activity] || b.activity }))),
        ...(b.notes || []).map((n) => h('p', { class: 'note', text: n })));
    },

    meal_plan(b) {
      const kids = [];
      b.meals.forEach((m) => {
        kids.push(h('div', { class: 'meal-head' }, h('h3', { text: m.name }),
          h('span', { class: 'num', text: `${int(m.totals.kcal)} kcal, ${gram(m.totals.protein)} g protein` })));
        m.items.forEach((it) => kids.push(h('div', { class: 'row' },
          h('div', { class: 'name' }, it.name, h('div', { class: 'sub', text: it.portion })),
          end(`${int(it.kcal)} kcal`))));
      });
      kids.push(h('div', { class: 'stats' },
        stat('Calories', int(b.totals.kcal), `of ${int(b.targets.calories)}`),
        stat('Protein', `${gram(b.totals.protein)} g`, `of ${b.targets.protein_g}`),
        stat('Carbs', `${gram(b.totals.carbs)} g`, `of ${b.targets.carbs_g}`),
        stat('Fat', `${gram(b.totals.fat)} g`, `of ${b.targets.fat_g}`)));
      return panel(...kids);
    },

    workout(b) {
      const kids = [h('div', { class: 'panel-title', text: b.title }), h('div', { class: 'sub', text: b.summary })];
      b.days.forEach((d, i) => {
        const details = h('details', { class: 'day' },
          h('summary', {}, h('span', { text: d.name }), h('span', { class: 'sub', text: `${d.exercises.length} exercises` })));
        if (i === 0) details.open = true;
        d.exercises.forEach((ex) => {
          const row = h('div', { class: 'row lift', role: 'button', tabindex: '0', 'aria-expanded': 'false' },
            h('div', { class: 'name' }, ex.name, h('span', { class: 'tag', text: ex.muscle }),
              h('div', { class: 'cue', text: `${ex.cue} Rest: ${ex.rest}.` })),
            end(ex.sets === 1 ? ex.reps : `${ex.sets} x ${ex.reps}`));
          const toggle = () => { row.classList.toggle('open'); row.setAttribute('aria-expanded', String(row.classList.contains('open'))); };
          row.addEventListener('click', toggle);
          row.addEventListener('keydown', (e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); toggle(); } });
          details.append(row);
        });
        kids.push(details);
      });
      const sets = Object.entries(b.weekly_sets || {});
      if (sets.length) kids.push(h('p', { class: 'note', text: 'Weekly sets: ' + sets.map(([m, n]) => `${m} ${n}`).join(', ') + '.' }));
      (b.notes || []).forEach((n) => kids.push(h('p', { class: 'note', text: n })));
      return panel(...kids);
    },

    foods(b) {
      const kids = [];
      if (b.meal) {
        kids.push(h('div', { class: 'meal-head' }, h('h3', { text: cap(b.meal) }),
          h('span', { class: 'num', text: `${int(b.totals.kcal)} kcal` })));
      }
      b.items.forEach((it) => kids.push(h('div', { class: 'row' },
        h('div', { class: 'name' }, it.name,
          h('div', { class: 'sub', text: it.portion })),
        end(`${int(it.kcal)} kcal`, `${gram(it.protein)} g protein`))));
      if (b.items.length > 1) {
        kids.push(h('div', { class: 'row total' }, h('span', { text: 'Total' }),
          end(`${int(b.totals.kcal)} kcal`, `${gram(b.totals.protein)} g protein, ${gram(b.totals.carbs)} g carbs, ${gram(b.totals.fat)} g fat`)));
      }
      if (b.entry_ids && b.entry_ids.length) {
        const undo = h('button', { class: 'btn ghost glow tilt undo', type: 'button', text: b.undone ? 'Removed' : 'Undo', disabled: b.undone });
        undo.addEventListener('click', () => {
          removeEntries(b.entry_ids);
          b.undone = true;
          undo.textContent = 'Removed';
          undo.disabled = true;
          saveState();
          renderToday();
        });
        kids.push(undo);
      }
      return panel(...kids);
    },

    food_info(b) {
      return panel(
        h('div', { class: 'panel-title', text: b.name }),
        h('div', { class: 'sub', text: 'Per 100 g' }),
        h('div', { class: 'stats' },
          stat('Calories', int(b.per100.kcal)), stat('Protein', `${gram(b.per100.protein)} g`),
          stat('Carbs', `${gram(b.per100.carbs)} g`), stat('Fat', `${gram(b.per100.fat)} g`)),
        h('p', { class: 'note', text: `One serving: ${b.serving.portion}, ${int(b.serving.kcal)} kcal, ${gram(b.serving.protein)} g protein, ${gram(b.serving.fiber)} g fibre.` }),
        !b.exact && b.query ? h('p', { class: 'note', text: `Closest match for "${b.query}".` }) : null);
    },

    foods_list(b) {
      return panel(
        h('div', { class: 'panel-title', text: b.title }),
        ...b.items.map((it) => h('div', { class: 'row' },
          h('div', { class: 'name' }, it.name, h('div', { class: 'sub', text: it.note })),
          end(`${int(it.kcal)} kcal`, it.portion))));
    },

    trend(b) {
      const kids = [
        h('div', { class: 'hero' },
          h('div', { class: 'big' }, gram(b.trend_kg), h('small', { text: 'kg trend' })),
          h('div', { class: 'sub', text: `${signed(b.weekly_rate_kg)} kg a week` })),
        trendChart(b.points, b.goal_weight_kg),
      ];
      const facts = [];
      if (b.goal_weight_kg) facts.push(h('div', { class: 'fact' }, h('span', { text: 'Goal weight' }), h('b', { text: `${gram(b.goal_weight_kg)} kg` })));
      if (b.eta_weeks) facts.push(h('div', { class: 'fact' }, h('span', { text: 'Estimated time' }), h('b', { text: `${b.eta_weeks} weeks` })));
      if (b.adaptive) facts.push(h('div', { class: 'fact' }, h('span', { text: 'Real maintenance' }), h('b', { text: `${int(b.adaptive.tdee_est)} kcal` })));
      if (facts.length) kids.push(h('div', { class: 'facts' }, ...facts));
      return panel(...kids);
    },

    today(b) {
      const kids = [];
      if (b.targets) {
        const t = b.targets, s = b.totals;
        kids.push(h('div', { class: 'bars' },
          bar('Calories', `${int(s.kcal)} of ${int(t.calories)}`, (s.kcal / t.calories) * 100, '', s.kcal > t.calories * 1.05),
          bar('Protein', `${gram(s.protein)} of ${t.protein_g} g`, (s.protein / t.protein_g) * 100, 'protein'),
          bar('Carbs', `${gram(s.carbs)} of ${t.carbs_g} g`, (s.carbs / t.carbs_g) * 100, 'carbs'),
          bar('Fat', `${gram(s.fat)} of ${t.fat_g} g`, (s.fat / t.fat_g) * 100, 'fat')));
      }
      b.meals.forEach((m) => {
        kids.push(h('div', { class: 'meal-head' }, h('h3', { text: m.name }), h('span', { class: 'num', text: `${int(m.kcal)} kcal` })));
        m.items.forEach((it) => kids.push(h('div', { class: 'row' }, h('div', { class: 'name', text: it.name }), end(`${int(it.kcal)} kcal`, `${it.grams} g`))));
      });
      return panel(...kids);
    },

    exercise(b) {
      return panel(
        h('div', { class: 'panel-title' }, b.name, h('span', { class: 'tag', text: b.primary })),
        b.secondary && b.secondary.length ? h('div', { class: 'sub', text: `Also works: ${b.secondary.join(', ')}` }) : null,
        b.alternatives && b.alternatives.length
          ? h('div', { class: 'chips' }, ...b.alternatives.map((a) => h('button', { class: 'chip glow tilt', type: 'button', text: a, onclick: () => send(`How to do ${a}`) })))
          : null);
    },

    exercise_list(b) {
      return panel(
        h('div', { class: 'panel-title', text: b.title }),
        ...b.items.map((it) => h('div', { class: 'row' },
          h('div', { class: 'name' }, it.name, h('span', { class: 'tag', text: it.note }), h('div', { class: 'sub', text: `${it.cue.split('. ')[0].replace(/\.$/, '')}.` })))));
    },
  };

  /* ---------- messages ---------- */
  const userEl = (text) => h('div', { class: 'msg user' }, h('div', { class: 'bubble', text }));

  function chipsEl(list) {
    if (!list || !list.length) return null;
    return h('div', { class: 'chips' }, ...list.map((text) => h('button', { class: 'chip glow tilt', type: 'button', text, onclick: () => send(text) })));
  }

  function botEl(message, withChips) {
    const el = h('div', { class: 'msg bot' });
    String(message.reply || '').split('\n').filter(Boolean).forEach((line) => el.append(h('p', { class: 'text', text: line })));
    for (const block of message.blocks || []) {
      const render = R[block.type];
      if (render) el.append(render(block));
    }
    if (withChips) el.append(chipsEl(message.chips));
    return el;
  }

  const starterChips = () => (state.targets
    ? ['Plan my meals', 'Workout plan', "What's left today", 'I had 2 eggs for breakfast']
    : ['Calculate my calories', 'Plan my meals', 'Workout plan', 'I had 2 eggs for breakfast']);

  function showIntro() {
    const el = h('div', { class: 'msg bot', id: 'intro' },
      h('p', { class: 'text', text: 'Tell me about yourself, or ask about food and training.' }),
      chipsEl(starterChips()));
    feed.append(el);
  }

  function renderAll() {
    feed.textContent = '';
    if (!state.messages.length) { showIntro(); renderToday(); return; }
    state.messages.forEach((m, i) => {
      feed.append(m.role === 'user' ? userEl(m.text) : botEl(m, i === state.messages.length - 1));
    });
    animateFills(feed);
    feed.scrollTop = feed.scrollHeight;
    renderToday();
  }

  function renderToday() {
    const t = state.targets;
    todayBox.hidden = !t;
    if (!t) return;
    const totals = dayTotals();
    const goals = { kcal: t.calories, protein: t.protein_g, carbs: t.carbs_g, fat: t.fat_g };
    todayBox.querySelectorAll('.meter').forEach((meter) => {
      const key = meter.dataset.key, eaten = totals[key], goal = goals[key];
      meter.querySelector('[data-role="value"]').textContent = key === 'kcal' ? `${int(eaten)} / ${int(goal)}` : `${Math.round(eaten)} / ${goal} g`;
      const fill = meter.querySelector('.fill');
      fill.style.width = `${Math.min(100, goal ? (eaten / goal) * 100 : 0)}%`;
      fill.classList.toggle('over', goal > 0 && eaten > goal * 1.05);
    });
  }

  function scrollTo(el) {
    const behavior = calm ? 'auto' : 'smooth';
    if (el.offsetHeight > feed.clientHeight * 0.6) {
      const top = el.getBoundingClientRect().top - feed.getBoundingClientRect().top + feed.scrollTop - 8;
      feed.scrollTo({ top, behavior });
    } else {
      feed.scrollTo({ top: feed.scrollHeight, behavior });
    }
  }

  function typingEl() {
    return h('div', { class: 'msg bot' }, h('div', { class: 'typing', 'aria-label': 'Thinking' }, h('i'), h('i'), h('i')));
  }

  function setBusy(on) {
    sendBtn.disabled = on;
  }

  async function send(raw) {
    const text = String(raw || '').trim();
    if (!text || busy) return;
    busy = true;
    setBusy(true);
    feed.querySelectorAll('.chips').forEach((c) => { if (!c.closest('.panel')) c.remove(); });
    const intro = $('#intro');
    if (intro) intro.remove();

    state.messages.push({ role: 'user', text });
    feed.append(userEl(text));
    const typing = typingEl();
    feed.append(typing);
    feed.scrollTo({ top: feed.scrollHeight, behavior: calm ? 'auto' : 'smooth' });
    input.value = '';
    autosize();

    const wake = setTimeout(() => typing.append(h('div', { class: 'waking', text: 'Waking the server up, one moment.' })), WAKE_NOTICE_MS);
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS);
    try {
      const res = await fetch('/api/chat', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload(text)), signal: controller.signal,
      });
      if (!res.ok) throw Object.assign(new Error('request failed'), { status: res.status });
      const data = await res.json();
      applyResponse(data);
      const message = { role: 'bot', reply: data.reply, blocks: data.blocks || [], chips: data.chips || [] };
      state.messages.push(message);
      state.messages = state.messages.slice(-MAX_MESSAGES);
      typing.remove();
      const el = botEl(message, true);
      feed.append(el);
      animateFills(el);
      scrollTo(el);
    } catch (err) {
      typing.remove();
      const reply = err.status === 429 ? 'You are sending messages quickly. Give it a moment and try again.'
        : err.name === 'AbortError' ? 'That took too long. Please try again.'
        : 'I could not reach the server. Check your connection and try again.';
      const el = botEl({ reply, blocks: [], chips: [] }, false);
      feed.append(el);
      scrollTo(el);
    } finally {
      clearTimeout(wake);
      clearTimeout(timer);
      busy = false;
      setBusy(false);
      saveState();
      renderToday();
      if (finePointer) input.focus();
    }
  }

  /* ---------- input ---------- */
  function autosize() {
    input.style.height = 'auto';
    input.style.height = `${Math.min(input.scrollHeight, 140)}px`;
  }

  form.addEventListener('submit', (e) => { e.preventDefault(); send(input.value); });
  input.addEventListener('input', autosize);
  input.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey && !e.isComposing) { e.preventDefault(); send(input.value); }
  });
  $('#new-chat').addEventListener('click', () => {
    state.messages = [];
    state.pending = null;
    saveState();
    renderAll();
    input.focus();
  });

  /* ---------- pointer glow and tilt ---------- */
  document.addEventListener('pointermove', (e) => {
    if (e.pointerType !== 'mouse' || !(e.target instanceof Element)) return;
    const el = e.target.closest('.glow');
    if (!el) return;
    const r = el.getBoundingClientRect();
    el.style.setProperty('--mx', `${e.clientX - r.left}px`);
    el.style.setProperty('--my', `${e.clientY - r.top}px`);
    if (!calm && el.classList.contains('tilt')) {
      const px = (e.clientX - r.left) / r.width - 0.5;
      const py = (e.clientY - r.top) / r.height - 0.5;
      el.style.transform = `perspective(500px) rotateX(${(-py * 8).toFixed(2)}deg) rotateY(${(px * 10).toFixed(2)}deg)`;
    }
  }, { passive: true });
  document.addEventListener('pointerout', (e) => {
    if (!(e.target instanceof Element)) return;
    const el = e.target.closest('.tilt');
    if (el && !el.contains(e.relatedTarget)) el.style.transform = '';
  });

  renderAll();
})();
