document.querySelectorAll('.kit-card').forEach((card) => {
  const kitId = card.dataset.kit;
  const button = card.querySelector('.kit-run');
  const status = card.querySelector('.kit-status');
  const result = card.querySelector('.kit-result');
  const verdict = card.querySelector('.kit-verdict');
  const checksUl = card.querySelector('.kit-checks');
  const metricsPre = card.querySelector('.kit-metrics pre');
  const logPre = card.querySelector('.kit-log pre');

  button.addEventListener('click', async () => {
    button.disabled = true;
    status.textContent = 'Running…';
    result.hidden = true;
    const t0 = performance.now();
    try {
      const res = await fetch(`/api/kit/${kitId}/run`, { method: 'POST' });
      const data = await res.json();
      renderReport(data);
    } catch (err) {
      verdict.className = 'kit-verdict fail';
      verdict.textContent = `Error: ${err.message}`;
      checksUl.innerHTML = '';
      result.hidden = false;
    } finally {
      button.disabled = false;
      const dt = ((performance.now() - t0) / 1000).toFixed(2);
      status.textContent = `Last run: ${dt}s`;
    }
  });

  function renderReport(rep) {
    verdict.className = `kit-verdict ${rep.passed ? 'pass' : 'fail'}`;
    verdict.textContent = `${rep.passed ? 'PASS' : 'FAIL'} · ${rep.pass_count}/${rep.total_count} checks · ${rep.duration_seconds.toFixed(2)}s`;
    checksUl.innerHTML = '';
    for (const c of rep.checks) {
      const li = document.createElement('li');
      const mark = document.createElement('span');
      mark.className = `check-mark ${c.passed ? 'ok' : 'no'}`;
      mark.textContent = c.passed ? '✓' : '✗';
      const body = document.createElement('div');
      body.innerHTML = `<div>${escapeHtml(c.name)}</div>` +
        (c.detail ? `<span class="check-detail">${escapeHtml(c.detail)}</span>` : '');
      li.appendChild(mark);
      li.appendChild(body);
      checksUl.appendChild(li);
    }
    metricsPre.textContent = JSON.stringify(rep.metrics, null, 2);
    logPre.textContent = (rep.log || []).join('\n') + (rep.error ? `\n\nERROR: ${rep.error}` : '');
    result.hidden = false;
  }

  function escapeHtml(s) {
    return String(s).replace(/[&<>"']/g, (c) => ({
      '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;',
    }[c]));
  }
});
