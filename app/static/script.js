const form = document.getElementById('ask-form');
const qEl = document.getElementById('question');
const nsEl = document.getElementById('namespace');
const nameEl = document.getElementById('name');
const containerEl = document.getElementById('container');
const tailEl = document.getElementById('tail');
const explainEl = document.getElementById('explain');

const resCard = document.getElementById('result');
const summaryEl = document.getElementById('summary');
const suggestionsEl = document.getElementById('suggestions');
const kubectlEl = document.getElementById('kubectl');
const stdoutEl = document.getElementById('stdout');
const stderrEl = document.getElementById('stderr');

function pretty(obj) {
  try { return JSON.stringify(obj, null, 2); } catch { return String(obj); }
}

form.addEventListener('submit', async (e) => {
  e.preventDefault();
  const payload = {
    question: qEl.value,
    namespace: nsEl.value || undefined,
    name: nameEl.value || undefined,
    container: containerEl.value || undefined,
    tail_lines: tailEl.value ? parseInt(tailEl.value, 10) : undefined,
    explain: explainEl.checked,
  };

  try {
    const resp = await fetch('/ask', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    const data = await resp.json();
    resCard.classList.remove('hidden');

    summaryEl.textContent = data.summary || '';
    suggestionsEl.innerHTML = '';
    (data.suggestions || []).forEach(s => {
      const li = document.createElement('li');
      li.textContent = s;
      suggestionsEl.appendChild(li);
    });

    kubectlEl.textContent = (data.kubectl || []).join(' ');
    stdoutEl.textContent = typeof data.stdout === 'string' ? data.stdout : pretty(data.stdout);
    stderrEl.textContent = data.stderr || '';

    // Clear the form for the next request
    qEl.value = '';
    nsEl.value = '';
    nameEl.value = '';
    containerEl.value = '';
    tailEl.value = '';

  } catch (err) {
    resCard.classList.remove('hidden');
    summaryEl.textContent = '';
    suggestionsEl.innerHTML = '';
    kubectlEl.textContent = '';
    stdoutEl.textContent = '';
    stderrEl.textContent = String(err);
  }
});
