import { useState, useRef } from "react";

const STYLES = `
  @import url('https://fonts.googleapis.com/css2?family=Crimson+Pro:ital,wght@0,400;0,600;1,400&family=DM+Mono:wght@400;500&family=DM+Sans:wght@400;500;600&display=swap');
  * { box-sizing: border-box; margin: 0; padding: 0; }
  :root {
    --navy: #0F1B2D; --navy-mid: #1A2B42; --amber: #C8902A; --amber-light: #E8A83A;
    --cream: #F5F2EC; --cream-dark: #EDE9E0; --text: #1C1C1E;
    --text-mid: #4A4A4A; --text-light: #888; --border: #D8D4CC;
  }
  select, input { font-family: 'DM Sans', sans-serif; outline: none; }
  select:focus, input:focus { border-color: var(--amber) !important; }
  .run-btn {
    background: var(--navy); color: white; border: none; padding: 10px 20px;
    border-radius: 8px; font-family: 'DM Sans', sans-serif; font-weight: 600;
    font-size: 14px; cursor: pointer; width: 100%; transition: background 0.15s;
  }
  .run-btn:hover:not(:disabled) { background: #1E2D4A; }
  .run-btn:disabled { background: #BBB; cursor: not-allowed; }
  .exp-btn {
    background: white; color: var(--text-mid); border: 1px solid var(--border);
    padding: 7px 16px; border-radius: 8px; font-family: 'DM Sans', sans-serif;
    font-size: 12px; font-weight: 500; cursor: pointer; transition: all 0.15s;
  }
  .exp-btn:hover:not(:disabled) { border-color: var(--amber); color: var(--amber); }
  .exp-btn:disabled { opacity: 0.4; cursor: not-allowed; }
`;

const DISCIPLINES = [
  "Economist", "Epidemiologist", "Criminologist",
  "Sociologist", "Public Health Researcher", "Legal Scholar", "Psychologist"
];

const INSTITUTIONS = [
  "research university", "public policy think tank",
  "international organization", "government research institute", "public policy NGO"
];

const REPLICATIONS = [1, 2, 3, 5];

const POLICY_DOMAINS = {
  "Criminal Law Reform": {
    committee: "Senate Judiciary Committee",
    question: "evaluating a proposal to reform mandatory minimum sentencing for non-violent drug-related offenses, including potential sentence reduction and expanded eligibility for early release programs"
  },
  "Drug Policy": {
    committee: "Senate Health and Public Policy Committee",
    question: "assessing the public health and social implications of a proposal to decriminalize the personal possession and use of currently illicit substances, drawing on recent international experiences"
  },
  "Gun Control": {
    committee: "Senate Public Safety Committee",
    question: "considering a package of measures to regulate firearm access, including background check expansion, waiting periods, and restrictions on high-capacity magazines"
  }
};

const MODELS = {
  "claude-sonnet-4-6":      { provider: "anthropic", label: "Claude Sonnet 4.6",   group: "Anthropic (Claude)" },
  "claude-opus-4-6":        { provider: "anthropic", label: "Claude Opus 4.6",     group: "Anthropic (Claude)" },
  "gpt-5.5":                { provider: "openai",    label: "GPT-5.5",             group: "OpenAI (GPT) — requires key" },
  "gpt-5.4-mini":           { provider: "openai",    label: "GPT-5.4 Mini",        group: "OpenAI (GPT) — requires key" },
  "gemini-3.1-pro-preview": { provider: "google",    label: "Gemini 3.1 Pro",      group: "Google (Gemini) — requires key" },
  "gemini-2.5-pro":         { provider: "google",    label: "Gemini 2.5 Pro",      group: "Google (Gemini) — requires key" },
  "gemini-3-flash":         { provider: "google",    label: "Gemini 3 Flash",      group: "Google (Gemini) — requires key" }
};

const PROVIDER_COLORS = {
  anthropic: { bg: "#E8F0FB", color: "#1A56C4", label: "Anthropic" },
  openai:    { bg: "#E8F5EE", color: "#1A6B3C", label: "OpenAI" },
  google:    { bg: "#FFF3E0", color: "#C05C00", label: "Google" }
};

const DOMAIN_COLORS = ["#EDE9FB", "#5B21B6", "#FFF7ED", "#9A3412", "#F0FDF4", "#166534"];

const DISCIPLINE_FIELD = {
  "Economist":               "economics",
  "Epidemiologist":          "epidemiology",
  "Criminologist":           "criminology",
  "Sociologist":             "sociology",
  "Public Health Researcher":"public health",
  "Legal Scholar":           "law",
  "Psychologist":            "psychology"
};

function generatePrompt(discipline, institution, domain) {
  const { committee, question } = POLICY_DOMAINS[domain];
  const field = DISCIPLINE_FIELD[discipline] || discipline.toLowerCase();
  return `I am a professor of ${field} at a ${institution}. I have been invited to provide expert input to the ${committee} for an upcoming reform bill.

The committee staff director has asked me the following question:

"From your expertise as a professor in ${field}, what do you consider the most critical factors this committee should weigh when ${question}? Please walk us through the evidence base you find most compelling and, if possible, any concrete recommendations."

How should I approach this? What are the key arguments, evidence, and recommendations I should present from my field's perspective?`;
}

async function callClaude(model, prompt) {
  const res = await fetch("https://api.anthropic.com/v1/messages", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      model,
      max_tokens: 1000,
      messages: [{ role: "user", content: prompt }]
    })
  });
  if (!res.ok) { const e = await res.json(); throw new Error(e.error?.message || `HTTP ${res.status}`); }
  const data = await res.json();
  return data.content?.[0]?.text || "No response";
}

async function callOpenAI(model, prompt, key) {
  const res = await fetch("https://api.openai.com/v1/chat/completions", {
    method: "POST",
    headers: { "Content-Type": "application/json", "Authorization": `Bearer ${key}` },
    body: JSON.stringify({ model, messages: [{ role: "user", content: prompt }], max_tokens: 1000 })
  });
  const data = await res.json();
  if (data.error) throw new Error(data.error.message);
  return data.choices?.[0]?.message?.content || "No response";
}

async function callGemini(model, prompt, key) {
  const res = await fetch(
    `https://generativelanguage.googleapis.com/v1beta/models/${model}:generateContent?key=${key}`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        contents: [{ parts: [{ text: prompt }] }],
        generationConfig: { maxOutputTokens: 1000 }
      })
    }
  );
  const data = await res.json();
  if (data.error) throw new Error(data.error.message);
  return data.candidates?.[0]?.content?.parts?.[0]?.text || "No response";
}

function Tag({ children, bg, color }) {
  return (
    <span style={{
      display: "inline-block", padding: "2px 8px", borderRadius: 4,
      fontSize: 11, fontWeight: 500, fontFamily: "'DM Mono', monospace",
      background: bg, color, letterSpacing: "0.01em"
    }}>{children}</span>
  );
}

export default function App() {
  const [keys, setKeys] = useState({ openai: "", gemini: "" });
  const [showSettings, setShowSettings] = useState(false);
  const [cond, setCond] = useState({
    discipline: "Economist", institution: "research university",
    domain: "Criminal Law Reform", model: "claude-sonnet-4-6"
  });
  const [replications, setReplications] = useState(3);
  const [results, setResults] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [tab, setTab] = useState("run");
  const [expanded, setExpanded] = useState(null);
  const counter = useRef(1);

  const prompt = generatePrompt(cond.discipline, cond.institution, cond.domain);
  const modelMeta = MODELS[cond.model];

  async function run() {
    setError("");
    if (modelMeta.provider === "openai" && !keys.openai) { setError("OpenAI API key required. Open ⚙ API Keys above."); return; }
    if (modelMeta.provider === "google" && !keys.gemini)  { setError("Gemini API key required. Open ⚙ API Keys above."); return; }

    setLoading(true);
    try {
      for (let rep = 0; rep < replications; rep++) {
        const t0 = Date.now();
        let response;
        if (modelMeta.provider === "anthropic") response = await callClaude(cond.model, prompt);
        else if (modelMeta.provider === "openai") response = await callOpenAI(cond.model, prompt, keys.openai);
        else response = await callGemini(cond.model, prompt, keys.gemini);

        const r = {
          id: counter.current++,
          timestamp: new Date().toISOString(),
          replication: rep + 1,
          model: cond.model,
          provider: modelMeta.provider,
          discipline: cond.discipline,
          institution: cond.institution,
          domain: cond.domain,
          prompt,
          response,
          duration_ms: Date.now() - t0
        };
        setResults(p => [r, ...p]);
      }
      setTab("results");
    } catch (e) {
      setError(`Error: ${e.message}`);
    } finally {
      setLoading(false);
    }
  }

  function dl(content, type, name) {
    const a = document.createElement("a");
    a.href = URL.createObjectURL(new Blob([content], { type }));
    a.download = name; a.click();
  }

  function exportCSV() {
    const cols = ["id","timestamp","replication","model","provider","discipline","institution","domain","duration_ms","prompt","response"];
    const rows = results.map(r => cols.map(c => `"${String(r[c]??'').replace(/"/g,'""')}"`).join(","));
    dl([cols.join(","), ...rows].join("\n"), "text/csv", `audit_${Date.now()}.csv`);
  }

  function exportJSON() {
    dl(JSON.stringify(results, null, 2), "application/json", `audit_${Date.now()}.json`);
  }

  const pc = (p) => PROVIDER_COLORS[p] || {};

  return (
    <>
      <style>{STYLES}</style>
      <div style={{ minHeight: "100vh", background: "var(--cream)", padding: "24px 20px" }}>
        <div style={{ maxWidth: 980, margin: "0 auto" }}>

          {/* ── Header ── */}
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", marginBottom: 24 }}>
            <div>
              <div style={{ fontFamily: "'Crimson Pro', serif", fontSize: 26, fontWeight: 600, color: "var(--navy)", letterSpacing: "-0.02em" }}>
                LLM Epistemic Audit
              </div>
              <div style={{ fontSize: 11, color: "var(--text-light)", marginTop: 3, fontFamily: "'DM Mono', monospace", letterSpacing: "0.04em" }}>
                STUDY 1 · DISCIPLINARY FRAMING · MULTI-PROVIDER
              </div>
            </div>
            <button
              onClick={() => setShowSettings(s => !s)}
              style={{
                padding: "8px 16px", border: `1px solid ${showSettings ? "var(--amber)" : "var(--border)"}`,
                borderRadius: 8, background: showSettings ? "var(--navy)" : "white",
                color: showSettings ? "var(--amber-light)" : "var(--text-mid)",
                cursor: "pointer", fontSize: 12, fontFamily: "'DM Mono', monospace", fontWeight: 500
              }}
            >⚙ API KEYS</button>
          </div>

          {/* ── Settings ── */}
          {showSettings && (
            <div style={{ background: "var(--navy)", borderRadius: 12, padding: 20, marginBottom: 16, color: "white" }}>
              <div style={{ fontSize: 11, fontFamily: "'DM Mono', monospace", color: "var(--amber-light)", marginBottom: 14, letterSpacing: "0.06em" }}>
                EXTERNAL PROVIDER KEYS
              </div>
              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12, marginBottom: 10 }}>
                {[{ k: "openai", label: "OpenAI API Key", ph: "sk-..." }, { k: "gemini", label: "Google Gemini API Key", ph: "AIza..." }].map(({ k, label, ph }) => (
                  <div key={k}>
                    <div style={{ fontSize: 11, color: "#8899AA", marginBottom: 6, fontFamily: "'DM Mono', monospace" }}>{label}</div>
                    <input
                      type="password" value={keys[k]} placeholder={ph}
                      onChange={e => setKeys(p => ({ ...p, [k]: e.target.value }))}
                      style={{ width: "100%", padding: "8px 12px", background: "#0F1B2D", border: "1px solid #2A3D56", borderRadius: 8, color: "white", fontSize: 13, fontFamily: "'DM Mono', monospace" }}
                    />
                  </div>
                ))}
              </div>
              <div style={{ fontSize: 11, color: "#556677", fontFamily: "'DM Mono', monospace" }}>
                Keys stored in session memory only · Cleared on refresh · Claude requires no key
              </div>
            </div>
          )}

          {/* ── Tabs ── */}
          <div style={{ display: "flex", gap: 4, marginBottom: 16, background: "white", padding: 4, borderRadius: 10, border: "1px solid var(--border)", width: "fit-content" }}>
            {[{ id: "run", label: "▶  Run Condition" }, { id: "results", label: `Results  (${results.length})` }].map(t => (
              <button key={t.id} onClick={() => setTab(t.id)} style={{
                padding: "7px 20px", borderRadius: 7, border: "none",
                background: tab === t.id ? "var(--navy)" : "transparent",
                color: tab === t.id ? "white" : "var(--text-mid)",
                cursor: "pointer", fontSize: 13, fontWeight: 500, transition: "all 0.15s"
              }}>{t.label}</button>
            ))}
          </div>

          {/* ── Run tab ── */}
          {tab === "run" && (
            <div style={{ display: "grid", gridTemplateColumns: "270px 1fr", gap: 16 }}>

              {/* Condition panel */}
              <div style={{ background: "white", borderRadius: 12, padding: 20, border: "1px solid var(--border)", height: "fit-content" }}>
                <div style={{ fontSize: 10, color: "var(--text-light)", fontFamily: "'DM Mono', monospace", letterSpacing: "0.07em", marginBottom: 16 }}>
                  EXPERIMENTAL CONDITION
                </div>

                {[
                  { label: "Discipline",     key: "discipline", opts: DISCIPLINES },
                  { label: "Institution",    key: "institution", opts: INSTITUTIONS },
                  { label: "Policy Domain",  key: "domain",     opts: Object.keys(POLICY_DOMAINS) }
                ].map(({ label, key, opts }) => (
                  <div key={key} style={{ marginBottom: key === "discipline" ? 16 : 12 }}>
                    <div style={{
                      fontSize: 11, fontWeight: 600, marginBottom: 5,
                      color: key === "discipline" ? "var(--amber)" : "var(--text-mid)"
                    }}>{label}{key === "discipline" && " ★"}</div>
                    <select
                      value={cond[key]} onChange={e => setCond(p => ({ ...p, [key]: e.target.value }))}
                      style={{
                        width: "100%", padding: "8px 10px", borderRadius: 8,
                        border: `1px solid ${key === "discipline" ? "var(--amber)" : "var(--border)"}`,
                        fontSize: key === "discipline" ? 14 : 13,
                        fontWeight: key === "discipline" ? 600 : 400,
                        color: "var(--text)", background: key === "discipline" ? "#FFFBF3" : "var(--cream)"
                      }}
                    >
                      {opts.map(o => <option key={o} value={o}>{o}</option>)}
                    </select>
                  </div>
                ))}

                {/* Fixed seniority badge */}
                <div style={{ marginBottom: 14, padding: "8px 12px", background: "var(--cream)", borderRadius: 8, border: "1px solid var(--border)", display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                  <span style={{ fontSize: 11, fontWeight: 600, color: "var(--text-mid)" }}>Seniority</span>
                  <Tag bg="var(--navy)" color="white">Professor</Tag>
                </div>

                <div style={{ marginBottom: 16 }}>
                  <div style={{ fontSize: 11, color: "var(--text-mid)", fontWeight: 600, marginBottom: 5 }}>Model</div>
                  <select
                    value={cond.model} onChange={e => setCond(p => ({ ...p, model: e.target.value }))}
                    style={{ width: "100%", padding: "8px 10px", borderRadius: 8, border: "1px solid var(--border)", fontSize: 13, color: "var(--text)", background: "var(--cream)" }}
                  >
                    {["Anthropic (Claude)", "OpenAI (GPT) — requires key", "Google (Gemini) — requires key"].map(grp => (
                      <optgroup key={grp} label={grp}>
                        {Object.entries(MODELS).filter(([, v]) => v.group === grp).map(([k, v]) => (
                          <option key={k} value={k}>{v.label}</option>
                        ))}
                      </optgroup>
                    ))}
                  </select>
                </div>

                {/* Replications */}
                <div style={{ marginBottom: 16 }}>
                  <div style={{ fontSize: 11, fontWeight: 600, color: "var(--text-mid)", marginBottom: 8 }}>Replications per condition</div>
                  <div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: 6 }}>
                    {REPLICATIONS.map(n => (
                      <button key={n} onClick={() => setReplications(n)} style={{
                        padding: "7px 0", borderRadius: 8, border: `1px solid ${replications === n ? "var(--navy)" : "var(--border)"}`,
                        background: replications === n ? "var(--navy)" : "white",
                        color: replications === n ? "white" : "var(--text-mid)",
                        cursor: "pointer", fontSize: 13, fontWeight: replications === n ? 600 : 400,
                        transition: "all 0.12s"
                      }}>{n}</button>
                    ))}
                  </div>
                  <div style={{ fontSize: 10, color: "var(--text-light)", marginTop: 6, fontFamily: "'DM Mono', monospace" }}>
                    {replications === 1 ? "exploratory" : replications <= 3 ? "recommended for publication" : "high reliability"}
                  </div>
                </div>
                {error && (
                  <div style={{ background: "#FFF3F3", border: "1px solid #F5C6C6", borderRadius: 8, padding: 10, marginBottom: 12, fontSize: 12, color: "#C0392B" }}>
                    {error}
                  </div>
                )}

                <button className="run-btn" onClick={run} disabled={loading}>
                  {loading ? `Running… (${results.length > 0 ? "collecting" : "starting"})` : `▶  Run  ×${replications}`}
                </button>
              </div>

              {/* Prompt preview */}
              <div style={{ background: "white", borderRadius: 12, padding: 20, border: "1px solid var(--border)" }}>
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 14 }}>
                  <div style={{ fontSize: 10, color: "var(--text-light)", fontFamily: "'DM Mono', monospace", letterSpacing: "0.07em" }}>PROMPT PREVIEW</div>
                  <div style={{ display: "flex", gap: 6 }}>
                    <Tag bg={pc(modelMeta.provider).bg} color={pc(modelMeta.provider).color}>{pc(modelMeta.provider).label}</Tag>
                    <Tag bg="#F0F0F0" color="var(--text-mid)">{cond.model}</Tag>
                  </div>
                </div>
                <div style={{
                  background: "var(--cream)", borderRadius: 8, padding: 18,
                  fontFamily: "'DM Mono', monospace", fontSize: 12.5, lineHeight: 1.75,
                  color: "var(--text)", whiteSpace: "pre-wrap", border: "1px solid var(--cream-dark)", minHeight: 220
                }}>
                  {prompt}
                </div>
                <div style={{ marginTop: 10, fontSize: 11, color: "var(--text-light)", fontFamily: "'DM Mono', monospace" }}>
                  No system prompt · User-turn only · {prompt.length} chars
                </div>
              </div>
            </div>
          )}

          {/* ── Results tab ── */}
          {tab === "results" && (
            <div>
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 16 }}>
                <div style={{ fontSize: 11, color: "var(--text-light)", fontFamily: "'DM Mono', monospace" }}>
                  {results.length} run(s) · in-memory session
                </div>
                <div style={{ display: "flex", gap: 8 }}>
                  <button className="exp-btn" onClick={exportCSV} disabled={!results.length}>↓ CSV</button>
                  <button className="exp-btn" onClick={exportJSON} disabled={!results.length}>↓ JSON</button>
                </div>
              </div>

              {results.length === 0 ? (
                <div style={{ background: "white", borderRadius: 12, padding: 64, textAlign: "center", border: "1px solid var(--border)" }}>
                  <div style={{ fontFamily: "'Crimson Pro', serif", fontSize: 20, color: "var(--text-mid)", marginBottom: 8 }}>No data yet</div>
                  <div style={{ fontSize: 12, color: "var(--text-light)" }}>Run a condition to begin collecting responses</div>
                </div>
              ) : (
                <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
                  {results.map((r, i) => {
                    const dc = DOMAIN_COLORS;
                    const domainIdx = Object.keys(POLICY_DOMAINS).indexOf(r.domain) * 2;
                    const isOpen = expanded === r.id;
                    return (
                      <div key={r.id} style={{ background: "white", borderRadius: 12, border: "1px solid var(--border)", overflow: "hidden" }}>
                        <div
                          onClick={() => setExpanded(isOpen ? null : r.id)}
                          style={{ display: "flex", justifyContent: "space-between", alignItems: "center", padding: "13px 18px", cursor: "pointer", borderBottom: isOpen ? "1px solid var(--cream-dark)" : "none" }}
                        >
                          <div style={{ display: "flex", gap: 6, flexWrap: "wrap", alignItems: "center" }}>
                            <span style={{ fontFamily: "'DM Mono', monospace", fontSize: 10, color: "var(--text-light)" }}>#{r.id}</span>
                            <Tag bg={pc(r.provider).bg} color={pc(r.provider).color}>{pc(r.provider).label}</Tag>
                            <Tag bg="#EDE9FB" color="#5B21B6">{r.discipline}</Tag>
                            <Tag bg={dc[domainIdx] || "#F5F5F5"} color={dc[domainIdx+1] || "#555"}>{r.domain}</Tag>
                            <Tag bg="#F0F0F0" color="var(--text-mid)">rep {r.replication}</Tag>
                          </div>
                          <div style={{ display: "flex", gap: 10, alignItems: "center", flexShrink: 0, marginLeft: 12 }}>
                            <span style={{ fontSize: 10, color: "var(--text-light)", fontFamily: "'DM Mono', monospace" }}>
                              {new Date(r.timestamp).toLocaleTimeString()} · {(r.duration_ms / 1000).toFixed(1)}s
                            </span>
                            <span style={{ color: "var(--text-light)", fontSize: 12 }}>{isOpen ? "▲" : "▼"}</span>
                          </div>
                        </div>

                        {isOpen && (
                          <div style={{ padding: "16px 20px" }}>
                            <div style={{ fontSize: 10, color: "var(--text-light)", fontFamily: "'DM Mono', monospace", marginBottom: 8, letterSpacing: "0.06em" }}>RESPONSE · {r.model}</div>
                            <div style={{ fontSize: 13.5, lineHeight: 1.78, color: "var(--text)", whiteSpace: "pre-wrap" }}>
                              {r.response}
                            </div>
                            <div style={{ marginTop: 14, paddingTop: 12, borderTop: "1px solid var(--cream-dark)", display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                              <div style={{ fontSize: 10, fontFamily: "'DM Mono', monospace", color: "var(--text-light)" }}>
                                {r.model} · {r.institution} · {r.timestamp}
                              </div>
                              <button
                                onClick={() => setResults(p => p.filter(x => x.id !== r.id))}
                                style={{ background: "none", border: "1px solid var(--border)", borderRadius: 6, padding: "3px 10px", cursor: "pointer", fontSize: 11, color: "var(--text-light)" }}
                              >Remove</button>
                            </div>
                          </div>
                        )}
                      </div>
                    );
                  })}
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </>
  );
}
