import { useState, useEffect, useRef, useCallback } from "react";

const API_BASE = "http://localhost:8000";

const EXAMPLE_QUERIES = [
  "How do sparse autoencoder features explain GPT-2's failures on indirect object identification?",
  "What are the trade-offs between LoRA and full fine-tuning for large language models?",
  "How do retrieval-augmented generation systems compare to parametric memory approaches?",
  "What role does knowledge distillation play in making transformer models more efficient?",
];

const GRADIENT_MESH = `
  radial-gradient(ellipse at 20% 50%, rgba(99,58,189,0.15) 0%, transparent 60%),
  radial-gradient(ellipse at 80% 20%, rgba(14,165,233,0.12) 0%, transparent 55%),
  radial-gradient(ellipse at 60% 80%, rgba(236,72,153,0.08) 0%, transparent 50%),
  linear-gradient(135deg, #0a0a0f 0%, #0f0f1a 50%, #0a0a12 100%)
`;

function PaperCard({ paper, index }) {
  const [expanded, setExpanded] = useState(false);
  return (
    <div
      style={{
        background: "rgba(255,255,255,0.03)",
        border: "1px solid rgba(255,255,255,0.08)",
        borderRadius: 10,
        padding: "12px 14px",
        animation: `slideUp 0.4s ease ${index * 0.06}s both`,
        cursor: "pointer",
        transition: "background 0.2s",
      }}
      onMouseEnter={e => e.currentTarget.style.background = "rgba(255,255,255,0.06)"}
      onMouseLeave={e => e.currentTarget.style.background = "rgba(255,255,255,0.03)"}
      onClick={() => setExpanded(!expanded)}
    >
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", gap: 10 }}>
        <div style={{ flex: 1, minWidth: 0 }}>
          <a
            href={`https://arxiv.org/abs/${paper.arxiv_id}`}
            target="_blank"
            rel="noopener noreferrer"
            onClick={e => e.stopPropagation()}
            style={{ color: "#a78bfa", fontFamily: "'DM Sans', sans-serif", fontSize: 13, fontWeight: 500, lineHeight: 1.4, textDecoration: "none", display: "block", marginBottom: 3 }}
            onMouseEnter={e => e.target.style.color = "#c4b5fd"}
            onMouseLeave={e => e.target.style.color = "#a78bfa"}
          >
            {paper.title}
          </a>
          <div style={{ display: "flex", gap: 8, flexWrap: "wrap", alignItems: "center" }}>
            <span style={{ color: "rgba(255,255,255,0.3)", fontSize: 11, fontFamily: "'DM Sans', sans-serif" }}>
              {paper.authors?.slice(0, 2).join(", ")}{paper.authors?.length > 2 ? " et al." : ""}
            </span>
            {paper.year > 0 && <span style={{ color: "rgba(255,255,255,0.2)", fontSize: 11 }}>· {paper.year}</span>}
            {paper.citation_count > 0 && (
              <span style={{ color: "rgba(99,102,241,0.7)", fontSize: 11, background: "rgba(99,102,241,0.1)", padding: "1px 6px", borderRadius: 4 }}>
                ↗ {paper.citation_count.toLocaleString()} cites
              </span>
            )}
          </div>
        </div>
        <span style={{ color: "rgba(255,255,255,0.18)", fontSize: 10, fontFamily: "monospace", flexShrink: 0, marginTop: 2 }}>{paper.arxiv_id}</span>
      </div>
      {expanded && (paper.s2_tldr || paper.abstract) && (
        <div style={{ marginTop: 8, paddingTop: 8, borderTop: "1px solid rgba(255,255,255,0.06)", color: "rgba(255,255,255,0.45)", fontSize: 12, fontFamily: "'DM Sans', sans-serif", lineHeight: 1.6 }}>
          {paper.s2_tldr || paper.abstract}
        </div>
      )}
    </div>
  );
}

function AnswerText({ text }) {
  const parts = text.split(/(\[\d{4}\.\d{4,5}\])/g);
  return (
    <p style={{ color: "rgba(255,255,255,0.85)", fontFamily: "'DM Sans', sans-serif", fontSize: 15, lineHeight: 1.85, margin: 0, textAlign: "left" }}>
      {parts.map((part, i) => {
        const match = part.match(/\[(\d{4}\.\d{4,5})\]/);
        if (match) {
          return (
            <a key={i} href={`https://arxiv.org/abs/${match[1]}`} target="_blank" rel="noopener noreferrer"
              style={{ color: "#818cf8", fontSize: 11, verticalAlign: "super", fontFamily: "monospace", textDecoration: "none", background: "rgba(99,102,241,0.12)", padding: "1px 4px", borderRadius: 3, border: "1px solid rgba(99,102,241,0.25)" }}
              onMouseEnter={e => e.target.style.background = "rgba(99,102,241,0.25)"}
              onMouseLeave={e => e.target.style.background = "rgba(99,102,241,0.12)"}
            >
              {part}
            </a>
          );
        }
        return part;
      })}
    </p>
  );
}

function PipelineStep({ icon, label, active, done }) {
  return (
    <div style={{ display: "flex", alignItems: "center", gap: 6, opacity: done ? 0.35 : active ? 1 : 0.2, transition: "all 0.3s" }}>
      <div style={{
        width: 20, height: 20, borderRadius: "50%",
        background: active ? "rgba(99,102,241,0.3)" : done ? "rgba(16,185,129,0.2)" : "rgba(255,255,255,0.05)",
        border: `1px solid ${active ? "rgba(99,102,241,0.6)" : done ? "rgba(16,185,129,0.4)" : "rgba(255,255,255,0.1)"}`,
        display: "flex", alignItems: "center", justifyContent: "center", fontSize: 10,
        boxShadow: active ? "0 0 10px rgba(99,102,241,0.4)" : "none",
        animation: active ? "pulse 1.5s ease infinite" : "none",
      }}>
        {done ? "✓" : icon}
      </div>
      <span style={{ color: active ? "rgba(255,255,255,0.8)" : "rgba(255,255,255,0.3)", fontSize: 11, fontFamily: "'DM Sans', sans-serif" }}>{label}</span>
    </div>
  );
}

export default function PaperPilot() {
  const [query, setQuery] = useState("");
  const [isStreaming, setIsStreaming] = useState(false);
  const [answer, setAnswer] = useState("");
  const [papers, setPapers] = useState([]);
  const [status, setStatus] = useState("");
  const [currentStep, setCurrentStep] = useState("");
  const [doneData, setDoneData] = useState(null);
  const [error, setError] = useState("");
  const [corpusStats, setCorpusStats] = useState(null);
  const [hasResult, setHasResult] = useState(false);
  const [sentQuery, setSentQuery] = useState("");
  const chatRef = useRef(null);
  const textareaRef = useRef(null);

  useEffect(() => {
    fetch(`${API_BASE}/stats`).then(r => r.json()).then(setCorpusStats).catch(() => {});
  }, []);

  useEffect(() => {
    const style = document.createElement("style");
    style.textContent = `
      @import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;500;600;700;800&family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;1,9..40,300&display=swap');
      @keyframes slideUp { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: translateY(0); } }
      @keyframes fadeIn { from { opacity: 0; } to { opacity: 1; } }
      @keyframes pulse { 0%, 100% { box-shadow: 0 0 8px rgba(99,102,241,0.3); } 50% { box-shadow: 0 0 16px rgba(99,102,241,0.6); } }
      @keyframes gradientShift { 0%, 100% { background-position: 0% 50%; } 50% { background-position: 100% 50%; } }
      @keyframes blink { 0%, 100% { opacity: 1; } 50% { opacity: 0; } }
      ::placeholder { color: rgba(255,255,255,0.2) !important; }
      ::-webkit-scrollbar { width: 3px; } ::-webkit-scrollbar-track { background: transparent; } ::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.08); border-radius: 2px; }
      * { box-sizing: border-box; }
    `;
    document.head.appendChild(style);
    return () => document.head.removeChild(style);
  }, []);

  // Auto-scroll chat to bottom on new content
  useEffect(() => {
    if (chatRef.current) {
      chatRef.current.scrollTop = chatRef.current.scrollHeight;
    }
  }, [answer, papers, doneData, isStreaming]);

  const handleSubmit = useCallback(() => {
    if (!query.trim() || isStreaming) return;

    const submittedQuery = query.trim();
    setIsStreaming(true);
    setAnswer("");
    setPapers([]);
    setStatus("Starting...");
    setCurrentStep("start");
    setDoneData(null);
    setError("");
    setHasResult(true);
    setSentQuery(submittedQuery);
    setQuery("");

    fetch(`${API_BASE}/query/stream`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query: submittedQuery, max_iterations: 2 }),
    }).then(async (res) => {
      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n");
        buffer = lines.pop() || "";

        let eventType = "";
        let dataStr = "";

        for (const line of lines) {
          if (line.startsWith("event: ")) eventType = line.slice(7).trim();
          else if (line.startsWith("data: ")) dataStr = line.slice(6).trim();
          else if (line === "" && eventType && dataStr) {
            try {
              const data = JSON.parse(dataStr);
              if (eventType === "status") { setStatus(data.message); setCurrentStep(data.step); }
              else if (eventType === "token") { setAnswer(prev => prev + data.text); }
              else if (eventType === "citation") { setPapers(prev => [...prev, data]); }
              else if (eventType === "done") { setDoneData(data); setCurrentStep("done"); setIsStreaming(false); }
              else if (eventType === "error") { setError(data.message); setIsStreaming(false); }
            } catch {}
            eventType = ""; dataStr = "";
          }
        }
      }
      setIsStreaming(false);
    }).catch(err => {
      setError(err.message);
      setIsStreaming(false);
    });
  }, [query, isStreaming]);

  const handleKeyDown = (e) => {
    if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); handleSubmit(); }
  };

  const steps = [
    { id: "start", icon: "◈", label: "Init" },
    { id: "decompose", icon: "⊹", label: "Decompose" },
    { id: "retrieve", icon: "⊡", label: "Retrieve" },
    { id: "stream", icon: "⟡", label: "Synthesize" },
    { id: "citations", icon: "⊛", label: "Cite" },
    { id: "done", icon: "◉", label: "Done" },
  ];
  const stepOrder = steps.map(s => s.id);
  const currentIdx = stepOrder.indexOf(currentStep);

  return (
    <div style={{ height: "100vh", display: "flex", flexDirection: "column", background: GRADIENT_MESH, fontFamily: "'DM Sans', sans-serif", position: "relative", overflow: "hidden" }}>

      {/* Grain */}
      <div style={{ position: "fixed", inset: 0, backgroundImage: "url(\"data:image/svg+xml,%3Csvg viewBox='0 0 256 256' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='noise'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23noise)' opacity='0.03'/%3E%3C/svg%3E\")", pointerEvents: "none", zIndex: 0, opacity: 0.4 }} />
      <div style={{ position: "fixed", top: "-20%", left: "-10%", width: "60vw", height: "60vw", borderRadius: "50%", background: "radial-gradient(circle, rgba(99,58,189,0.06) 0%, transparent 70%)", pointerEvents: "none", zIndex: 0 }} />
      <div style={{ position: "fixed", bottom: "-20%", right: "-10%", width: "50vw", height: "50vw", borderRadius: "50%", background: "radial-gradient(circle, rgba(14,165,233,0.05) 0%, transparent 70%)", pointerEvents: "none", zIndex: 0 }} />

      {/* ── FIXED HEADER ── */}
      <div style={{ position: "relative", zIndex: 10, borderBottom: "1px solid rgba(255,255,255,0.05)", padding: "14px 24px", display: "flex", alignItems: "center", justifyContent: "space-between", background: "rgba(10,10,15,0.7)", backdropFilter: "blur(12px)" }}>
        <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
          <div style={{ width: 30, height: 30, borderRadius: 8, background: "linear-gradient(135deg, #6366f1, #a855f7)", display: "flex", alignItems: "center", justifyContent: "center", fontSize: 14, boxShadow: "0 0 16px rgba(99,102,241,0.4)" }}>✦</div>
          <span style={{ fontFamily: "'Syne', sans-serif", fontSize: 14, fontWeight: 700, color: "rgba(255,255,255,0.85)", letterSpacing: "0.06em" }}>PaperPilot</span>
        </div>
        <div style={{ display: "flex", alignItems: "center", gap: 16 }}>
          {corpusStats && (
            <span style={{ fontSize: 11, color: "rgba(255,255,255,0.2)", fontFamily: "monospace" }}>
              {corpusStats.papers?.toLocaleString()} papers · {corpusStats.chunks?.toLocaleString()} chunks
            </span>
          )}
        </div>
      </div>

      {/* ── SCROLLABLE CHAT AREA ── */}
      <div ref={chatRef} style={{ flex: 1, overflowY: "auto", position: "relative", zIndex: 1 }}>
        <div style={{ maxWidth: 760, margin: "0 auto", padding: "0 20px" }}>

          {/* Hero — full before first query, compact after */}
          {!hasResult ? (
            <div style={{ paddingTop: 80, paddingBottom: 40, animation: "slideUp 0.6s ease both" }}>
              <h1 style={{ fontFamily: "'Syne', sans-serif", fontSize: "clamp(28px, 4vw, 48px)", fontWeight: 800, color: "#fff", margin: "0 0 12px", lineHeight: 1.1, letterSpacing: "-0.02em", textAlign: "center" }}>
                Research{" "}
                <span style={{ background: "linear-gradient(90deg, #818cf8, #c084fc, #f472b6)", backgroundClip: "text", WebkitBackgroundClip: "text", color: "transparent", backgroundSize: "200% auto", animation: "gradientShift 4s ease infinite" }}>
                  intelligence
                </span>
                {" "}at depth
              </h1>
              <p style={{ color: "rgba(255,255,255,0.35)", fontSize: 15, margin: "0 0 40px", fontWeight: 300, lineHeight: 1.6, textAlign: "center" }}>
                Multi-hop reasoning across 800+ ML & NLP papers.
              </p>
              <div style={{ display: "flex", gap: 8, flexWrap: "wrap", justifyContent: "center" }}>
                {EXAMPLE_QUERIES.map((q, i) => (
                  <button key={i} onClick={() => setQuery(q)}
                    style={{ background: "rgba(255,255,255,0.04)", border: "1px solid rgba(255,255,255,0.08)", borderRadius: 20, color: "rgba(255,255,255,0.4)", fontSize: 12, fontFamily: "'DM Sans', sans-serif", padding: "7px 14px", cursor: "pointer", transition: "all 0.2s" }}
                    onMouseEnter={e => { e.target.style.background = "rgba(255,255,255,0.08)"; e.target.style.color = "rgba(255,255,255,0.7)"; }}
                    onMouseLeave={e => { e.target.style.background = "rgba(255,255,255,0.04)"; e.target.style.color = "rgba(255,255,255,0.4)"; }}
                  >
                    {q.length > 55 ? q.slice(0, 55) + "…" : q}
                  </button>
                ))}
              </div>
            </div>
          ) : (
            <div style={{ paddingTop: 20, paddingBottom: 4, textAlign: "center" }}>
              <span style={{ fontFamily: "'Syne', sans-serif", fontSize: 12, fontWeight: 600, color: "rgba(255,255,255,0.12)", letterSpacing: "0.06em" }}>
                Research intelligence at depth
              </span>
            </div>
          )}

          {/* ── USER QUERY BUBBLE (right) ── */}
          {sentQuery && (
            <div style={{ display: "flex", justifyContent: "flex-end", padding: "24px 0 8px", animation: "slideUp 0.3s ease both" }}>
              <div style={{ maxWidth: "70%", background: "linear-gradient(135deg, rgba(99,102,241,0.22), rgba(139,92,246,0.16))", border: "1px solid rgba(99,102,241,0.28)", borderRadius: "18px 18px 4px 18px", padding: "12px 18px" }}>
                <p style={{ color: "rgba(255,255,255,0.82)", fontSize: 14, fontFamily: "'DM Sans', sans-serif", margin: 0, lineHeight: 1.55 }}>{sentQuery}</p>
              </div>
            </div>
          )}

          {/* ── PIPELINE STATUS (left, shown while streaming) ── */}
          {isStreaming && (
            <div style={{ display: "flex", justifyContent: "flex-start", padding: "8px 0", animation: "fadeIn 0.3s ease" }}>
              <div style={{ background: "rgba(255,255,255,0.03)", border: "1px solid rgba(255,255,255,0.07)", borderRadius: "4px 18px 18px 18px", padding: "12px 16px", maxWidth: "85%" }}>
                <div style={{ display: "flex", gap: 14, flexWrap: "wrap", marginBottom: 8 }}>
                  {steps.map((step, i) => (
                    <PipelineStep key={step.id} icon={step.icon} label={step.label}
                      active={currentStep === step.id}
                      done={currentIdx > i && currentStep !== step.id}
                    />
                  ))}
                </div>
                <p style={{ color: "rgba(255,255,255,0.3)", fontSize: 11, margin: 0, fontFamily: "monospace" }}>{status}</p>
              </div>
            </div>
          )}

          {/* ── ANSWER BUBBLE (left) ── */}
          {(answer || (!isStreaming && hasResult && !error)) && (
            <div style={{ display: "flex", justifyContent: "flex-start", padding: "8px 0 4px", animation: "slideUp 0.4s ease both" }}>
              <div style={{ maxWidth: "88%" }}>
                {/* PaperPilot avatar label */}
                <div style={{ display: "flex", alignItems: "center", gap: 6, marginBottom: 8 }}>
                  <div style={{ width: 20, height: 20, borderRadius: 5, background: "linear-gradient(135deg, #6366f1, #a855f7)", display: "flex", alignItems: "center", justifyContent: "center", fontSize: 10 }}>✦</div>
                  <span style={{ fontFamily: "'Syne', sans-serif", fontSize: 11, fontWeight: 600, color: "rgba(255,255,255,0.3)", letterSpacing: "0.08em" }}>PAPERPILOT</span>
                  {doneData && (
                    <span style={{ fontSize: 10, color: "rgba(255,255,255,0.18)", marginLeft: 4 }}>
                      {doneData.citations_count} citations · {doneData.question_type}
                    </span>
                  )}
                </div>

                <div style={{ background: "rgba(255,255,255,0.03)", border: "1px solid rgba(255,255,255,0.07)", borderRadius: "4px 18px 18px 18px", padding: "20px 24px" }}>
                  <AnswerText text={answer} />
                  {isStreaming && (
                    <span style={{ display: "inline-block", width: 7, height: 15, background: "rgba(99,102,241,0.8)", borderRadius: 2, marginLeft: 2, animation: "blink 0.8s ease infinite", verticalAlign: "middle" }} />
                  )}
                </div>
              </div>
            </div>
          )}

          {/* Error */}
          {error && (
            <div style={{ display: "flex", justifyContent: "flex-start", padding: "8px 0" }}>
              <div style={{ background: "rgba(239,68,68,0.1)", border: "1px solid rgba(239,68,68,0.3)", borderRadius: "4px 18px 18px 18px", padding: "12px 18px", color: "#f87171", fontSize: 13 }}>
                ⚠ {error}
              </div>
            </div>
          )}

          {/* Sub-questions */}
          {doneData?.sub_questions?.length > 0 && (
            <div style={{ padding: "4px 0 8px 28px", animation: "slideUp 0.4s ease 0.1s both" }}>
              <details style={{ cursor: "pointer" }}>
                <summary style={{ fontFamily: "'Syne', sans-serif", fontSize: 10, fontWeight: 600, color: "rgba(255,255,255,0.2)", letterSpacing: "0.1em", textTransform: "uppercase", listStyle: "none", marginBottom: 8 }}>
                  ↳ {doneData.sub_questions.length} sub-questions explored
                </summary>
                <div style={{ display: "flex", flexDirection: "column", gap: 5, paddingTop: 8 }}>
                  {doneData.sub_questions.map((sq, i) => (
                    <div key={i} style={{ display: "flex", gap: 8, alignItems: "flex-start" }}>
                      <span style={{ color: "rgba(99,102,241,0.5)", fontSize: 11, fontFamily: "monospace", flexShrink: 0, marginTop: 2 }}>{i + 1}</span>
                      <span style={{ color: "rgba(255,255,255,0.28)", fontSize: 12, fontFamily: "'DM Sans', sans-serif", lineHeight: 1.5 }}>{sq}</span>
                    </div>
                  ))}
                </div>
              </details>
            </div>
          )}

          {/* Papers */}
          {papers.length > 0 && (
            <div style={{ padding: "4px 0 32px 0", animation: "slideUp 0.4s ease 0.15s both" }}>
              <p style={{ fontFamily: "'Syne', sans-serif", fontSize: 10, fontWeight: 600, color: "rgba(255,255,255,0.2)", letterSpacing: "0.1em", textTransform: "uppercase", marginBottom: 10, paddingLeft: 28 }}>
                {papers.length} cited papers
              </p>
              <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
                {papers.map((paper, i) => <PaperCard key={paper.arxiv_id} paper={paper} index={i} />)}
              </div>
            </div>
          )}

        </div>
      </div>

      {/* ── PINNED INPUT BAR ── */}
      <div style={{ position: "relative", zIndex: 10, borderTop: "1px solid rgba(255,255,255,0.05)", background: "rgba(10,10,15,0.85)", backdropFilter: "blur(16px)", padding: "12px 20px 16px" }}>
        <div style={{ maxWidth: 760, margin: "0 auto" }}>
          <div style={{
            display: "flex", alignItems: "flex-end", gap: 10,
            background: "rgba(255,255,255,0.04)", border: "1px solid rgba(255,255,255,0.1)",
            borderRadius: 16, padding: "10px 14px", transition: "border-color 0.2s",
          }}
            onFocus={e => e.currentTarget.style.borderColor = "rgba(99,102,241,0.4)"}
            onBlur={e => e.currentTarget.style.borderColor = "rgba(255,255,255,0.1)"}
          >
            <textarea
              ref={textareaRef}
              value={query}
              onChange={e => setQuery(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="Ask a research question... (Enter to send)"
              rows={1}
              style={{
                flex: 1, background: "transparent", border: "none", outline: "none", resize: "none",
                color: "rgba(255,255,255,0.87)", fontSize: 14, fontFamily: "'DM Sans', sans-serif",
                lineHeight: 1.6, fontWeight: 400, maxHeight: 120, overflowY: "auto",
              }}
              onInput={e => {
                e.target.style.height = "auto";
                e.target.style.height = Math.min(e.target.scrollHeight, 120) + "px";
              }}
            />
            <button
              onClick={handleSubmit}
              disabled={!query.trim() || isStreaming}
              style={{
                flexShrink: 0,
                background: query.trim() && !isStreaming ? "linear-gradient(135deg, #6366f1, #8b5cf6)" : "rgba(99,102,241,0.15)",
                border: "none", borderRadius: 10, color: "#fff", padding: "8px 16px",
                fontSize: 13, fontFamily: "'DM Sans', sans-serif", fontWeight: 500,
                cursor: query.trim() && !isStreaming ? "pointer" : "not-allowed",
                opacity: !query.trim() ? 0.4 : 1,
                transition: "all 0.2s",
                boxShadow: query.trim() && !isStreaming ? "0 0 14px rgba(99,102,241,0.35)" : "none",
                whiteSpace: "nowrap",
              }}
            >
              {isStreaming ? "●●●" : "Send →"}
            </button>
          </div>
          <p style={{ fontSize: 10, color: "rgba(255,255,255,0.15)", margin: "6px 0 0", textAlign: "center", fontFamily: "monospace" }}>
            Multi-hop RAG · BGE-large-en-v1.5 · Pinecone · DeepSeek · Built by Siddhi Dhupar
          </p>
        </div>
      </div>

    </div>
  );
}