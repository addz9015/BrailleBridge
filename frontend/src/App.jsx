import React, { useEffect, useMemo, useState } from "react";

const API_URL = (
  import.meta.env.VITE_API_URL || "http://127.0.0.1:8000"
).replace(/\/$/, "");

const LOW_CONFIDENCE_THRESHOLD = 0.75;
// Render order for a 2-column grid so visual positions match dots 1..6:
// [top-left, top-right, mid-left, mid-right, bot-left, bot-right]
// maps to backend indices [1,4,2,5,3,6] => zero-based [0,3,1,4,2,5].
const TAP_DOT_UI_ORDER = [0, 3, 1, 4, 2, 5];

const BRAILLE_PATTERNS = {
  a: [1, 0, 0, 0, 0, 0],
  b: [1, 1, 0, 0, 0, 0],
  c: [1, 0, 0, 1, 0, 0],
  d: [1, 0, 0, 1, 1, 0],
  e: [1, 0, 0, 0, 1, 0],
  f: [1, 1, 0, 1, 0, 0],
  g: [1, 1, 0, 1, 1, 0],
  h: [1, 1, 0, 0, 1, 0],
  i: [0, 1, 0, 1, 0, 0],
  j: [0, 1, 0, 1, 1, 0],
  k: [1, 0, 1, 0, 0, 0],
  l: [1, 1, 1, 0, 0, 0],
  m: [1, 0, 1, 1, 0, 0],
  n: [1, 0, 1, 1, 1, 0],
  o: [1, 0, 1, 0, 1, 0],
  p: [1, 1, 1, 1, 0, 0],
  q: [1, 1, 1, 1, 1, 0],
  r: [1, 1, 1, 0, 1, 0],
  s: [0, 1, 1, 1, 0, 0],
  t: [0, 1, 1, 1, 1, 0],
  u: [1, 0, 1, 0, 0, 1],
  v: [1, 1, 1, 0, 0, 1],
  w: [0, 1, 0, 1, 1, 1],
  x: [1, 0, 1, 1, 0, 1],
  y: [1, 0, 1, 1, 1, 1],
  z: [1, 0, 1, 0, 1, 1],
  " ": [0, 0, 0, 0, 0, 0],
};

async function requestJson(path, options = {}) {
  const response = await fetch(`${API_URL}${path}`, {
    headers: {
      "Content-Type": "application/json",
      ...(options.headers || {}),
    },
    ...options,
  });

  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(
      payload.detail || `Request failed with status ${response.status}`,
    );
  }

  return payload;
}

function parseSampleId(choice) {
  if (!choice) return 0;
  return Number.parseInt(choice.split(" ")[0], 10);
}

function formatScore(score) {
  return Number(score).toFixed(3);
}

function normalizeCandidates(candidates) {
  if (!candidates?.length) return [];
  const scores = candidates.map((candidate) => candidate.score);
  const max = Math.max(...scores);
  const min = Math.min(...scores);
  const span = Math.max(max - min, 1e-6);
  return candidates.map((candidate) => ({
    ...candidate,
    width: ((candidate.score - min) / span) * 100,
  }));
}

function linePath(values, width, height) {
  if (!values?.length) return "";
  const min = Math.min(...values);
  const max = Math.max(...values);
  const span = Math.max(max - min, 1e-6);
  return values
    .map((value, index) => {
      const x = (index / Math.max(values.length - 1, 1)) * width;
      const y = height - ((value - min) / span) * height;
      return `${index === 0 ? "M" : "L"} ${x.toFixed(2)} ${y.toFixed(2)}`;
    })
    .join(" ");
}

function SignalChart({ label, noisy, clean, accent = "#6ce5b1" }) {
  const width = 900;
  const height = 220;
  const noisyValues = noisy?.map((row) => row?.[0] ?? 0) || [];
  const cleanValues = clean?.map((row) => row?.[0] ?? 0) || [];
  const noisyPath = useMemo(
    () => linePath(noisyValues, width, height),
    [noisy],
  );
  const cleanPath = useMemo(
    () => linePath(cleanValues, width, height),
    [clean],
  );

  return (
    <section className="panel chart-panel">
      <div className="panel-head">
        <div>
          <p className="eyebrow">Signal view</p>
          <h3>{label}</h3>
        </div>
        <span className="chip">Channel 0</span>
      </div>
      <div className="chart-wrap">
        <svg viewBox={`0 0 ${width} ${height}`} role="img" aria-label={label}>
          <defs>
            <linearGradient id="noisyGradient" x1="0" y1="0" x2="1" y2="0">
              <stop offset="0%" stopColor="#ff8c6b" />
              <stop offset="100%" stopColor="#ff5f7f" />
            </linearGradient>
            <linearGradient id="cleanGradient" x1="0" y1="0" x2="1" y2="0">
              <stop offset="0%" stopColor={accent} />
              <stop offset="100%" stopColor="#82a8ff" />
            </linearGradient>
          </defs>
          <rect
            x="0"
            y="0"
            width={width}
            height={height}
            rx="18"
            fill="rgba(255,255,255,0.02)"
          />
          <path
            d={noisyPath}
            fill="none"
            stroke="url(#noisyGradient)"
            strokeWidth="3.5"
            strokeLinecap="round"
          />
          <path
            d={cleanPath}
            fill="none"
            stroke="url(#cleanGradient)"
            strokeWidth="3.5"
            strokeLinecap="round"
            strokeDasharray="6 5"
          />
        </svg>
      </div>
      <div className="legend">
        <span>
          <i className="dot noisy" />
          Noisy
        </span>
        <span>
          <i className="dot clean" />
          Denoised
        </span>
      </div>
    </section>
  );
}

function MetricCard({ label, value, tone }) {
  return (
    <div className={`metric-card tone-${tone}`}>
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

function sanitizeTapWord(word) {
  return (word || "").toLowerCase().replace(/[^a-z ]/g, "").trim();
}

function buildEmptyTapCells(length) {
  return Array.from({ length: Math.max(1, length) }, () => [0, 0, 0, 0, 0, 0]);
}

function patternToUnicode(pattern = [0, 0, 0, 0, 0, 0]) {
  let code = 0x2800;
  pattern.slice(0, 6).forEach((active, index) => {
    if (active) code |= 1 << index;
  });
  return String.fromCharCode(code);
}

function wordToBrailleUnicode(word) {
  const cleaned = sanitizeTapWord(word);
  if (!cleaned) return "";
  return cleaned
    .split("")
    .map((ch) => patternToUnicode(BRAILLE_PATTERNS[ch] || BRAILLE_PATTERNS[" "]))
    .join("");
}

function wordToPatterns(word) {
  const cleaned = sanitizeTapWord(word);
  if (!cleaned) return [];
  return cleaned
    .split("")
    .map((ch) => [...(BRAILLE_PATTERNS[ch] || BRAILLE_PATTERNS[" "])]);
}

function cellsToSyntheticSignal(cells) {
  return (cells || []).map((pattern) => {
    const safe = Array.isArray(pattern) ? pattern : [0, 0, 0, 0, 0, 0];
    const taxels = [
      safe[0], safe[1], safe[2], safe[3], safe[4], safe[5],
      safe[0], safe[1], safe[2], safe[3], safe[4], safe[5],
    ];
    return taxels.map((v) => Number(v));
  });
}

function spellOutWord(word) {
  return (word || "").split("").join(" ");
}

function buildSpokenWord(word, spellMode) {
  return spellMode ? spellOutWord(word) : word;
}

function speakText(text) {
  if (typeof window === "undefined" || !("speechSynthesis" in window)) {
    return false;
  }
  const utterance = new window.SpeechSynthesisUtterance(text);
  utterance.rate = 0.95;
  utterance.pitch = 1.0;
  utterance.volume = 1.0;
  window.speechSynthesis.cancel();
  window.speechSynthesis.speak(utterance);
  return true;
}

function downloadJson(fileName, payload) {
  const blob = new Blob([JSON.stringify(payload, null, 2)], {
    type: "application/json",
  });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = fileName;
  anchor.click();
  URL.revokeObjectURL(url);
}

export default function App() {
  const [samples, setSamples] = useState([]);
  const [sampleChoice, setSampleChoice] = useState("");
  const [useLm, setUseLm] = useState(true);
  const [beamWidth, setBeamWidth] = useState(8);
  const [lmWeight, setLmWeight] = useState(0.3);
  const [loading, setLoading] = useState(false);
  const [hydrating, setHydrating] = useState(true);
  const [error, setError] = useState("");
  const [result, setResult] = useState(null);

  const [tapWord, setTapWord] = useState("cat");
  const [tapCells, setTapCells] = useState(buildEmptyTapCells(3));
  const [tapLoading, setTapLoading] = useState(false);
  const [tapError, setTapError] = useState("");
  const [tapResult, setTapResult] = useState(null);
  const [spellMode, setSpellMode] = useState(false);
  const [spokenStatus, setSpokenStatus] = useState("");
  const [spokenAlternatives, setSpokenAlternatives] = useState([]);

  const speechSupported =
    typeof window !== "undefined" && "speechSynthesis" in window;

  useEffect(() => {
    let active = true;

    async function load() {
      try {
        const payload = await requestJson("/samples");
        const choices = payload.choices || [];
        if (!active) return;
        setSamples(choices);
        setSampleChoice((current) => current || choices[0] || "");
        setError("");
      } catch (exception) {
        if (!active) return;
        setError(exception.message);
      } finally {
        if (active) setHydrating(false);
      }
    }

    load();
    return () => {
      active = false;
    };
  }, []);

  useEffect(() => {
    const cleaned = sanitizeTapWord(tapWord);
    const targetLength = Math.max(1, cleaned.length || tapWord.length || 1);
    setTapCells((current) => {
      if (current.length === targetLength) return current;
      return Array.from({ length: targetLength }, (_, idx) =>
        current[idx] ? [...current[idx]] : [0, 0, 0, 0, 0, 0],
      );
    });
  }, [tapWord]);

  const selectedSampleId = parseSampleId(sampleChoice);
  const normalizedCandidates = useMemo(
    () => normalizeCandidates(result?.candidates || []),
    [result],
  );

  async function runAnalysis() {
    setLoading(true);
    setError("");
    try {
      const payload = await requestJson("/analyze", {
        method: "POST",
        body: JSON.stringify({
          sample_id: selectedSampleId,
          use_lm: useLm,
          beam_width: Number(beamWidth),
          lm_weight: Number(lmWeight),
        }),
      });
      setResult(payload);
    } catch (exception) {
      setError(exception.message);
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    if (!sampleChoice) return;
    if (!result && !loading && !hydrating) {
      runAnalysis();
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [sampleChoice, hydrating]);

  const groundedLabel = result?.word || "—";
  const greedyLabel = result?.greedy_pred || "—";
  const lmLabel = result?.lm_pred || "—";

  function toggleTapDot(cellIndex, dotIndex) {
    setTapCells((current) =>
      current.map((cell, idx) => {
        if (idx !== cellIndex) return cell;
        return cell.map((value, dIdx) => (dIdx === dotIndex ? (value ? 0 : 1) : value));
      }),
    );
    setTapResult(null);
    setTapError("");
  }

  function clearTapCells() {
    setTapCells((current) => current.map(() => [0, 0, 0, 0, 0, 0]));
    setTapResult(null);
    setTapError("");
    setSpokenStatus("");
    setSpokenAlternatives([]);
  }

  function announceWord(word, options = {}) {
    if (!word) return;
    const spokenWord = buildSpokenWord(word, options.spellMode ?? spellMode);
    const prefix = options.prefix || "Decoded word";
    const spokenText = `${prefix}: ${spokenWord}.`;
    if (!speechSupported) {
      setSpokenStatus("Speech is not supported in this browser.");
      return;
    }
    speakText(spokenText);
    setSpokenStatus(spokenText);
  }

  function announceAlternatives(alternatives) {
    if (!alternatives?.length) return;
    if (!speechSupported) {
      setSpokenStatus("Speech is not supported in this browser.");
      return;
    }
    const message = alternatives
      .map((word, index) => `Option ${index + 1}: ${buildSpokenWord(word, spellMode)}`)
      .join(". ");
    const speech = `Low confidence. ${message}.`;
    speakText(speech);
    setSpokenStatus(speech);
  }

  function exportTapBundle() {
    if (!tapResult?.predicted_word) {
      setTapError("Decode taps first before exporting.");
      return;
    }

    const predicted = tapResult.predicted_word;
    const cellPatterns =
      tapResult?.cells?.map((cell) => cell.pattern) || wordToPatterns(predicted);
    const brailleUnicode =
      tapResult?.cells?.map((cell) => cell.unicode).join("") ||
      wordToBrailleUnicode(predicted);

    const bundle = {
      created_at: new Date().toISOString(),
      input_word: sanitizeTapWord(tapWord),
      predicted_word: predicted,
      direct_word: tapResult?.direct_word || "",
      decode_mode: tapResult?.decode_mode || "direct",
      confidence: Number(tapResult?.confidence ?? 0),
      braille_unicode: brailleUnicode,
      braille_cells: cellPatterns,
      regenerated_signal: cellsToSyntheticSignal(cellPatterns),
    };

    downloadJson(
      `tap_accessibility_${predicted.replace(/\s+/g, "_") || "output"}.json`,
      bundle,
    );
    setSpokenStatus("Exported accessibility bundle as JSON.");
  }

  async function runTapDecode() {
    const cleaned = sanitizeTapWord(tapWord);
    if (!cleaned) {
      setTapError("Enter a word first (a-z and spaces).");
      return;
    }

    setTapLoading(true);
    setTapError("");
    try {
      const payload = await requestJson("/tap/analyze", {
        method: "POST",
        body: JSON.stringify({
          word: cleaned,
          cells: tapCells.map((dots) => ({ dots })),
        }),
      });
      setTapResult(payload);

      const predicted = payload?.predicted_word || "";
      const alternatives = [
        predicted,
        payload?.direct_word,
      ].filter((word, idx, all) => word && !word.includes("?") && all.indexOf(word) === idx);

      const confidence = Number(payload?.confidence ?? 0);
      const lowConfidence = confidence < LOW_CONFIDENCE_THRESHOLD;
      setSpokenAlternatives(lowConfidence ? alternatives : []);

      if (predicted) {
        announceWord(predicted, {
          prefix: lowConfidence
            ? "Decoded word with low confidence"
            : "Decoded word",
        });
      }

      if (lowConfidence && alternatives.length > 1) {
        announceAlternatives(alternatives);
      }
    } catch (exception) {
      setTapError(exception.message);
    } finally {
      setTapLoading(false);
    }
  }

  return (
    <div className="app-shell">
      <div className="ambient ambient-one" />
      <div className="ambient ambient-two" />

      <main className="layout">
        <section className="hero panel">
          <div>
            <p className="eyebrow">Braille recognition platform</p>
            <h1>
              FastAPI backend, React frontend, model controls that actually
              matter.
            </h1>
            <p className="hero-copy">
              Inspect a deterministic sample, compare noisy and denoised
              signals, and tune beam search plus language-model weight without
              leaving the interface.
            </p>
          </div>
          <div className="hero-badges">
            <div className="status-pill online">Backend online</div>
            <div className="status-pill">React UI</div>
            <div className="status-pill">CTC + LM</div>
          </div>
        </section>

        <section className="workspace">
          <aside className="panel controls">
            <div className="panel-head">
              <div>
                <p className="eyebrow">Control deck</p>
                <h3>Inference settings</h3>
              </div>
              <span className="chip">HTTP</span>
            </div>

            <label className="field">
              <span>Sample</span>
              <select
                value={sampleChoice}
                onChange={(event) => setSampleChoice(event.target.value)}
              >
                {samples.length === 0 ? (
                  <option>Loading samples…</option>
                ) : null}
                {samples.map((choice) => (
                  <option key={choice} value={choice}>
                    {choice}
                  </option>
                ))}
              </select>
            </label>

            <label className="toggle-row">
              <span>Use bigram LM</span>
              <button
                type="button"
                className={`toggle ${useLm ? "on" : ""}`}
                onClick={() => setUseLm((current) => !current)}
              >
                <span />
              </button>
            </label>

            <label className="field">
              <span>Beam width</span>
              <input
                type="range"
                min="4"
                max="24"
                step="1"
                value={beamWidth}
                onChange={(event) => setBeamWidth(Number(event.target.value))}
              />
              <strong>{beamWidth}</strong>
            </label>

            <label className="field">
              <span>LM weight</span>
              <input
                type="range"
                min="0"
                max="1"
                step="0.05"
                value={lmWeight}
                onChange={(event) => setLmWeight(Number(event.target.value))}
              />
              <strong>{lmWeight.toFixed(2)}</strong>
            </label>

            <button
              type="button"
              className="primary-btn"
              onClick={runAnalysis}
              disabled={loading || !samples.length}
            >
              {loading ? "Running inference…" : "Run analysis"}
            </button>

            <div className="status-box">
              <span>API</span>
              <strong>{API_URL}</strong>
            </div>
            {error ? <div className="error-box">{error}</div> : null}
          </aside>

          <section className="results">
            <div className="metrics-grid">
              <MetricCard
                label="Ground truth"
                value={groundedLabel}
                tone="gold"
              />
              <MetricCard label="Greedy CTC" value={greedyLabel} tone="rose" />
              <MetricCard label="Beam + LM" value={lmLabel} tone="blue" />
            </div>

            <div className="panel results-panel">
              <div className="panel-head">
                <div>
                  <p className="eyebrow">Output detail</p>
                  <h3>Decoder summary</h3>
                </div>
                <span className="chip">
                  {result?.candidates?.length || 0} candidates
                </span>
              </div>

              <div className="summary-grid">
                <div>
                  <span>Sample id</span>
                  <strong>{result?.sample_id ?? "—"}</strong>
                </div>
                <div>
                  <span>Greedy correct</span>
                  <strong>
                    {result ? (result.greedy_ok ? "Yes" : "No") : "—"}
                  </strong>
                </div>
                <div>
                  <span>LM correct</span>
                  <strong>
                    {result ? (result.lm_ok ? "Yes" : "No") : "—"}
                  </strong>
                </div>
                <div>
                  <span>Characters</span>
                  <strong>{result?.word?.length ?? "—"}</strong>
                </div>
              </div>
            </div>

            <div className="chart-grid">
              <SignalChart
                label="Noisy vs denoised signal"
                noisy={result?.noisy}
                clean={result?.denoised}
                accent="#6ce5b1"
              />

              <section className="panel candidate-panel">
                <div className="panel-head">
                  <div>
                    <p className="eyebrow">Ranking</p>
                    <h3>Top candidate sequences</h3>
                  </div>
                  <span className="chip">LM {useLm ? "on" : "off"}</span>
                </div>

                <div className="candidate-list">
                  {normalizedCandidates.length ? (
                    normalizedCandidates.map((candidate, index) => (
                      <div
                        key={`${candidate.text}-${index}`}
                        className="candidate-row"
                      >
                        <div className="candidate-labels">
                          <strong>{candidate.text}</strong>
                          <span>{formatScore(candidate.score)}</span>
                        </div>
                        <div className="candidate-bar-track">
                          <div
                            className="candidate-bar-fill"
                            style={{ width: `${candidate.width}%` }}
                          />
                        </div>
                      </div>
                    ))
                  ) : (
                    <p className="empty-state">
                      Run analysis to see the ranked candidates.
                    </p>
                  )}
                </div>
              </section>
            </div>

            <section className="panel tap-panel">
              <div className="panel-head">
                <div>
                  <p className="eyebrow">Dot tap mode</p>
                  <h3>Tap Braille dots directly</h3>
                </div>
                <span className="chip">High accuracy</span>
              </div>

              <div className="tap-controls">
                <label className="field">
                  <span>Word length target</span>
                  <input
                    className="text-input"
                    type="text"
                    value={tapWord}
                    onChange={(event) => setTapWord(event.target.value)}
                    placeholder="Type target word"
                    maxLength={32}
                  />
                </label>

                <div className="tap-actions">
                  <button type="button" className="primary-btn" onClick={runTapDecode} disabled={tapLoading}>
                    {tapLoading ? "Decoding taps..." : "Decode tapped dots"}
                  </button>
                  <button type="button" className="ghost-btn" onClick={clearTapCells}>
                    Clear dots
                  </button>
                </div>

                <div className="tap-accessibility-row">
                  <label className="toggle-row compact">
                    <span>Spell mode</span>
                    <button
                      type="button"
                      className={`toggle ${spellMode ? "on" : ""}`}
                      onClick={() => setSpellMode((current) => !current)}
                    >
                      <span />
                    </button>
                  </label>

                  <div className="tap-accessibility-actions">
                    <button
                      type="button"
                      className="secondary-btn"
                      disabled={!tapResult?.predicted_word}
                      onClick={() => announceWord(tapResult?.predicted_word || "")}
                    >
                      Speak prediction
                    </button>
                    <button
                      type="button"
                      className="secondary-btn"
                      disabled={!tapResult?.predicted_word}
                      onClick={exportTapBundle}
                    >
                      Export word + Braille + signal
                    </button>
                  </div>
                </div>
              </div>

              <div className="tap-cell-grid">
                {tapCells.map((cell, cellIndex) => (
                  <div key={`tap-cell-${cellIndex}`} className="tap-cell-card">
                    <p>Cell {cellIndex + 1}</p>
                    <div className="tap-dots">
                      {TAP_DOT_UI_ORDER.map((dotIndex) => (
                        <button
                          key={`tap-dot-${cellIndex}-${dotIndex}`}
                          type="button"
                          className={`tap-dot ${cell[dotIndex] ? "active" : ""}`}
                          onClick={() => toggleTapDot(cellIndex, dotIndex)}
                          aria-label={`Cell ${cellIndex + 1} dot ${dotIndex + 1}`}
                        />
                      ))}
                    </div>
                  </div>
                ))}
              </div>

              <div className="tap-summary-grid">
                <div>
                  <span>Target</span>
                  <strong>{sanitizeTapWord(tapWord) || "-"}</strong>
                </div>
                <div>
                  <span>Predicted</span>
                  <strong>{tapResult?.predicted_word || "-"}</strong>
                </div>
                <div>
                  <span>Direct decode</span>
                  <strong>{tapResult?.direct_word || "-"}</strong>
                </div>
                <div>
                  <span>Mode</span>
                  <strong>{tapResult?.decode_mode || "-"}</strong>
                </div>
                <div>
                  <span>Confidence</span>
                  <strong>
                    {tapResult?.confidence
                      ? `${Math.round(Number(tapResult.confidence) * 100)}%`
                      : "-"}
                  </strong>
                </div>
                <div>
                  <span>Unknown cells</span>
                  <strong>{tapResult?.unknown_cells ?? "-"}</strong>
                </div>
                <div>
                  <span>Match</span>
                  <strong>{tapResult ? (tapResult.correct ? "Yes" : "No") : "-"}</strong>
                </div>
              </div>

              {tapResult?.warnings?.length ? (
                <div className="warn-box">
                  <span>Tap decoding warning</span>
                  <strong>{tapResult.warnings[0]}</strong>
                </div>
              ) : null}

              {spokenAlternatives.length > 1 ? (
                <div className="status-box">
                  <span>Low-confidence spoken options</span>
                  <div className="tap-option-actions">
                    {spokenAlternatives.map((word) => (
                      <button
                        key={`spoken-option-${word}`}
                        type="button"
                        className="ghost-btn compact"
                        onClick={() => announceWord(word, { prefix: "Alternative" })}
                      >
                        Speak: {word}
                      </button>
                    ))}
                  </div>
                </div>
              ) : null}

              {spokenStatus ? (
                <div className="status-box" aria-live="polite">
                  <span>Audio status</span>
                  <strong>{spokenStatus}</strong>
                </div>
              ) : null}

              {!speechSupported ? (
                <div className="warn-box">
                  <span>Speech support</span>
                  <strong>
                    This browser does not support speech synthesis. Export JSON still works.
                  </strong>
                </div>
              ) : null}

              {tapError ? <div className="error-box">{tapError}</div> : null}
            </section>
          </section>
        </section>
      </main>
    </div>
  );
}
