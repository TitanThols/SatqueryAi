// ui/src/App.jsx — SatQuery AI main application
import { useState, useCallback } from 'react'
import './index.css'

const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000'
const ROLES = ['single', 't1', 't2', 'optical', 'sar']
const TASK_ICONS = {
  single_vqa: '🔍', caption: '📝', ground: '📍',
  change: '🔄', sar_optical: '📡'
}

// ── Components ─────────────────────────────────────────────────────────────────

function Header() {
  return (
    <header className="header">
      <div className="header-logo">
        <div className="icon">🛰</div>
        <div>
          <h1>SatQuery AI</h1>
          <div className="subtitle">ISRO/SAC · Problem Statement 26167</div>
        </div>
      </div>
      <span className="header-badge">Phase 1 · Baseline</span>
    </header>
  )
}

function UploadZone({ onFiles }) {
  const [dragging, setDragging] = useState(false)

  const handleDrop = useCallback((e) => {
    e.preventDefault(); setDragging(false)
    onFiles([...e.dataTransfer.files])
  }, [onFiles])

  return (
    <div
      className={`upload-zone ${dragging ? 'drag-over' : ''}`}
      onDragOver={e => { e.preventDefault(); setDragging(true) }}
      onDragLeave={() => setDragging(false)}
      onDrop={handleDrop}
    >
      <input type="file" accept=".tif,.tiff,.geotiff,.png,.jpg,.jpeg"
        multiple onChange={e => onFiles([...e.target.files])} />
      <div className="icon">🛰️</div>
      <p><strong>Drop GeoTIFFs here</strong> or click to browse</p>
      <p>Supports GeoTIFF/TIFF · PNG · JPEG · up to 500 MB each</p>
    </div>
  )
}

function ImageItem({ file, preview, role, onRoleChange, onRemove }) {
  return (
    <div className="image-item">
      {preview && <img src={preview} alt={file.name} />}
      <div className="info">
        <div className="name">{file.name}</div>
        <div className="meta">{(file.size / 1024 / 1024).toFixed(1)} MB</div>
      </div>
      <select className="role-select" value={role} onChange={e => onRoleChange(e.target.value)}>
        {ROLES.map(r => <option key={r} value={r}>{r}</option>)}
      </select>
      <button className="remove-btn" onClick={onRemove} title="Remove">✕</button>
    </div>
  )
}

function ConfidenceBar({ value }) {
  const pct = Math.round(value * 100)
  const color = pct > 70 ? '#10b981' : pct > 40 ? '#f59e0b' : '#ef4444'
  return (
    <div className="confidence-bar">
      <div className="bar-track">
        <div className="bar-fill" style={{ width: `${pct}%`, background: color }} />
      </div>
      <div className="bar-value" style={{ color }}>{pct}%</div>
    </div>
  )
}

function ExecutionTrace({ toolCalls }) {
  const [open, setOpen] = useState(false)
  if (!toolCalls?.length) return null
  return (
    <div className="card">
      <button className="trace-toggle" onClick={() => setOpen(o => !o)}>
        <span className="card-title" style={{ marginBottom: 0 }}>⚙️ Execution Trace · {toolCalls.length} step{toolCalls.length !== 1 ? 's' : ''}</span>
        <span style={{ fontSize: '0.8rem', color: 'var(--c-text-dim)' }}>{open ? '▲ collapse' : '▼ expand'}</span>
      </button>
      {open && (
        <div className="trace-content">
          {toolCalls.map((tc, i) => (
            <div key={i} className="trace-step">
              <div className="trace-step-header">
                <span className="trace-step-name">🔧 {tc.tool_name}</span>
                <span className="trace-step-time">{tc.duration_s?.toFixed(3)}s</span>
              </div>
              {tc.error && (
                <div style={{ color: 'var(--c-error)', fontSize: '0.78rem', marginBottom: '4px' }}>
                  ✗ {tc.error}
                </div>
              )}
              {tc.output && (
                <pre className="trace-pre">
                  {JSON.stringify(
                    Object.fromEntries(Object.entries(tc.output).filter(([k]) =>
                      !['change_mask_array', 'prob_map'].includes(k)
                    )),
                    null, 2
                  )}
                </pre>
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  )
}

function ValidationMessages({ errors, warnings }) {
  if (!errors?.length && !warnings?.length) return null
  return (
    <div className="card fade-in">
      <div className="card-title">⚠️ Validation</div>
      <div className="validation-list">
        {errors.map((e, i) => <div key={i} className="validation-item error">✗ {e}</div>)}
        {warnings.map((w, i) => <div key={i} className="validation-item warn">⚠ {w}</div>)}
      </div>
    </div>
  )
}

function ResultsPanel({ result, loading }) {
  if (loading) {
    return (
      <div className="empty-state">
        <div className="spinner" style={{ width: 48, height: 48, borderWidth: 4 }} />
        <p>Analyzing satellite imagery…</p>
      </div>
    )
  }
  if (!result) {
    return (
      <div className="empty-state">
        <div className="icon">🌍</div>
        <h3>Ready for Analysis</h3>
        <p>Upload satellite imagery, assign roles, and ask a question.</p>
      </div>
    )
  }

  const { task, answer, confidence, model_label, tool_calls,
          validation_errors, validation_warnings, image_urls,
          report_url, session_id, total_duration_s, error } = result

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 'var(--sp-lg)' }}>
      {/* Task badge + timing */}
      <div style={{ display: 'flex', gap: 'var(--sp-sm)', alignItems: 'center', flexWrap: 'wrap' }}>
        <span className="chip chip-blue">{TASK_ICONS[task] || '🔎'} {task?.replace('_', ' ')}</span>
        <span className="chip chip-green">⏱ {total_duration_s?.toFixed(2)}s</span>
        {model_label && <span className="chip chip-orange">🤖 {model_label}</span>}
      </div>

      {/* Validation */}
      <ValidationMessages errors={validation_errors} warnings={validation_warnings} />

      {/* Error state */}
      {error && (
        <div className="card fade-in">
          <div className="card-title">❌ Error</div>
          <p style={{ color: 'var(--c-error)' }}>{error}</p>
        </div>
      )}

      {/* Answer */}
      {answer && !error && (
        <div className="card fade-in">
          <div className="card-title">💬 Answer</div>
          <div className="answer-text">{answer}</div>
          {confidence > 0 && (
            <div style={{ marginTop: 'var(--sp-md)' }}>
              <div style={{ fontSize: '0.78rem', color: 'var(--c-text-dim)', marginBottom: '4px' }}>
                Confidence (token log-prob)
              </div>
              <ConfidenceBar value={confidence} />
            </div>
          )}
        </div>
      )}

      {/* Result images */}
      {image_urls?.length > 0 && (
        <div className="card fade-in">
          <div className="card-title">🗺️ Visual Output</div>
          {image_urls.map((url, i) => (
            <div key={i} className="result-image-wrapper" style={{ marginBottom: 'var(--sp-sm)' }}>
              <img src={`${API_BASE}${url}`} alt={`Result ${i + 1}`} />
              <span className="result-image-label">
                {url.includes('change') ? 'Change Mask' : url.includes('grounding') ? 'Grounding' : 'Fusion Mask'}
              </span>
            </div>
          ))}
        </div>
      )}

      {/* Execution trace */}
      <ExecutionTrace toolCalls={tool_calls} />

      {/* Download report */}
      {report_url && (
        <a href={`${API_BASE}${report_url}`} download="satquery_report.pdf" style={{ textDecoration: 'none' }}>
          <button className="btn btn-secondary" style={{ width: '100%' }}>
            📄 Download PDF Report
          </button>
        </a>
      )}
    </div>
  )
}

// ── Main App ──────────────────────────────────────────────────────────────────

export default function App() {
  const [imageItems, setImageItems] = useState([])  // {file, preview, role}
  const [query, setQuery]           = useState('')
  const [loading, setLoading]       = useState(false)
  const [result, setResult]         = useState(null)

  const handleFiles = useCallback((newFiles) => {
    const items = newFiles.map((f, i) => {
      const preview = f.type.startsWith('image/') ? URL.createObjectURL(f) : null
      // Auto-assign role based on position and filename
      let role = 'single'
      const stem = f.name.toLowerCase()
      if (stem.includes('sar') || stem.includes('s1')) role = 'sar'
      else if (stem.includes('optical') || stem.includes('s2')) role = 'optical'
      else if (stem.includes('t1') || imageItems.length === 0) role = 't1'
      else if (stem.includes('t2') || imageItems.length >= 1) role = 't2'
      return { file: f, preview, role }
    })
    setImageItems(prev => [...prev, ...items])
  }, [imageItems.length])

  const updateRole = (i, role) => {
    setImageItems(prev => prev.map((item, idx) => idx === i ? { ...item, role } : item))
  }
  const removeImage = (i) => {
    setImageItems(prev => prev.filter((_, idx) => idx !== i))
  }

  const handleSubmit = async () => {
    if (!imageItems.length) { alert('Please upload at least one image.'); return }
    if (!query.trim())      { alert('Please enter a question.'); return }

    setLoading(true); setResult(null)

    const form = new FormData()
    imageItems.forEach(item => form.append('files', item.file))
    form.append('roles', JSON.stringify(imageItems.map(i => i.role)))
    form.append('query', query)
    form.append('session', crypto.randomUUID())

    try {
      const resp = await fetch(`${API_BASE}/analyze`, { method: 'POST', body: form })
      if (!resp.ok) {
        const err = await resp.json().catch(() => ({}))
        setResult({ error: err.detail || `Server error ${resp.status}`, tool_calls: [] })
      } else {
        setResult(await resp.json())
      }
    } catch (e) {
      setResult({ error: `Network error: ${e.message}. Is the backend running?`, tool_calls: [] })
    } finally {
      setLoading(false)
    }
  }

  const suggestedQueries = [
    "What land use is visible in this scene?",
    "Describe this satellite image in detail.",
    "Where are the buildings in this image?",
    "What has changed between the two images?",
    "Show built-up areas using SAR and optical fusion.",
  ]

  return (
    <div className="app">
      <Header />
      <div className="main-grid">
        {/* ── Left panel ────────────────────────────────────────────────────── */}
        <div className="panel-left">
          {/* Upload */}
          <div className="card">
            <div className="card-title">📁 Upload Imagery</div>
            <UploadZone onFiles={handleFiles} />
            {imageItems.length > 0 && (
              <div className="image-list">
                {imageItems.map((item, i) => (
                  <ImageItem key={i} {...item}
                    onRoleChange={r => updateRole(i, r)}
                    onRemove={() => removeImage(i)}
                  />
                ))}
              </div>
            )}
          </div>

          {/* Query */}
          <div className="card query-box">
            <div className="card-title">💬 Query</div>
            <textarea
              id="query-input"
              placeholder="Ask a question about the satellite image(s)…&#10;e.g. 'What has changed between these two images?'"
              value={query}
              onChange={e => setQuery(e.target.value)}
              onKeyDown={e => { if (e.key === 'Enter' && e.metaKey) handleSubmit() }}
            />
            <div style={{ marginTop: 'var(--sp-sm)', display: 'flex', flexWrap: 'wrap', gap: 4 }}>
              {suggestedQueries.slice(0, 3).map((q, i) => (
                <button key={i} className="chip chip-blue"
                  style={{ cursor: 'pointer', fontSize: '0.7rem' }}
                  onClick={() => setQuery(q)}
                >{q}</button>
              ))}
            </div>
          </div>

          {/* Submit */}
          <button id="analyze-btn" className="btn btn-primary" onClick={handleSubmit} disabled={loading}>
            {loading ? <><div className="spinner" /> Analyzing…</> : '🚀 Analyze'}
          </button>

          {/* Help */}
          <div className="card" style={{ fontSize: '0.78rem', color: 'var(--c-text-dim)', lineHeight: 1.8 }}>
            <div className="card-title">📖 Image Roles</div>
            <p><b style={{ color: 'var(--c-text)' }}>single</b> — one image VQA or captioning</p>
            <p><b style={{ color: 'var(--c-text)' }}>t1 / t2</b> — bi-temporal change detection</p>
            <p><b style={{ color: 'var(--c-text)' }}>optical / sar</b> — fusion analysis</p>
          </div>
        </div>

        {/* ── Right panel ───────────────────────────────────────────────────── */}
        <div className="panel-right">
          <ResultsPanel result={result} loading={loading} />
        </div>
      </div>
    </div>
  )
}
