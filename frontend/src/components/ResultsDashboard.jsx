import ScoreGauge from './ScoreGauge.jsx'
import './ResultsDashboard.css'

function ResultsDashboard({ result, onReset }) {
  const {
    session_id,
    prediction,
    suspicion_score,
    confidence,
    risk_level,
    key_indicators,
    behavioral_stats,
  } = result

  const isGenuine = prediction === 'GENUINE_SOLVING'

  const riskColors = {
    LOW: 'var(--accent-green)',
    MEDIUM: 'var(--accent-yellow)',
    HIGH: 'var(--accent-orange)',
    CRITICAL: 'var(--accent-red)',
  }

  const riskColor = riskColors[risk_level] || 'var(--text-muted)'

  const getRecommendation = () => {
    if (suspicion_score < 0.3)
      return {
        text: 'Session appears legitimate. No further action required.',
        type: 'good',
      }
    if (suspicion_score < 0.6)
      return {
        text: 'Some anomalies detected. Consider manual review of the submission.',
        type: 'warning',
      }
    if (suspicion_score < 0.8)
      return {
        text: 'Significant anomalies found. Recommend follow-up proctoring or viva for this candidate.',
        type: 'danger',
      }
    return {
      text: 'Strong indicators of automated or pre-typed input. Flag for immediate review and possible disqualification.',
      type: 'critical',
    }
  }

  const recommendation = getRecommendation()

  const statCards = [
    { label: 'Arrow Keys', value: behavioral_stats?.arrow_keys ?? '—', icon: '↕' },
    { label: 'Backspaces', value: behavioral_stats?.backspaces ?? '—', icon: '⌫' },
    { label: 'Deletion Ratio', value: behavioral_stats?.deletion_ratio != null ? (behavioral_stats.deletion_ratio * 100).toFixed(1) + '%' : '—', icon: '✂' },
    { label: 'Editor Clicks', value: behavioral_stats?.editor_clicks ?? '—', icon: '🖱' },
    { label: 'Run Count', value: behavioral_stats?.run_count ?? '—', icon: '▶' },
    { label: 'Submit Count', value: behavioral_stats?.submit_count ?? '—', icon: '📤' },
    { label: 'Edit Switches', value: behavioral_stats?.edit_switches ?? '—', icon: '⇄' },
    { label: 'Modifier Usage', value: behavioral_stats?.modifier_usage ?? '—', icon: '⌘' },
  ]

  const indicatorIcon = (type) => {
    if (type === 'critical') return '🔴'
    if (type === 'warning') return '🟡'
    return '🟢'
  }

  return (
    <div className="dashboard">
      {/* Top bar */}
      <div className="dash-topbar">
        <div className="session-label">
          <span className="session-tag">Session</span>
          <span className="session-id">{session_id || 'Unknown'}</span>
        </div>
        <button className="reset-btn" onClick={onReset}>
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <polyline points="1 4 1 10 7 10" />
            <path d="M3.51 15a9 9 0 1 0 2.13-9.36L1 10" />
          </svg>
          New Analysis
        </button>
      </div>

      {/* Verdict Banner */}
      <div className={`verdict-banner ${isGenuine ? 'genuine' : 'suspicious'}`}>
        <div className="verdict-left">
          <span className="verdict-icon">{isGenuine ? '✓' : '⚠'}</span>
          <div>
            <h2 className="verdict-text">
              {isGenuine ? 'GENUINE SESSION' : 'LINEAR TYPING DETECTED'}
            </h2>
            <p className="verdict-sub">
              Confidence: {(confidence * 100).toFixed(1)}%
            </p>
          </div>
        </div>
        <div className="verdict-right">
          <span className="risk-badge" style={{ '--risk-color': riskColor }}>
            {risk_level}
          </span>
        </div>
      </div>

      {/* Score + Recommendation row */}
      <div className="dash-row score-row">
        <div className="card score-card">
          <h3 className="card-title">Suspicion Score</h3>
          <ScoreGauge score={suspicion_score} />
        </div>
        <div className={`card recommendation-card rec-${recommendation.type}`}>
          <h3 className="card-title">Recommendation</h3>
          <p className="rec-text">{recommendation.text}</p>
          <div className="rec-score-bar">
            <div className="rec-bar-track">
              <div
                className="rec-bar-fill"
                style={{
                  width: `${suspicion_score * 100}%`,
                  background: riskColor,
                }}
              />
            </div>
            <span className="rec-score-label">
              Score: {(suspicion_score * 100).toFixed(1)}%
            </span>
          </div>
        </div>
      </div>

      {/* Behavioral Stats Grid */}
      <div className="card">
        <h3 className="card-title">Behavioral Statistics</h3>
        <div className="stats-grid">
          {statCards.map((s) => (
            <div className="stat-item" key={s.label}>
              <span className="stat-icon">{s.icon}</span>
              <span className="stat-value">{s.value}</span>
              <span className="stat-label">{s.label}</span>
            </div>
          ))}
        </div>
      </div>

      {/* Key Indicators */}
      {key_indicators && key_indicators.length > 0 && (
        <div className="card">
          <h3 className="card-title">Key Behavioral Indicators</h3>
          <div className="indicators-list">
            {key_indicators.map((ind, i) => (
              <div className={`indicator indicator-${ind.type}`} key={i}>
                <span className="ind-icon">{indicatorIcon(ind.type)}</span>
                <div className="ind-content">
                  <div className="ind-header">
                    <strong>{ind.feature}</strong>
                    <span className="ind-value">
                      {typeof ind.value === 'number'
                        ? ind.value.toLocaleString(undefined, {
                            maximumFractionDigits: 4,
                          })
                        : ind.value}
                    </span>
                  </div>
                  <p className="ind-message">{ind.message}</p>
                  {ind.detail && <p className="ind-detail">{ind.detail}</p>}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}

export default ResultsDashboard
