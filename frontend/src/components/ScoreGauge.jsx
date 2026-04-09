import './ScoreGauge.css'

function ScoreGauge({ score }) {
  const percentage = Math.round(score * 100)
  const circumference = 2 * Math.PI * 54
  const offset = circumference - (score * 0.75) * circumference // 270deg arc

  const getColor = () => {
    if (score < 0.3) return 'var(--accent-green)'
    if (score < 0.6) return 'var(--accent-yellow)'
    if (score < 0.8) return 'var(--accent-orange)'
    return 'var(--accent-red)'
  }

  const getLabel = () => {
    if (score < 0.3) return 'Low Risk'
    if (score < 0.6) return 'Moderate'
    if (score < 0.8) return 'High Risk'
    return 'Critical'
  }

  return (
    <div className="score-gauge">
      <svg viewBox="0 0 120 120" className="gauge-svg">
        {/* Background arc */}
        <circle
          cx="60"
          cy="60"
          r="54"
          fill="none"
          stroke="var(--border)"
          strokeWidth="8"
          strokeDasharray={circumference}
          strokeDashoffset={circumference * 0.25}
          strokeLinecap="round"
          transform="rotate(135 60 60)"
        />
        {/* Value arc */}
        <circle
          cx="60"
          cy="60"
          r="54"
          fill="none"
          stroke={getColor()}
          strokeWidth="8"
          strokeDasharray={circumference}
          strokeDashoffset={offset}
          strokeLinecap="round"
          transform="rotate(135 60 60)"
          className="gauge-value"
        />
      </svg>
      <div className="gauge-center">
        <span className="gauge-number" style={{ color: getColor() }}>
          {percentage}
        </span>
        <span className="gauge-percent">%</span>
        <span className="gauge-label">{getLabel()}</span>
      </div>
    </div>
  )
}

export default ScoreGauge
