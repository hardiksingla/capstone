import { useState } from 'react'
import UploadSection from './components/UploadSection.jsx'
import ResultsDashboard from './components/ResultsDashboard.jsx'

const API_URL = 'http://localhost:5000/api/analyze'

function App() {
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  const handleUpload = async (file) => {
    setLoading(true)
    setError(null)
    setResult(null)

    const formData = new FormData()
    formData.append('file', file)

    try {
      const res = await fetch(API_URL, { method: 'POST', body: formData })
      const data = await res.json()
      if (!res.ok) throw new Error(data.error || 'Analysis failed')
      setResult(data)
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  const handleReset = () => {
    setResult(null)
    setError(null)
  }

  return (
    <div className="app">
      <header className="app-header">
        <div className="header-content">
          <div className="logo">
            <span className="logo-icon">&#9678;</span>
            <h1>BBIA-OCA</h1>
          </div>
          <p className="subtitle">Behavioral Biometrics Based Integrity Analysis</p>
        </div>
      </header>

      <main className="app-main">
        {!result ? (
          <UploadSection
            onUpload={handleUpload}
            loading={loading}
            error={error}
          />
        ) : (
          <ResultsDashboard result={result} onReset={handleReset} />
        )}
      </main>

      <footer className="app-footer">
        <p>Online Coding Assessment Integrity System</p>
      </footer>
    </div>
  )
}

export default App
