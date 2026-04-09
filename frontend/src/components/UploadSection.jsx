import { useState, useRef } from 'react'
import './UploadSection.css'

function UploadSection({ onUpload, loading, error }) {
  const [dragActive, setDragActive] = useState(false)
  const [selectedFile, setSelectedFile] = useState(null)
  const inputRef = useRef(null)

  const handleDrag = (e) => {
    e.preventDefault()
    e.stopPropagation()
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setDragActive(true)
    } else if (e.type === 'dragleave') {
      setDragActive(false)
    }
  }

  const handleDrop = (e) => {
    e.preventDefault()
    e.stopPropagation()
    setDragActive(false)
    const file = e.dataTransfer.files[0]
    if (file && file.name.endsWith('.json')) {
      setSelectedFile(file)
    }
  }

  const handleChange = (e) => {
    const file = e.target.files[0]
    if (file) setSelectedFile(file)
  }

  const handleSubmit = () => {
    if (selectedFile) onUpload(selectedFile)
  }

  const formatSize = (bytes) => {
    if (bytes < 1024) return bytes + ' B'
    return (bytes / 1024).toFixed(1) + ' KB'
  }

  return (
    <div className="upload-section">
      <div className="upload-hero">
        <h2>Analyze Coding Session</h2>
        <p>Upload a session JSON file to detect behavioral anomalies and assess integrity</p>
      </div>

      <div
        className={`drop-zone ${dragActive ? 'active' : ''} ${selectedFile ? 'has-file' : ''}`}
        onDragEnter={handleDrag}
        onDragLeave={handleDrag}
        onDragOver={handleDrag}
        onDrop={handleDrop}
        onClick={() => inputRef.current?.click()}
      >
        <input
          ref={inputRef}
          type="file"
          accept=".json"
          onChange={handleChange}
          hidden
        />

        {!selectedFile ? (
          <div className="drop-content">
            <div className="drop-icon">
              <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
                <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
                <polyline points="17 8 12 3 7 8" />
                <line x1="12" y1="3" x2="12" y2="15" />
              </svg>
            </div>
            <p className="drop-text">Drag & drop session JSON here</p>
            <p className="drop-subtext">or click to browse files</p>
          </div>
        ) : (
          <div className="file-preview" onClick={(e) => e.stopPropagation()}>
            <div className="file-icon">
              <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
                <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
                <polyline points="14 2 14 8 20 8" />
                <line x1="16" y1="13" x2="8" y2="13" />
                <line x1="16" y1="17" x2="8" y2="17" />
              </svg>
            </div>
            <div className="file-info">
              <span className="file-name">{selectedFile.name}</span>
              <span className="file-size">{formatSize(selectedFile.size)}</span>
            </div>
            <button
              className="file-remove"
              onClick={() => setSelectedFile(null)}
              title="Remove file"
            >
              &times;
            </button>
          </div>
        )}
      </div>

      {error && (
        <div className="upload-error">
          <span className="error-icon">!</span>
          {error}
        </div>
      )}

      <button
        className="analyze-btn"
        onClick={handleSubmit}
        disabled={!selectedFile || loading}
      >
        {loading ? (
          <>
            <span className="spinner" />
            Analyzing Session...
          </>
        ) : (
          <>
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <circle cx="11" cy="11" r="8" />
              <line x1="21" y1="21" x2="16.65" y2="16.65" />
            </svg>
            Analyze Session
          </>
        )}
      </button>
    </div>
  )
}

export default UploadSection
