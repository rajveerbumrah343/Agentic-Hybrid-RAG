import React, { useState } from 'react';

export default function App() {
  const [file, setFile] = useState(null);
  const [uploadStatus, setUploadStatus] = useState('');
  const [question, setQuestion] = useState('');
  const [loading, setLoading] = useState(false);
  const [response, setResponse] = useState(null);

  const handleUpload = async (e) => {
    e.preventDefault();
    if (!file) return;
    const formData = new FormData();
    formData.append('file', file);

    setUploadStatus('Uploading and indexing...');
    try {
      const res = await fetch('http://localhost:8009/upload', {
        method: 'POST',
        body: formData,
      });
      const data = await res.json();
      if (res.ok) {
        setUploadStatus(`Indexed successfully: ${data.info.filename} (${data.info.pages} pages)`);
      } else {
        setUploadStatus(`Error: ${data.detail}`);
      }
    } catch (err) {
      setUploadStatus('Upload failed.');
    }
  };

  const handleAsk = async (e) => {
    e.preventDefault();
    if (!question) return;
    setLoading(true);
    setResponse(null);

    try {
      const res = await fetch('http://localhost:8009/ask', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question }),
      });
      const data = await res.json();
      if (res.ok) {
        setResponse(data);
      } else {
        setResponse({ answer: data.detail, sources: [], steps: [] });
      }
    } catch (err) {
      setResponse({ answer: 'Failed to connect to backend.', sources: [], steps: [] });
    }
    setLoading(false);
  };

  return (
    <div style={{ maxWidth: '700px', margin: '40px auto', fontFamily: 'Arial, sans-serif', padding: '20px' }}>
      <h2>Ask your PDF: Agentic Hybrid RAG</h2>

      {/* Step 1: Upload PDF */}
      <form onSubmit={handleUpload} style={{ marginBottom: '20px' }}>
        <label><strong>1. Upload PDF Document:</strong></label><br />
        <input 
          type="file" 
          accept=".pdf,.txt" 
          onChange={(e) => setFile(e.target.files[0])} 
          style={{ marginTop: '5px' }}
        />
        <button type="submit" style={{ marginLeft: '10px', padding: '6px 12px' }}>Upload & Index</button>
        <p style={{ fontSize: '14px', color: '#555' }}>{uploadStatus}</p>
      </form>

      {/* Step 2: Ask Question */}
      <form onSubmit={handleAsk} style={{ marginBottom: '20px' }}>
        <label><strong>2. Ask a Question:</strong></label><br />
        <input 
          type="text" 
          value={question} 
          onChange={(e) => setQuestion(e.target.value)} 
          placeholder="What is this document about?" 
          style={{ width: '100%', padding: '8px', marginTop: '5px' }}
        />
        <button type="submit" disabled={loading} style={{ marginTop: '10px', padding: '8px 16px' }}>
          {loading ? 'Thinking...' : 'Ask Agent'}
        </button>
      </form>

      {/* Results Section */}
      {response && (
        <div style={{ background: '#f4f4f4', padding: '15px', borderRadius: '5px', marginTop: '20px' }}>
          <h3>Answer</h3>
          <p style={{ whiteSpace: 'pre-wrap' }}>{response.answer}</p>
          
          {response.sources && response.sources.length > 0 && (
            <p><strong>Sources:</strong> {response.sources.join('; ')}</p>
          )}

          <h4>How it was found:</h4>
          <ul>
            {response.steps.map((step, idx) => (
              <li key={idx} style={{ fontSize: '14px', color: '#666' }}>{step}</li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}