import { useEffect, useRef, useState } from 'react'

// Browser-visible environment variables must use Next.js's NEXT_PUBLIC_ prefix.
const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8888'
const NETWORK_ERROR =
  'Backend is not reachable. Please make sure FastAPI is running on port 8888.'

function message(role, content, sources = [], loading = false) {
  // A single message shape supports user messages, pending placeholders, and sourced answers.
  return {
    id: `${Date.now()}-${Math.random()}`,
    role,
    content,
    sources,
    loading,
  }
}

function apiError(error) {
  // fetch uses TypeError for transport failures; HTTP failures are normalized by readResponse.
  if (error instanceof TypeError) return NETWORK_ERROR
  return error.message || 'Something went wrong. Please try again.'
}

async function readResponse(response) {
  // Keep FastAPI's normal errors and Pydantic validation errors consistent for the UI.
  const body = await response.json().catch(() => ({}))

  if (response.ok) return body

  if (Array.isArray(body.detail)) {
    return (
      body.detail.map((item) => item.msg).filter(Boolean).join(', ') ||
      'Validation error.'
    )
  }

  throw new Error(body.detail || 'Request failed')
}

export default function Home() {
  // Local React state is divided into chat, upload, backend status/history, and layout concerns.
  const [messages, setMessages] = useState([])
  const [question, setQuestion] = useState('')
  const [loading, setLoading] = useState(false)
  const [uploading, setUploading] = useState(false)
  const [selectedFile, setSelectedFile] = useState(null)
  const [chatError, setChatError] = useState('')
  const [uploadError, setUploadError] = useState('')
  const [uploadMessage, setUploadMessage] = useState('')
  const [knowledge, setKnowledge] = useState(null)
  const [knowledgeError, setKnowledgeError] = useState(false)
  const [conversations, setConversations] = useState([])
  const [documents, setDocuments] = useState([])
  const [collapsed, setCollapsed] = useState(false)
  const [currentUser, setCurrentUser] = useState(null)
  const [authReady, setAuthReady] = useState(false)
  const [authMode, setAuthMode] = useState('login')
  const [authEmail, setAuthEmail] = useState('')
  const [authPassword, setAuthPassword] = useState('')
  const [authError, setAuthError] = useState('')
  const [authLoading, setAuthLoading] = useState(false)
  const fileRef = useRef(null)
  const endRef = useRef(null)
  const tokenRef = useRef(null)

  // All read-only API calls pass through the same JSON/error handling path.
  const authHeaders = () => tokenRef.current ? { Authorization: `Bearer ${tokenRef.current}` } : {}
  const get = async (path) => readResponse(await fetch(API_URL + path, { headers: authHeaders() }))

  const refreshKnowledge = async () => {
    try {
      setKnowledge(await get('/knowledge'))
      setKnowledgeError(false)
    } catch {
      setKnowledgeError(true)
    }
  }

  const refreshHistory = async () => {
    // History calls are independent: one failed service response does not hide the other panel.
    const results = await Promise.allSettled([
      get('/conversations'),
      get('/documents'),
    ])

    if (results[0].status === 'fulfilled') {
      setConversations(results[0].value.slice(0, 5))
    }

    if (results[1].status === 'fulfilled') {
      setDocuments(results[1].value.slice(0, 5))
    }
  }

  useEffect(() => {
    const restoreSession = async () => {
      const token = window.localStorage.getItem('documind_token')
      if (!token) {
        setAuthReady(true)
        return
      }
      tokenRef.current = token
      try {
        const data = await get('/auth/me')
        setCurrentUser(data.user)
      } catch {
        window.localStorage.removeItem('documind_token')
        tokenRef.current = null
      } finally {
        setAuthReady(true)
      }
    }
    restoreSession()
  }, [])

  useEffect(() => {
    if (!currentUser) return
    refreshKnowledge()
    refreshHistory()
  }, [currentUser])

  useEffect(() => {
    // Keep the newest chat message visible whenever the conversation array changes.
    endRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  const ask = async (event) => {
    // Optimistic UI: append the question and a temporary assistant row before the API returns.
    event.preventDefault()

    const text = question.trim()
    if (!text) {
      setChatError('Please enter a question.')
      return
    }

    const user = message('user', text)
    const pending = message('assistant', 'Thinking...', [], true)

    setMessages((current) => [...current, user, pending])
    setQuestion('')
    setChatError('')
    setLoading(true)

    try {
      const response = await fetch(API_URL + '/ask', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', ...authHeaders() },
        body: JSON.stringify({ question: text }),
      })
      const data = await readResponse(response)

      setMessages((current) =>
        // Replace only this request's placeholder, preserving every other message.
        current.map((item) =>
          item.id === pending.id
            ? message('assistant', data.answer, data.sources || [])
            : item
        )
      )
      refreshHistory()
    } catch (error) {
      setMessages((current) =>
        current.filter((item) => item.id !== pending.id)
      )
      setChatError(apiError(error))
    } finally {
      setLoading(false)
    }
  }

  const onKeyDown = (event) => {
    // Enter submits like a chat app; Shift+Enter remains available for a newline.
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault()
      event.currentTarget.form.requestSubmit()
    }
  }

  const selectFile = (event) => {
    // Client validation gives immediate feedback; the backend repeats validation for security.
    const file = event.target.files?.[0]
    setUploadMessage('')

    if (!file) return

    if (!file.name.toLowerCase().endsWith('.txt')) {
      event.target.value = ''
      setUploadError('Only .txt files are supported.')
      return
    }

    setSelectedFile(file)
    setUploadError('')
  }

  const cancelFile = () => {
    // Reset both React state and the native input so the same file can be selected again.
    setSelectedFile(null)
    setUploadError('')
    setUploadMessage('')
    if (fileRef.current) fileRef.current.value = ''
  }

  const upload = async () => {
    if (!selectedFile) {
      setUploadError('Please choose a .txt file to upload.')
      return
    }

    setUploading(true)
    setUploadError('')
    setUploadMessage('')

    try {
      // Files use multipart FormData, so the browser supplies the correct Content-Type boundary.
      const form = new FormData()
      form.append('file', selectedFile)

      const response = await fetch(API_URL + '/upload', {
        method: 'POST',
        headers: authHeaders(),
        body: form,
      })
      const data = await readResponse(response)

      setUploadMessage(
        data.message || 'Document uploaded and indexed successfully.'
      )
      setSelectedFile(null)

      if (fileRef.current) fileRef.current.value = ''

      await Promise.all([refreshKnowledge(), refreshHistory()])
    } catch (error) {
      setUploadError(apiError(error))
    } finally {
      setUploading(false)
    }
  }

  const openHistory = (item) => {
    // Rehydrate a persisted exchange into the current in-memory chat transcript.
    setMessages((current) => [
      ...current,
      message('user', item.question),
      message(
        'assistant',
        item.response,
        Array.isArray(item.sources) ? item.sources : []
      ),
    ])
  }

  const newChat = () => {
    setMessages([])
    setQuestion('')
    setChatError('')
  }

  const submitAuth = async (event) => {
    event.preventDefault()
    setAuthLoading(true)
    setAuthError('')
    try {
      const response = await fetch(API_URL + `/auth/${authMode}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ email: authEmail, password: authPassword }),
      })
      const data = await readResponse(response)
      tokenRef.current = data.token
      window.localStorage.setItem('documind_token', data.token)
      setCurrentUser(data.user)
      setAuthPassword('')
    } catch (error) {
      setAuthError(apiError(error))
    } finally {
      setAuthLoading(false)
    }
  }

  const logout = () => {
    window.localStorage.removeItem('documind_token')
    tokenRef.current = null
    setCurrentUser(null)
    setMessages([])
    setConversations([])
    setDocuments([])
  }

  if (!authReady) return <main className="auth-shell">Checking your session…</main>

  if (!currentUser) {
    return (
      <main className="auth-shell">
        <form className="auth-card" onSubmit={submitAuth}>
          <div className="auth-brand"><b>DM</b><span>DocuMind</span></div>
          <h1>{authMode === 'login' ? 'Welcome back' : 'Create your account'}</h1>
          <p>Sign in to keep your documents and conversations private.</p>
          <label>Email<input type="email" value={authEmail} onChange={(event) => setAuthEmail(event.target.value)} required /></label>
          <label>Password<input type="password" minLength="8" value={authPassword} onChange={(event) => setAuthPassword(event.target.value)} required /></label>
          {authError && <div className="error">{authError}</div>}
          <button disabled={authLoading}>{authLoading ? 'Please wait…' : authMode === 'login' ? 'Sign in' : 'Create account'}</button>
          <button className="auth-switch" type="button" onClick={() => { setAuthMode(authMode === 'login' ? 'register' : 'login'); setAuthError('') }}>
            {authMode === 'login' ? 'Need an account? Sign up' : 'Already have an account? Sign in'}
          </button>
        </form>
      </main>
    )
  }

  return (
    // The page has two visual regions: a knowledge/history sidebar and the active chat workspace.
    <main className={collapsed ? 'app is-collapsed' : 'app'}>
      <aside className="sidebar">
        <button
          className="sidebar-toggle"
          type="button"
          aria-label={collapsed ? 'Open sidebar' : 'Close sidebar'}
          aria-expanded={!collapsed}
          title={collapsed ? 'Open sidebar' : 'Close sidebar'}
          onClick={() => setCollapsed((value) => !value)}
        >
          <span aria-hidden="true">{collapsed ? '›' : '‹'}</span>
        </button>

        <div className="sidebar-body">
          <div className="brand">
            <b>DM</b>
            <div>
              <strong>DocuMind</strong>
              <small>Created by Aman Singh Chauhan</small>
            </div>
          </div>

          <button className="new-chat" type="button" onClick={newChat}>
            + New Chat
          </button>
          <button className="account-button" type="button" onClick={logout} title="Sign out">
            {currentUser.email}<span>Sign out</span>
          </button>

          {/* Upload controls activate a new backend knowledge source. */}
          <section className="knowledge-upload">
            <label>Knowledge file</label>
            <input
              ref={fileRef}
              id="knowledge-file"
              className="file-input"
              type="file"
              accept=".txt"
              onChange={selectFile}
            />
            {!selectedFile ? (
              <label className="file-picker" htmlFor="knowledge-file">
                <span className="file-picker-icon" aria-hidden="true">↑</span>
                <span>
                  <strong>Choose a file</strong>
                  <small>TXT documents only</small>
                </span>
              </label>
            ) : (
              <div className="file-selection" aria-live="polite">
                <div className="file-selection-name">
                  <span className="file-check" aria-hidden="true">✓</span>
                  <span title={selectedFile.name}>{selectedFile.name}</span>
                </div>
                <div className="file-actions">
                  <button
                    className="upload-action"
                    type="button"
                    onClick={upload}
                    disabled={uploading}
                  >
                    {uploading ? 'Uploading...' : 'Upload document'}
                  </button>
                  <button
                    className="cancel-action"
                    type="button"
                    onClick={cancelFile}
                    disabled={uploading}
                  >
                    Cancel
                  </button>
                </div>
              </div>
            )}
            {uploadError && <em className="error">{uploadError}</em>}
            {uploadMessage && <em className="success">{uploadMessage}</em>}
          </section>

          {/* Knowledge status reports the backend index, not merely the selected browser file. */}
          <section>
            <label>Knowledge status</label>
            {knowledgeError ? (
              <p>Knowledge status unavailable</p>
            ) : (
              <>
                <b>{knowledge?.chunks_indexed ?? '--'} chunks</b>
                <p>{knowledge?.active_sources?.join(', ') || 'No active source'}</p>
              </>
            )}
          </section>

          {/* These panels are backed by PostgreSQL and remain empty when persistence is unavailable. */}
          <section className="history-panel">
            <label>Recent questions</label>
            <div>
              {conversations.length ? (
                conversations.map((item) => (
                  <button
                    className="history-item"
                    type="button"
                    key={item.id}
                    onClick={() => openHistory(item)}
                  >
                    {item.question}
                  </button>
                ))
              ) : (
                <p>No saved questions yet</p>
              )}
            </div>
          </section>

          <section className="history-panel">
            <label>Recent documents</label>
            <div>
              {documents.length ? (
                documents.map((item) => (
                  <p key={item.id}>
                    {item.filename} - {item.chunks_created} chunks
                  </p>
                ))
              ) : (
                <p>No uploaded documents yet</p>
              )}
            </div>
          </section>
        </div>
      </aside>

      <section className="chat">
        <header>
          <div>
            <small>RAG assistant</small>
            <h1>Ask questions from your uploaded document</h1>
          </div>
          <span>
            {knowledge?.active_sources?.length
              ? 'Knowledge ready'
              : 'Waiting for a document'}
          </span>
        </header>

        <div className="messages">
          {!messages.length && (
            <div className="empty">
              <h2>Start a grounded conversation</h2>
              <p>Upload a text file, then ask a question.</p>
            </div>
          )}

          {/* Sources are rendered beside each completed assistant answer for auditability. */}
          {messages.map((item) => (
            <article className={item.role} key={item.id}>
              <b>{item.role === 'user' ? 'You' : 'AI'}</b>
              <div>
                <small>{item.role === 'user' ? 'You' : 'DocuMind'}</small>
                <p>{item.content}</p>

                {item.role === 'assistant' && !item.loading && (
                  <>
                    {item.sources.length ? (
                      <details>
                        <summary>Sources used ({item.sources.length})</summary>
                        {item.sources.map((source) => (
                          <div className="source" key={source.chunk_id}>
                            <b>Chunk {source.chunk_id}</b>
                            <span>{source.text}</span>
                          </div>
                        ))}
                      </details>
                    ) : (
                      <em>No source chunks found.</em>
                    )}
                  </>
                )}
              </div>
            </article>
          ))}

          <div ref={endRef} />
        </div>

        <footer>
          {chatError && <div className="error">{chatError}</div>}
          <form onSubmit={ask}>
            <textarea
              value={question}
              onChange={(event) => setQuestion(event.target.value)}
              onKeyDown={onKeyDown}
              placeholder="Ask something about your knowledge file..."
              rows={1}
              disabled={loading}
            />
            <button disabled={loading || !question.trim()}>
              {loading ? 'Thinking...' : 'Send'}
            </button>
          </form>
        </footer>
      </section>
    </main>
  )
}
