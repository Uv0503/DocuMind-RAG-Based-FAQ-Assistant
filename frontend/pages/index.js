import { useEffect, useRef, useState } from 'react'

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8888'
const NETWORK_ERROR =
  'Backend is not reachable. Please make sure FastAPI is running on port 8888.'

function message(role, content, sources = [], loading = false) {
  return {
    id: `${Date.now()}-${Math.random()}`,
    role,
    content,
    sources,
    loading,
  }
}

function apiError(error) {
  if (error instanceof TypeError) return NETWORK_ERROR
  return error.message || 'Something went wrong. Please try again.'
}

async function readResponse(response) {
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
  const fileRef = useRef(null)
  const endRef = useRef(null)

  const get = async (path) => readResponse(await fetch(API_URL + path))

  const refreshKnowledge = async () => {
    try {
      setKnowledge(await get('/knowledge'))
      setKnowledgeError(false)
    } catch {
      setKnowledgeError(true)
    }
  }

  const refreshHistory = async () => {
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
    refreshKnowledge()
    refreshHistory()
  }, [])

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  const ask = async (event) => {
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
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question: text }),
      })
      const data = await readResponse(response)

      setMessages((current) =>
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
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault()
      event.currentTarget.form.requestSubmit()
    }
  }

  const selectFile = (event) => {
    const file = event.target.files?.[0]
    setUploadMessage('')

    if (!file) return

    if (!file.name.endsWith('.txt')) {
      event.target.value = ''
      setUploadError('Only .txt files are supported.')
      return
    }

    setSelectedFile(file)
    setUploadError('')
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
      const form = new FormData()
      form.append('file', selectedFile)

      const response = await fetch(API_URL + '/upload', {
        method: 'POST',
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

  return (
    <main className={collapsed ? 'app is-collapsed' : 'app'}>
      <aside className="sidebar">
        <button
          className="sidebar-toggle"
          type="button"
          onClick={() => setCollapsed((value) => !value)}
        >
          {collapsed ? '>' : '<'}
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

          <section>
            <label>Knowledge file</label>
            <input
              ref={fileRef}
              type="file"
              accept=".txt"
              onChange={selectFile}
            />
            <p>{selectedFile?.name || 'Choose a .txt file'}</p>
            <button
              type="button"
              onClick={upload}
              disabled={!selectedFile || uploading}
            >
              {uploading ? 'Uploading...' : 'Upload document'}
            </button>
            {uploadError && <em className="error">{uploadError}</em>}
            {uploadMessage && <em className="success">{uploadMessage}</em>}
          </section>

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
