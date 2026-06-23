````markdown
# DocuMind API Documentation

**Created by**: Aman Singh Chauhan

**Version**: 0.1.0  
**OpenAPI Spec**: 3.1  
**Base URL**: [https://llm-faq-bot.onrender.com](https://llm-faq-bot.onrender.com)  
**OpenAPI JSON**: `/openapi.json`  

---

## Endpoints

---

### `GET /` - Root

**Description**: Returns a welcome message.

#### Curl
```bash
curl -X 'GET' \
  'https://llm-faq-bot.onrender.com/' \
  -H 'accept: application/json'
````

#### Response

**Status Code**: `200 OK`
**Body**:

```json
{
  "message": "Welcome to the LLM FAQ Bot API. Visit /docs to test it."
}
```

**Headers**:

```
content-type: application/json
content-encoding: br
x-render-origin-server: uvicorn
...
```

---

### `GET /health/db` - Health DB

**Description**: Checks database connectivity and returns available tables.

#### Curl

```bash
curl -X 'GET' \
  'https://llm-faq-bot.onrender.com/health/db' \
  -H 'accept: application/json'
```

#### Response

**Status Code**: `200 OK`
**Body**:

```json
[
  "conversations"
]
```

---

### `GET /conversations` - List Conversations

**Description**: Lists all stored conversations.

#### Curl

```bash
curl -X 'GET' \
  'https://llm-faq-bot.onrender.com/conversations' \
  -H 'accept: application/json'
```

#### Response

**Status Code**: `200 OK`
**Body** (Example):

```json
[
  {
    "id": 3,
    "question": "What's your name?",
    "context": "...(context string)...",
    "response": "Hello! I'm DocuMind, created by Aman Singh Chauhan to help answer your questions and provide information. How can I assist you today?",
    "created_at": "2025-06-29T14:18:55.757851"
  },
  {
    "id": 2,
    "question": "Who are you?",
    "context": "...(context string)...",
    "response": "Hello! I'm DocuMind, a friendly question-answer assistant created by Aman Singh Chauhan. I'm here to help answer any questions you may have, provide information, and engage in small talk. How can I assist you today?",
    "created_at": "2025-06-29T13:12:34.237462"
  }
]
```

---

### `POST /ask` - Ask a Question

**Description**: Sends a question to the bot and receives an answer.

#### Request Body

```json
{
  "question": "What is your name?"
}
```

#### Curl

```bash
curl -X 'POST' \
  'https://llm-faq-bot.onrender.com/ask' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "question": "What is your name?"
}'
```

#### Response

**Status Code**: `200 OK`
**Body**:

```json
{
  "answer": "Hello! I'm DocuMind, created by Aman Singh Chauhan to assist with questions and provide helpful information. How can I help you today?"
}
```

**Possible Errors**:

* `422 Unprocessable Entity`: Validation error

---

## Schemas

### `Answer` (object)

```json
{
  "answer": "string"
}
```

### `Conversation` (object)

```json
{
  "id": 0,
  "question": "string",
  "context": "string",
  "response": "string",
  "created_at": "2025-06-29T14:44:24.095Z"
}
```

### `HTTPValidationError` (object)

```json
{
  "detail": [
    {
      "loc": ["string", 0],
      "msg": "string",
      "type": "string"
    }
  ]
}
```

---

## Notes

* Created by **Aman Singh Chauhan**
* Frontend: Next.js (port 3000)
* Backend: FastAPI (port 8888)
* Every interaction is logged in a PostgreSQL database
* CORS is enabled for local and deployed frontend
* Behavior is governed by an internal `CONTEXT` prompt