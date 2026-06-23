# DocuMind Frontend

Chat-style Next.js frontend for DocuMind, created by Aman Singh Chauhan.

## Setup

Create `frontend/.env.local`:

```env
NEXT_PUBLIC_API_URL=http://localhost:8888
```

Do not add `GROQ_API_KEY` or database credentials to the frontend.

## Run

```bash
cd frontend
npm install
npm run dev
```

The UI calls `POST /ask`, `POST /upload`, `GET /knowledge`, `GET /conversations`, and `GET /documents` through `NEXT_PUBLIC_API_URL`.

## Test

1. Start FastAPI on port 8888 and run the frontend.
2. Upload a `.txt` file and confirm the knowledge status updates.
3. Ask a related question and expand its source chunks.
4. Stop FastAPI and confirm the frontend reports that the backend is unreachable.
