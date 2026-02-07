# Neon Chatbot Service

Isolated Flask chatbot service integrated with Neon PostgreSQL and Neon Auth. Does **not** interfere with the existing book project.

## Features

- **JWT Authentication**: Integrates with Neon Auth for secure user authentication.
- **PostgreSQL Backend**: Stores chat messages in Neon PostgreSQL database.
- **CORS Enabled**: Ready for frontend integration.
- **Logging**: Comprehensive error and info logging.
- **Health Check**: `/health` endpoint for monitoring.

## Setup

### 1. Create a Python Virtual Environment

```powershell
python -m venv venv
venv\Scripts\Activate.ps1
```

### 2. Install Dependencies

```powershell
pip install -r requirements.txt
```

### 3. Configure Environment Variables

Copy `.env.example` to `.env` and update with your Neon credentials:

```powershell
copy .env.example .env
```

Edit `.env` and replace:
- `DATABASE_URL`: Your Neon connection string (from `damp-fire-74721367` project)
- `JWT_SECRET`: A random secret key (e.g., `python -c "import secrets; print(secrets.token_hex(32))"`)

**Neon Credentials from Project `damp-fire-74721367`:**
```
Host: ep-weathered-water-ak2bzb9p-pooler.c-3.us-west-2.aws.neon.tech
Database: neondb
User: neondb_owner
Password: npg_Leb34VSvUAwt (replace with actual password from Neon console)
Neon Auth URL: https://ep-weathered-water-ak2bzb9p.neonauth.c-3.us-west-2.aws.neon.tech/neondb/auth
JWKS URL: https://ep-weathered-water-ak2bzb9p.neonauth.c-3.us-west-2.aws.neon.tech/neondb/auth/.well-known/jwks.json
```

## Running the Service

```powershell
python app.py
```

Server runs on `http://localhost:5000`.

## API Endpoints

### Health Check
```
GET /health
```
Returns: `{ "status": "ok" }`

### Send a Message (Requires Authentication)
```
POST /chat
Authorization: Bearer <JWT_TOKEN>
Content-Type: application/json

{
  "message": "Hello, chatbot!"
}
```

Response:
```json
{
  "id": 1,
  "user_id": "user-123",
  "message": "Hello, chatbot!",
  "response": "Echo: Hello, chatbot!",
  "created_at": "2026-02-06T10:30:00"
}
```

### Retrieve Messages (Requires Authentication)
```
GET /messages
Authorization: Bearer <JWT_TOKEN>
```

Response:
```json
{
  "messages": [
    {
      "id": 1,
      "user_id": "user-123",
      "message": "Hello, chatbot!",
      "created_at": "2026-02-06T10:30:00"
    }
  ]
}
```

## Testing with cURL

### 1. Health Check
```powershell
curl http://localhost:5000/health
```

### 2. Send Message (with dummy token)
```powershell
$token = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJ1c2VyLTEyMyJ9.XXX"
curl -X POST http://localhost:5000/chat `
  -H "Authorization: Bearer $token" `
  -H "Content-Type: application/json" `
  -d '{"message":"Hello"}'
```

## Database Schema

The service auto-creates the `chat_messages` table:

```sql
CREATE TABLE chat_messages (
    id SERIAL PRIMARY KEY,
    user_id TEXT NOT NULL,
    message TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT NOW()
);
```

## Integration with Book Project

This chatbot service is **completely isolated** from the book project and runs independently:
- Uses a **separate virtual environment** (`venv` locally)
- Connects to the **same Neon database** (`neondb` in `damp-fire-74721367`)
- Can be **deployed separately** (e.g., Docker, Heroku, AWS Lambda)

## Next Steps

1. Replace echo response with actual LLM integration (OpenAI, LM Studio, Ollama, etc.)
2. Add authentication/login endpoints using Neon Auth
3. Implement conversation history and context management
4. Add rate limiting and security middleware
5. Deploy to production (Docker, Heroku, etc.)

## Troubleshooting

**Connection Error?**
- Verify `DATABASE_URL` is correct in `.env`
- Check Neon project status: https://console.neon.tech/app/projects/damp-fire-74721367

**Missing Dependencies?**
```powershell
pip install -r requirements.txt
```

**CORS Issues?**
- Check that frontend is sending requests to `http://localhost:5000`
- CORS is enabled for all origins by default (restrict in production)
