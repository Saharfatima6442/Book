# Neon API Chatbot Service

This service integrates with Neon PostgreSQL database to provide persistent chatbot functionality with conversation history and user preferences.

## Setup

### Prerequisites

- Python 3.11+
- Pip package manager
- Git
- Access to Neon PostgreSQL database instance
- API key for Neon database

### Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```

2. Navigate to the backend directory:
   ```bash
   cd backend
   ```

3. Create a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

4. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

5. Configure environment variables:
   - Copy `.env.example` to `.env`
   - Add your Neon database API key and connection string to the `.env` file

## Environment Variables

Create a `.env` file in the backend directory with the following variables:

```env
NEON_DB_URL=your_neon_database_url
NEON_API_KEY=your_neon_api_key
DB_POOL_SIZE=10
DB_POOL_OVERFLOW=20
LOG_LEVEL=INFO
```

### Secure API Key Management

The application implements secure handling of API keys with the following features:

1. **Environment Variable Storage**: The Neon API key is stored in the `.env` file and loaded at application startup using python-dotenv. This ensures credentials are not hardcoded in the source code.

2. **Validation**: The application validates the format of the API key at startup to ensure it meets expected patterns.

3. **Secure Logging**: A custom logging filter prevents the API key from being exposed in logs or error messages. Any occurrence of the API key in log messages is automatically replaced with `[HIDDEN_API_KEY]`.

4. **Runtime Checking**: The application performs runtime validation of the API key to ensure it remains valid during operation.

5. **Health Checks**: The `/health/config` endpoint validates that the API key exists and is properly formatted without exposing its value.

## Running the Application

1. Activate the virtual environment:
   ```bash
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

2. Run the application:
   ```bash
   python main.py
   ```

The application will be available at `http://localhost:8000`.

## API Endpoints

### Chat Endpoint
- `POST /chat` - Send a message to the chatbot and receive a response

### Conversations
- `GET /conversations?user_id={userId}&limit={limit}&offset={offset}` - Get user's conversations
- `GET /conversation/{id}` - Get a specific conversation
- `DELETE /conversation/{id}` - Delete a conversation

### User Preferences
- `GET /user/{id}/preferences` - Get user preferences
- `PUT /user/{id}/preferences` - Update user preferences

### Health Checks
- `GET /health` - Overall service health
- `GET /health/db` - Database connectivity
- `GET /health/config` - Configuration validation

## Security Features

1. **API Key Protection**: The Neon API key is never exposed in logs, error messages, or API responses.
2. **Input Validation**: All API inputs are validated to prevent injection attacks.
3. **Authentication**: API endpoints require proper authorization headers.
4. **Rate Limiting**: The service implements rate limiting to prevent abuse (coming soon).

## Database Schema

The application creates the following tables:

- `users`: Stores user information and preferences
- `conversations`: Tracks conversation threads
- `messages`: Stores individual messages within conversations

## Error Handling

The application implements comprehensive error handling with appropriate HTTP status codes:

- 400: Bad Request - Invalid input data
- 401: Unauthorized - Missing or invalid API key
- 404: Not Found - Requested resource not found
- 429: Too Many Requests - Rate limit exceeded (coming soon)
- 500: Internal Server Error - Unexpected server error

## Testing

To run the tests:

```bash
pytest tests/
```

The test suite includes:
- Unit tests for all models and services
- Integration tests for database operations
- Security tests to verify API key protection