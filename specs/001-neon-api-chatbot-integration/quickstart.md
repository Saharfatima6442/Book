# Quickstart Guide: Neon API Chatbot Integration

## Prerequisites

- Python 3.11+
- pip package manager
- Git
- Access to Neon PostgreSQL database instance
- API key for Neon database

## Setup

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

## Running the Application

1. Activate the virtual environment:
   ```bash
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

2. Run the application:
   ```bash
   python src/main.py
   ```

## Running Tests

1. Make sure your virtual environment is activated
2. Run unit tests:
   ```bash
   pytest tests/unit/
   ```
   
3. Run integration tests:
   ```bash
   pytest tests/integration/
   ```

## API Endpoints

Once running, the chatbot API will be available at:
- `POST /chat` - Send a message to the chatbot
- `GET /conversations` - Get list of conversations for a user
- `GET /conversation/{id}` - Get a specific conversation
- `DELETE /conversation/{id}` - Delete a conversation

## Database Migrations

To initialize the database tables:
```bash
python -m src.database.init_db
```

## Troubleshooting

- If you get database connection errors, verify your Neon API key and connection string in the `.env` file
- If dependencies fail to install, ensure you're using Python 3.11+
- Check logs for detailed error messages if the application fails to start