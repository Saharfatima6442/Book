"""
Isolated Flask chatbot service with Neon Auth integration.
Uses environment variables for PostgreSQL and Neon Auth configuration.
"""

import os
import logging
from flask import Flask, request, jsonify
from flask_cors import CORS
import psycopg2
from psycopg2.extras import RealDictCursor
import jwt
from functools import wraps

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Configuration from environment
DATABASE_URL = os.getenv('DATABASE_URL')
NEON_AUTH_URL = os.getenv('NEON_AUTH_URL')
JWKS_URL = os.getenv('JWKS_URL')
JWT_SECRET = os.getenv('JWT_SECRET')

def get_db_connection():
    """Create a connection to the Neon database."""
    try:
        conn = psycopg2.connect(DATABASE_URL)
        return conn
    except Exception as e:
        logger.error(f"Database connection error: {e}")
        raise

def verify_token(f):
    """Decorator to verify JWT token from request headers."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        token = None
        
        # Check for token in Authorization header
        if 'Authorization' in request.headers:
            try:
                token = request.headers['Authorization'].split(' ')[1]
            except IndexError:
                return jsonify({'error': 'Invalid token format'}), 401
        
        if not token:
            return jsonify({'error': 'Token missing'}), 401
        
        try:
            # In production, fetch JWKS from JWKS_URL and verify
            # For now, we verify locally if JWT_SECRET is provided
            if JWT_SECRET:
                payload = jwt.decode(token, JWT_SECRET, algorithms=['HS256'])
                request.user_id = payload.get('sub')
            else:
                logger.warning("JWT_SECRET not set, skipping token verification")
                request.user_id = 'anonymous'
        except jwt.ExpiredSignatureError:
            return jsonify({'error': 'Token expired'}), 401
        except jwt.InvalidTokenError:
            return jsonify({'error': 'Invalid token'}), 401
        
        return f(*args, **kwargs)
    
    return decorated_function

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({'status': 'ok'}), 200

@app.route('/chat', methods=['POST'])
@verify_token
def chat():
    """Chat endpoint that processes messages."""
    try:
        data = request.get_json()
        message = data.get('message', '').strip()
        
        if not message:
            return jsonify({'error': 'Message cannot be empty'}), 400
        
        user_id = request.user_id
        conn = get_db_connection()
        cur = conn.cursor(cursor_factory=RealDictCursor)
        
        # Store message in database
        cur.execute(
            """
            INSERT INTO chat_messages (user_id, message, created_at)
            VALUES (%s, %s, NOW())
            RETURNING id, user_id, message, created_at
            """,
            (user_id, message)
        )
        result = cur.fetchone()
        conn.commit()
        cur.close()
        conn.close()
        
        # TODO: Process message with LLM (e.g., OpenAI, Local LM Studio)
        response_text = f"Echo: {message}"
        
        return jsonify({
            'id': result['id'],
            'user_id': result['user_id'],
            'message': result['message'],
            'response': response_text,
            'created_at': str(result['created_at'])
        }), 200
    
    except Exception as e:
        logger.error(f"Chat error: {e}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/messages', methods=['GET'])
@verify_token
def get_messages():
    """Retrieve chat messages for the authenticated user."""
    try:
        user_id = request.user_id
        conn = get_db_connection()
        cur = conn.cursor(cursor_factory=RealDictCursor)
        
        cur.execute(
            """
            SELECT id, user_id, message, created_at
            FROM chat_messages
            WHERE user_id = %s
            ORDER BY created_at DESC
            LIMIT 100
            """,
            (user_id,)
        )
        messages = cur.fetchall()
        cur.close()
        conn.close()
        
        return jsonify({'messages': messages}), 200
    
    except Exception as e:
        logger.error(f"Get messages error: {e}")
        return jsonify({'error': 'Internal server error'}), 500

@app.before_request
def create_tables():
    """Create tables if they don't exist (runs once on startup)."""
    if not hasattr(app, '_tables_created'):
        try:
            conn = get_db_connection()
            cur = conn.cursor()
            
            cur.execute("""
                CREATE TABLE IF NOT EXISTS chat_messages (
                    id SERIAL PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    message TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT NOW()
                )
            """)
            conn.commit()
            cur.close()
            conn.close()
            
            app._tables_created = True
            logger.info("Database tables initialized")
        except Exception as e:
            logger.error(f"Table creation error: {e}")

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.getenv('PORT', 5000)), debug=os.getenv('DEBUG', 'False') == 'True')
