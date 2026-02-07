# Data Model for Neon API Chatbot Integration

## Entities

### Conversation
Represents a chat session with unique identifier, timestamp, user queries, and chatbot responses

**Fields**:
- id (UUID, primary key)
- created_at (timestamp, not null)
- updated_at (timestamp, not null)
- title (string, nullable) - auto-generated from first message
- user_id (UUID, foreign key to User)

**Validation**:
- created_at must be in the past
- updated_at must be >= created_at

### User
Represents a chatbot user with preferences and historical interaction data

**Fields**:
- id (UUID, primary key)
- username (string, unique, not null)
- email (string, unique, nullable)
- preferences (JSON, nullable) - user preferences in JSON format
- created_at (timestamp, not null)
- updated_at (timestamp, not null)

**Validation**:
- username must be 3-30 characters
- email must be valid email format if provided
- preferences must be valid JSON

### Message
Individual message within a conversation containing content, timestamp, and sender type

**Fields**:
- id (UUID, primary key)
- conversation_id (UUID, foreign key to Conversation, not null)
- sender_type (enum: 'user' | 'bot', not null)
- content (text, not null)
- created_at (timestamp, not null)
- parent_message_id (UUID, foreign key to Message, nullable) - for threaded conversations

**Validation**:
- content must be 1-10000 characters
- created_at must be in the past
- sender_type must be 'user' or 'bot'

## Relationships

- User (1) → (Many) Conversation
- Conversation (1) → (Many) Message
- Message (0..1) → (1) Message (parent-child relationship for threading)

## State Transitions

None required for this feature - all entities are persistent with CRUD operations.

## Indexes

- Conversation: index on user_id, created_at
- User: index on username, email
- Message: index on conversation_id, created_at