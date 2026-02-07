# API Contract: Chatbot Service

## Overview
This document defines the API contract for the chatbot service that integrates with the Neon PostgreSQL database.

## Base URL
`http://localhost:8000/api/v1` (development)
`https://api.yourdomain.com/v1` (production)

## Authentication
All endpoints require authentication via API key in the header:
```
Authorization: Bearer {your_neon_api_key}
```

## Endpoints

### POST /chat
Send a message to the chatbot and receive a response.

#### Request
```json
{
  "user_id": "string",
  "message": "string",
  "conversation_id": "string (optional)"
}
```

#### Response (Success 200)
```json
{
  "conversation_id": "string",
  "message_id": "string",
  "response": "string",
  "timestamp": "ISO 8601 datetime"
}
```

#### Response (Error 400)
```json
{
  "error": "string",
  "code": "string"
}
```

### GET /conversations
Retrieve all conversations for a specific user.

#### Query Parameters
- `user_id` (required): The ID of the user
- `limit` (optional): Number of conversations to return (default: 10)
- `offset` (optional): Offset for pagination (default: 0)

#### Response (Success 200)
```json
{
  "conversations": [
    {
      "id": "string",
      "title": "string",
      "created_at": "ISO 8601 datetime",
      "updated_at": "ISO 8601 datetime"
    }
  ],
  "total_count": "integer"
}
```

### GET /conversation/{id}
Retrieve a specific conversation by ID.

#### Path Parameters
- `id` (required): The conversation ID

#### Response (Success 200)
```json
{
  "id": "string",
  "title": "string",
  "created_at": "ISO 8601 datetime",
  "updated_at": "ISO 8601 datetime",
  "messages": [
    {
      "id": "string",
      "sender_type": "user|bot",
      "content": "string",
      "timestamp": "ISO 8601 datetime"
    }
  ]
}
```

### DELETE /conversation/{id}
Delete a specific conversation by ID.

#### Path Parameters
- `id` (required): The conversation ID

#### Response (Success 204)
No content returned.

#### Response (Error 404)
```json
{
  "error": "Conversation not found",
  "code": "CONVERSATION_NOT_FOUND"
}
```

## Error Codes
- `INVALID_INPUT`: Request data is malformed
- `AUTHENTICATION_FAILED`: Invalid or missing API key
- `DATABASE_ERROR`: Internal database error
- `CONVERSATION_NOT_FOUND`: Requested conversation does not exist
- `RATE_LIMIT_EXCEEDED`: Too many requests from the same user

## Rate Limiting
All endpoints are subject to rate limiting:
- 100 requests per minute per IP
- 1000 requests per hour per authenticated user