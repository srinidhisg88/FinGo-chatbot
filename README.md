# Supply Chain Management AI Chat

This is an AI-powered chat application for supply chain management using Flask, PostgreSQL, Qdrant, and Google Gemini 2.0.

## Features

- Single endpoint for all types of supply chain queries
- Integration with Google Gemini 2.0 for AI responses
- PostgreSQL database for chat history
- Qdrant for vector storage and similarity search

## Prerequisites

- Python 3.8+
- PostgreSQL
- Qdrant
- Google Gemini API key

## Setup

1. Clone the repository
2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Copy `.env.example` to `.env` and fill in your configuration:
   ```bash
   cp .env.example .env
   ```
5. Set up PostgreSQL database:
   ```sql
   CREATE DATABASE supply_chain_db;
   ```
6. Start Qdrant:
   ```bash
   docker run -p 6333:6333 qdrant/qdrant
   ```

## Running the Application

1. Start the Flask application:
   ```bash
   python app.py
   ```
2. The application will be available at `http://localhost:5000`

## API Usage

Send a POST request to `/chat` endpoint with the following JSON body:
```json
{
    "query": "Your supply chain question here"
}
```

Example response:
```json
{
    "response": "AI-generated response",
    "timestamp": "2024-03-21T12:34:56.789Z"
}
```

## Error Handling

The API returns appropriate HTTP status codes:
- 200: Success
- 400: Bad Request (missing query)
- 500: Internal Server Error 