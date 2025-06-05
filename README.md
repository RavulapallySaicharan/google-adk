# Google ADK Agent Implementation

This project implements a Google ADK agent with session management capabilities using both in-memory and database storage.

## Prerequisites

- Python 3.8 or higher
- Azure OpenAI API credentials
- SQLite (for database session storage)

## Environment Setup

1. Create a `.env` file in the root directory with the following variables:
```
AZURE_API_KEY=your_azure_api_key
AZURE_API_BASE=your_azure_api_base_url
AZURE_API_VERSION=your_azure_api_version
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Project Structure

```
google-adk/
├── README.md
├── requirements.txt
├── .env
├── src/
│   ├── __init__.py
│   ├── agent.py
│   ├── models.py
│   ├── sessions/
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── memory.py
│   │   └── database.py
│   └── tools/
│       ├── __init__.py
│       └── weather_time.py
└── tests/
    └── __init__.py
```

## Running the Agent

1. Make sure your environment variables are set up correctly in the `.env` file.

2. Run the agent:
```bash
python src/agent.py
```

## Features

- In-memory session management
- Database-backed session persistence
- Weather and time information retrieval
- Azure OpenAI integration

## Session Management

The agent supports two types of session management:
1. In-memory sessions (default)
2. Database sessions (SQLite)

To switch between session types, modify the session configuration in `src/agent.py`. 