# Google ADK Memory Agent with Azure OpenAI

This project implements a memory-enabled agent using Google's Agent Development Kit (ADK) and Azure OpenAI, with persistent session management using SQLite.

## Features

- Persistent session management using SQLite database
- Azure OpenAI integration via LiteLLM
- Reminder management capabilities
- User session persistence
- Interactive chat interface

## Prerequisites

- Python 3.8 or higher
- Azure OpenAI API access
- Google ADK installed

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd <repository-name>
```

2. Create and activate a virtual environment:
```bash
python -m venv env
source env/bin/activate  # On Windows: env\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Create a `.env` file in the project root with the following variables:
```env
# Azure OpenAI Configuration
AZURE_API_KEY=your_azure_openai_key_here
AZURE_API_BASE=https://your-resource.openai.azure.com
AZURE_API_VERSION=2023-05-15
AZURE_API_TYPE=azure
AZURE_DEPLOYMENT_NAME=your_deployment_name

# Database Configuration
DATABASE_URL=sqlite:///agent_sessions.db

# Agent Configuration
AGENT_NAME=memory_agent
MODEL_NAME=azure/your_deployment_name
```

## Usage

Run the agent:
```bash
python main.py
```

The agent will start an interactive chat session where you can:
- Add reminders
- View existing reminders
- Delete reminders
- Update your username

Example interactions:
```
You: Add a reminder to buy groceries tomorrow
Agent: I've added the reminder: "buy groceries tomorrow"

You: What are my current reminders?
Agent: Here are your current reminders:
1. buy groceries tomorrow

You: Update my name to John
Agent: I've updated your name from 'User' to 'John'
```

## Project Structure

```
.
├── agent/
│   └── memory_agent.py    # Agent implementation
├── main.py               # Main application
├── requirements.txt      # Dependencies
├── .env                 # Environment variables
└── README.md           # This file
```

## Session Management

The agent uses SQLite for persistent session storage. Sessions are automatically created for new users and maintained across application restarts. The database file (`agent_sessions.db`) will be created automatically in the project root directory.

## Troubleshooting

1. **Azure OpenAI Connection Issues**
   - Verify your Azure API credentials in the `.env` file
   - Check your Azure OpenAI deployment status
   - Ensure your API version is correct

2. **Database Issues**
   - If the database becomes corrupted, delete `agent_sessions.db` and restart the application
   - Ensure the application has write permissions in the project directory

3. **Memory/Session Issues**
   - Sessions are stored in the SQLite database
   - Each user gets a unique session ID
   - Sessions persist until explicitly deleted

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details. 