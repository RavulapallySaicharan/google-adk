from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List, Dict
import os
import asyncio
import uuid
import importlib
import pkgutil
from pathlib import Path
from dotenv import load_dotenv
from google.adk.runners import Runner
from google.adk.sessions import DatabaseSessionService
from google.adk.artifacts import InMemoryArtifactService
from google.adk.memory import InMemoryMemoryService
from google.genai import types

# Load environment variables
load_dotenv()

app = FastAPI(title="Reminder Agent API")

# Create services
session_service = DatabaseSessionService(
    db_url=os.getenv("DATABASE_URL", "sqlite:///agent_sessions.db")
)
artifact_service = InMemoryArtifactService()
memory_service = InMemoryMemoryService()

def load_agents():
    """Dynamically load all agents from the agent directory."""
    agents = {}
    agent_dir = Path("agent")
    
    if not agent_dir.exists():
        return agents
    
    # Import the agent package
    import agent
    
    # Walk through all subdirectories in the agent directory
    for _, name, is_pkg in pkgutil.iter_modules([str(agent_dir)]):
        if is_pkg:
            try:
                # Import the agent module
                module = importlib.import_module(f"agent.{name}.agent")
                # Get the agent instance (assuming it's named after the directory)
                agent_instance = getattr(module, name.replace("-", "_"))
                agents[name] = agent_instance
            except (ImportError, AttributeError) as e:
                print(f"Error loading agent {name}: {str(e)}")
                continue
    
    return agents

# Load all available agents
AVAILABLE_AGENTS = load_agents()

# Define initial state for new sessions
initial_state = {
    "username": "User",
    "reminders": []
}

# Define discoverable agents based on loaded agents
DISCOVERABLE_AGENTS = {
    name: {
        "name": agent.name,
        "description": agent.description,
        "capabilities": agent.instruction.split("\n") if agent.instruction else []
    }
    for name, agent in AVAILABLE_AGENTS.items()
}

class AgentRequest(BaseModel):
    message: str
    user_id: str = "admin"
    app_name: str = "ReminderApp"

class AgentResponse(BaseModel):
    response: str
    session_id: str

class SessionInfo(BaseModel):
    id: str
    user_id: str
    app_name: str
    created_at: str

class DiscoverableAgent(BaseModel):
    name: str
    description: str
    capabilities: List[str]

async def get_or_create_session(user_id: str, app_name: str) -> str:
    """Get existing session or create a new one for the user."""
    existing_sessions = await session_service.list_sessions(
        app_name=app_name,
        user_id=user_id
    )
    
    if existing_sessions.sessions and len(existing_sessions.sessions) > 0:
        # Use the most recent session
        return existing_sessions.sessions[0].id
    
    # Create a new session if none exists
    session_id = str(uuid.uuid4())
    await session_service.create_session(
        app_name=app_name,
        user_id=user_id,
        session_id=session_id,
        state=initial_state
    )
    return session_id

@app.post("/ask_agent", response_model=AgentResponse)
async def ask_agent(request: AgentRequest):
    try:
        # Get or create session automatically
        session_id = await get_or_create_session(request.user_id, request.app_name)

        # Get the requested agent
        agent = AVAILABLE_AGENTS.get(request.app_name)
        if not agent:
            raise HTTPException(status_code=404, detail=f"Agent {request.app_name} not found")

        # Create a runner with our agent and services
        runner = Runner(
            app_name=request.app_name,
            agent=agent,
            session_service=session_service,
            artifact_service=artifact_service,
            memory_service=memory_service
        )

        # Create a Content object for the user input
        user_message = types.Content(
            role="user",
            parts=[types.Part(text=request.message)]
        )

        # Process the user input
        response_text = ""
        async for event in runner.run_async(
            user_id=request.user_id,
            session_id=session_id,
            new_message=user_message
        ):
            if event.content and event.content.role == "agent":
                response_text = event.content.parts[0].text

        # Clean up the runner
        await runner.close()

        return AgentResponse(
            response=response_text,
            session_id=session_id
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/sessions", response_model=List[SessionInfo])
async def list_all_sessions():
    """Get all available sessions across all users and apps."""
    try:
        # Get all sessions from the database
        all_sessions = []
        # Note: This is a simplified version. In a real application,
        # you might want to add pagination and filtering
        sessions = await session_service.list_all_sessions()
        for session in sessions.sessions:
            all_sessions.append(SessionInfo(
                id=session.id,
                user_id=session.user_id,
                app_name=session.app_name,
                created_at=str(session.created_at)
            ))
        return all_sessions
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/sessions/{user_id}", response_model=List[SessionInfo])
async def list_user_sessions(user_id: str, app_name: str = "ReminderApp"):
    """Get all sessions for a specific user."""
    try:
        sessions = await session_service.list_sessions(
            app_name=app_name,
            user_id=user_id
        )
        return [SessionInfo(
            id=s.id,
            user_id=s.user_id,
            app_name=s.app_name,
            created_at=str(s.created_at)
        ) for s in sessions.sessions]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/discoverable_agent_flags", response_model=Dict[str, DiscoverableAgent])
async def get_discoverable_agents():
    """Get all discoverable agents and their capabilities."""
    return DISCOVERABLE_AGENTS

@app.delete("/session/{session_id}")
async def delete_session(session_id: str):
    """Delete a specific session."""
    try:
        # First check if the session exists
        session = await session_service.get_session(session_id=session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        
        # Delete the session
        await session_service.delete_session(session_id=session_id)
        return {"message": f"Session {session_id} deleted successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 