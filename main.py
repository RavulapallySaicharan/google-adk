import os
import asyncio
import uuid
from dotenv import load_dotenv
from google.adk.runners import Runner
from google.adk.sessions import DatabaseSessionService
from google.adk.artifacts import InMemoryArtifactService
from google.adk.memory import InMemoryMemoryService
from google.genai import types
from agent.memory_agent import memory_agent

# Load environment variables
load_dotenv()
async def main():
    # Create services
    session_service = DatabaseSessionService(
        db_url=os.getenv("DATABASE_URL", "sqlite:///agent_sessions.db")
    )
    artifact_service = InMemoryArtifactService()
    memory_service = InMemoryMemoryService()

    # Define initial state for new sessions
    initial_state = {
        "username": "User",
        "reminders": []
    }

    # Application and user identifiers
    app_name = "ReminderApp"
    user_id = "example_user"

    # Check if we have an existing session for this user
    existing_sessions = await session_service.list_sessions(
        app_name=app_name,
        user_id=user_id
    )

    if existing_sessions.sessions and len(existing_sessions.sessions) > 0:
        # Use the existing session
        session_id = existing_sessions.sessions[0].id
        print(f"Continuing existing session: {session_id}")
    else:
        # Create a new session
        session_id = str(uuid.uuid4())
        await session_service.create_session(
            app_name=app_name,
            user_id=user_id,
            session_id=session_id,
            state=initial_state
        )
        print(f"Created new session: {session_id}")

    # Create a runner with our agent and services
    runner = Runner(
        app_name=app_name,
        agent=memory_agent,
        session_service=session_service,
        artifact_service=artifact_service,
        memory_service=memory_service
    )

    # Interactive chat loop
    print("\nReminder Agent Chat (Type 'exit' or 'quit' to end)")
    print("--------------------------------------------------------")
    
    while True:
        user_input = input("\nYou: ")
        if user_input.lower() in ["exit", "quit"]:
            print("Goodbye! Your reminders have been saved.")
            break

        # Create a Content object for the user input
        user_message = types.Content(
            role="user",
            parts=[types.Part(text=user_input)]
        )

        # Process the user input
        async for event in runner.run_async(
            user_id=user_id,
            session_id=session_id,
            new_message=user_message
        ):
            if event.content and event.content.role == "agent":
                print(f"\nAgent: {event.content.parts[0].text}")

    # Clean up the runner
    await runner.close()

if __name__ == "__main__":
    asyncio.run(main()) 