import os
import asyncio
import uuid
from dotenv import load_dotenv
from google.adk.orchestration import Runner
from google.adk.orchestration.session import DatabaseSessionService
from agent.memory_agent import memory_agent

# Load environment variables
load_dotenv()

async def main():
    # Create a database session service
    session_service = DatabaseSessionService(
        database_url=os.getenv("DATABASE_URL", "sqlite:///agent_sessions.db")
    )

    # Define initial state for new sessions
    initial_state = {
        "username": "User",
        "reminders": []
    }

    # Application and user identifiers
    app_name = "ReminderApp"
    user_id = "example_user"

    # Check if we have an existing session for this user
    existing_sessions = session_service.list_sessions(
        app_name=app_name,
        user_id=user_id
    )

    if existing_sessions and len(existing_sessions) > 0:
        # Use the existing session
        session_id = existing_sessions[0].id
        print(f"Continuing existing session: {session_id}")
    else:
        # Create a new session
        session_id = str(uuid.uuid4())
        session_service.create_session(
            app_name=app_name,
            user_id=user_id,
            session_id=session_id,
            state=initial_state
        )
        print(f"Created new session: {session_id}")

    # Create a runner with our agent and session service
    runner = Runner(
        root_agent=memory_agent,
        session_service=session_service
    )

    # Interactive chat loop
    print("\nReminder Agent Chat (Type 'exit' or 'quit' to end)")
    print("--------------------------------------------------------")
    
    while True:
        user_input = input("\nYou: ")
        if user_input.lower() in ["exit", "quit"]:
            print("Goodbye! Your reminders have been saved.")
            break

        # Process the user input
        response = await runner.run_async(
            user_id=user_id,
            session_id=session_id,
            content=user_input
        )

        # Print the agent's response
        for event in response.events:
            if event.type == "content" and event.content.role == "agent":
                print(f"\nAgent: {event.content.parts[0].text}")

if __name__ == "__main__":
    asyncio.run(main()) 