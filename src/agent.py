import os
from datetime import datetime
from zoneinfo import ZoneInfo
from dotenv import load_dotenv
from google.adk.agents import LlmAgent
from google.adk.models.lite_llm import LiteLlm
from .tools.weather_time import get_weather, get_current_time
from .sessions.memory import InMemorySession
from .sessions.database import DatabaseSession

# Load environment variables
load_dotenv()

# Azure OpenAI configuration
AZURE_API_KEY = os.getenv("AZURE_API_KEY")
AZURE_API_BASE = os.getenv("AZURE_API_BASE")
AZURE_API_VERSION = os.getenv("AZURE_API_VERSION")

class WeatherTimeAgent:
    def __init__(self, use_database: bool = False):
        # Initialize session management
        self.session_manager = DatabaseSession() if use_database else InMemorySession()
        
        # Initialize the agent
        self.agent = LlmAgent(
            name="weather_time_agent",
            model=LiteLlm(model="azure/gpt-4"),  # Using Azure GPT-4
            description="Agent to answer questions about the time and weather in a city.",
            instruction="I can answer your questions about the time and weather in a city.",
            tools=[get_weather, get_current_time]
        )
    
    def process_query(self, session_id: str, query: str) -> str:
        """Process a user query and maintain session context."""
        # Get or create session
        session_data = self.session_manager.get_session(session_id)
        if not session_data:
            self.session_manager.create_session(session_id)
            session_data = {}
        
        # Update session with query
        session_data["last_query"] = query
        session_data["timestamp"] = datetime.now().isoformat()
        self.session_manager.update_session(session_id, session_data)
        
        # Process query with agent
        response = self.agent.process(query)
        
        # Update session with response
        session_data["last_response"] = response
        self.session_manager.update_session(session_id, session_data)
        
        return response

def main():
    # Example usage
    agent = WeatherTimeAgent(use_database=False)  # Set to True for database sessions
    
    # Example session
    session_id = "user_123"
    queries = [
        "What's the weather in New York?",
        "What time is it in London?",
        "Tell me about the weather in Tokyo"
    ]
    
    for query in queries:
        response = agent.process_query(session_id, query)
        print(f"Query: {query}")
        print(f"Response: {response}\n")

if __name__ == "__main__":
    main() 