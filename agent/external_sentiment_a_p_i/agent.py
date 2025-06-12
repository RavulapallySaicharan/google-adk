from google.adk import Agent
from google.adk.models.lite_llm import LiteLlm
from google.adk.tools import FunctionTool
import requests
from typing import Dict, Any



def call_agent(inputs: Dict[str, Any]) -> str:
    try:
        response = requests.post("http://external.api/sentiment", json=inputs)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        return f"API call failed: {str(e)}"




external_sentiment_a_p_i = Agent(
    name='external_sentiment_a_p_i',
    model=LiteLlm(model='openai/gpt-4.1'),
    description='Calls external API for sentiment analysis',
    instruction='Process text and call external API for sentiment analysis'
,
    tools=[FunctionTool(call_agent)]

)
