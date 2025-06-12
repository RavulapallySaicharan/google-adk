from google.adk import Agent
from google.adk.models.lite_llm import LiteLlm







sentiment_analyzer = Agent(
    name='sentiment_analyzer',
    model=LiteLlm(model='openai/gpt-4.1'),
    description='Analyzes text to determine sentiment and emotional tone',
    instruction='Provide accurate sentiment analysis and emotional insights from text'


)
