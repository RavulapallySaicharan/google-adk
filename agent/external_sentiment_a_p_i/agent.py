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




external sentiment api = name='external_sentiment_a_p_i' description='Calls external API for sentiment analysis' parent_agent=None sub_agents=[] before_agent_callback=None after_agent_callback=None model=LiteLlm(model='openai/gpt-4.1', llm_client=<google.adk.models.lite_llm.LiteLLMClient object at 0x000001693758C190>) instruction='Process text and call external API for sentiment analysis' global_instruction='' tools=[<google.adk.tools.function_tool.FunctionTool object at 0x00000169375B41A0>] generate_content_config=None disallow_transfer_to_parent=False disallow_transfer_to_peers=False include_contents='default' input_schema=None output_schema=None output_key=None planner=None code_executor=None examples=None before_model_callback=None after_model_callback=None before_tool_callback=None after_tool_callback=None
