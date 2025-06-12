from google.adk import Agent
from google.adk.models.lite_llm import LiteLlm







sentiment analyzer = name='sentiment_analyzer' description='Analyzes text to determine sentiment and emotional tone' parent_agent=None sub_agents=[] before_agent_callback=None after_agent_callback=None model=LiteLlm(model='openai/gpt-4.1', llm_client=<google.adk.models.lite_llm.LiteLLMClient object at 0x00000169375B4050>) instruction='Provide accurate sentiment analysis and emotional insights from text' global_instruction='' tools=[] generate_content_config=None disallow_transfer_to_parent=False disallow_transfer_to_peers=False include_contents='default' input_schema=None output_schema=None output_key=None planner=None code_executor=None examples=None before_model_callback=None after_model_callback=None before_tool_callback=None after_tool_callback=None
