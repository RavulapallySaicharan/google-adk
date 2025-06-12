from google.adk import Agent
from google.adk.models.lite_llm import LiteLlm

from google.adk.agents import LlmAgent




greeter = LlmAgent(
    name='greeter',
    model=LiteLlm(model='openai/gpt-4.1')
)
task_executor = LlmAgent(
    name='task_executor',
    model=LiteLlm(model='openai/gpt-4.1')
)
result_validator = LlmAgent(
    name='result_validator',
    model=LiteLlm(model='openai/gpt-4.1')
)



task_coordinator = Agent(
    name='task_coordinator',
    model=LiteLlm(model='openai/gpt-4.1'),
    description='Coordinates multiple agents to complete complex tasks',
    instruction='Manage and coordinate sub-agents to complete tasks efficiently'

,
    sub_agents=[greeter, task_executor, result_validator]
)
