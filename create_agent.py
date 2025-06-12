import os
import re
from typing import List, Optional
from pathlib import Path
import requests
from google.adk import Agent
from google.adk.models.lite_llm import LiteLlm
from google.adk.tools import FunctionTool
from google.adk.agents import LlmAgent

def to_snake_case(name: str) -> str:
    """Convert a string to snake_case and ensure it's a valid identifier."""
    # First convert to snake case
    snake = re.sub(r'(?<!^)(?=[A-Z])', '_', name).lower()
    # Remove any non-alphanumeric characters except underscores
    snake = re.sub(r'[^a-z0-9_]', '_', snake)
    # Ensure it starts with a letter or underscore
    if not snake[0].isalpha() and snake[0] != '_':
        snake = 'a_' + snake
    # Remove consecutive underscores
    snake = re.sub(r'_+', '_', snake)
    # Remove leading/trailing underscores
    snake = snake.strip('_')
    return snake

def validate_inputs(agent_name: str, agent_inputs: List[str], agent_description: str,
                   agent_instruction: str, agent_tags: List[str], agent_port: int,
                   agent_url: Optional[str], sub_agents: Optional[List[str]]) -> None:
    """Validate all input parameters."""
    if not agent_name or not isinstance(agent_name, str):
        raise ValueError("agent_name must be a non-empty string")
    
    if not agent_inputs or not isinstance(agent_inputs, list):
        raise ValueError("agent_inputs must be a non-empty list")
    
    if not agent_description or not isinstance(agent_description, str):
        raise ValueError("agent_description must be a non-empty string")
    
    if not agent_instruction or not isinstance(agent_instruction, str):
        raise ValueError("agent_instruction must be a non-empty string")
    
    if not agent_tags or not isinstance(agent_tags, list):
        raise ValueError("agent_tags must be a non-empty list")
    
    if not isinstance(agent_port, int) or agent_port < 1024 or agent_port > 65535:
        raise ValueError("agent_port must be a valid port number between 1024 and 65535")
    
    # Validate agent type specific inputs
    if agent_url is not None and sub_agents is not None:
        raise ValueError("Cannot specify both agent_url and sub_agents")
    
    if agent_url is not None and not isinstance(agent_url, str):
        raise ValueError("agent_url must be a string")
    
    if sub_agents is not None and not isinstance(sub_agents, list):
        raise ValueError("sub_agents must be a list")

def create_agent_directory(agent_name: str, overwrite: bool) -> Path:
    """Create the agent directory structure."""
    snake_name = to_snake_case(agent_name)
    agent_dir = Path("agent") / snake_name
    
    if agent_dir.exists() and not overwrite:
        raise FileExistsError(f"Agent directory {agent_dir} already exists and overwrite=False")
    
    agent_dir.mkdir(parents=True, exist_ok=True)
    return agent_dir

def create_llm_agent(agent_name: str, agent_description: str, agent_instruction: str) -> Agent:
    """Create an LLM agent."""
    return Agent(
        name=to_snake_case(agent_name),
        model=LiteLlm(model="openai/gpt-4.1"),
        description=agent_description,
        instruction=agent_instruction
    )

def create_external_agent(agent_name: str, agent_description: str, agent_instruction: str,
                         agent_url: str) -> Agent:
    """Create an external agent."""
    def call_agent(inputs: dict) -> str:
        try:
            response = requests.post(agent_url, json=inputs)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            return f"API call failed: {str(e)}"
    
    return Agent(
        name=to_snake_case(agent_name),
        model=LiteLlm(model="openai/gpt-4.1"),
        description=agent_description,
        instruction=agent_instruction,
        tools=[FunctionTool(call_agent)]
    )

def create_multi_agent(agent_name: str, agent_description: str, agent_instruction: str,
                      sub_agents: List[str]) -> LlmAgent:
    """Create a multi-agent coordinator."""
    sub_agent_instances = []
    for sub_agent_name in sub_agents:
        sub_agent = LlmAgent(
            name=to_snake_case(sub_agent_name),
            model=LiteLlm(model="openai/gpt-4.1")
        )
        sub_agent_instances.append(sub_agent)
    
    return LlmAgent(
        name=to_snake_case(agent_name),
        model=LiteLlm(model="openai/gpt-4.1"),
        description=agent_description,
        sub_agents=sub_agent_instances
    )

def create_agent(
    agent_name: str,
    agent_inputs: List[str],
    agent_description: str,
    agent_instruction: str,
    agent_tags: List[str],
    agent_port: int,
    overwrite: bool = True,
    agent_url: Optional[str] = None,
    sub_agents: Optional[List[str]] = None
) -> None:
    """
    Create a Google ADK agent based on the provided parameters.
    
    Args:
        agent_name: Name of the agent
        agent_inputs: List of input parameters
        agent_description: Description of the agent's functionality
        agent_instruction: Instructions for the agent
        agent_tags: List of tags for categorization
        agent_port: Port number for the agent
        overwrite: Whether to overwrite existing agent directory
        agent_url: URL for external agent (if applicable)
        sub_agents: List of sub-agent names (if applicable)
    """
    # Validate inputs
    validate_inputs(agent_name, agent_inputs, agent_description, agent_instruction,
                   agent_tags, agent_port, agent_url, sub_agents)
    
    # Create directory structure
    agent_dir = create_agent_directory(agent_name, overwrite)
    
    # Create __init__.py
    init_file = agent_dir / "__init__.py"
    init_file.touch()
    
    # Create agent.py with appropriate agent type
    agent_file = agent_dir / "agent.py"
    
    if agent_url is not None:
        agent = create_external_agent(agent_name, agent_description, agent_instruction, agent_url)
    elif sub_agents is not None:
        agent = create_multi_agent(agent_name, agent_description, agent_instruction, sub_agents)
    else:
        agent = create_llm_agent(agent_name, agent_description, agent_instruction)
    
    # Write agent code to file
    with open(agent_file, "w") as f:
        f.write(f"""from google.adk import Agent
from google.adk.models.lite_llm import LiteLlm
{'' if agent_url is None else 'from google.adk.tools import FunctionTool\nimport requests\nfrom typing import Dict, Any'}
{'' if sub_agents is None else 'from google.adk.agents import LlmAgent'}

{'' if agent_url is None else f'''
def call_agent(inputs: Dict[str, Any]) -> str:
    try:
        response = requests.post("{agent_url}", json=inputs)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        return f"API call failed: {{str(e)}}"
'''}

{'' if sub_agents is None else f'''
{"".join(f"{to_snake_case(name)} = LlmAgent(\n    name='{to_snake_case(name)}',\n    model=LiteLlm(model='openai/gpt-4.1')\n)\n" for name in sub_agents)}
'''}

{to_snake_case(agent_name)} = {'Agent' if agent_url is None else 'Agent'}(
    name='{to_snake_case(agent_name)}',
    model=LiteLlm(model='openai/gpt-4.1'),
    description='{agent_description}',
    instruction='{agent_instruction}'
{'' if agent_url is None else f',\n    tools=[FunctionTool(call_agent)]'}
{'' if sub_agents is None else f',\n    sub_agents=[{", ".join(to_snake_case(name) for name in sub_agents)}]'}
)
""")

if __name__ == "__main__":
    # Example 1: LLM Agent
    create_agent(
        agent_name="Sentiment Analyzer",
        agent_inputs=["text"],
        agent_description="Analyzes text to determine sentiment and emotional tone",
        agent_instruction="Provide accurate sentiment analysis and emotional insights from text",
        agent_tags=["nlp", "sentiment-analysis", "emotion-detection"],
        agent_port=5013,
        overwrite=True,
        agent_url=None,
        sub_agents=None
    )

    # Example 2: External Agent
    create_agent(
        agent_name="External Sentiment API",
        agent_inputs=["text"],
        agent_description="Calls external API for sentiment analysis",
        agent_instruction="Process text and call external API for sentiment analysis",
        agent_tags=["api", "sentiment", "external"],
        agent_port=5014,
        overwrite=True,
        agent_url="http://external.api/sentiment",
        sub_agents=None
    )

    # Example 3: Multi-Agent
    create_agent(
        agent_name="Task Coordinator",
        agent_inputs=["task", "context"],
        agent_description="Coordinates multiple agents to complete complex tasks",
        agent_instruction="Manage and coordinate sub-agents to complete tasks efficiently",
        agent_tags=["coordination", "multi-agent", "workflow"],
        agent_port=5015,
        overwrite=True,
        agent_url=None,
        sub_agents=["Greeter", "TaskExecutor", "ResultValidator"]
    ) 