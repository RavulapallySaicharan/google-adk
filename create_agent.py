import os
import re
from typing import List, Optional
from pathlib import Path
import requests
from google.adk import Agent
from google.adk.models.lite_llm import LiteLlm
from google.adk.tools import FunctionTool
from google.adk.agents import LlmAgent, SequentialAgent, ParallelAgent

def to_snake_case(name: str) -> str:
    """Convert a string to snake_case and ensure it's a valid identifier."""
    # Replace spaces and hyphens with underscores
    name = re.sub(r'[\s\-]+', '_', name)
    # Add underscore before a capital letter that follows a lowercase letter or number
    name = re.sub(r'(?<=[a-z0-9])([A-Z])', r'_\1', name)
    # Lowercase everything
    name = name.lower()
    # Remove any non-alphanumeric characters except underscores
    name = re.sub(r'[^a-z0-9_]', '', name)
    # Remove consecutive underscores
    name = re.sub(r'_+', '_', name)
    # Remove leading/trailing underscores
    name = name.strip('_')
    return name

def validate_inputs(agent_name: str, agent_inputs: List[str], agent_description: str,
                   agent_instruction: str, agent_tags: List[str],
                   agent_url: Optional[str], sub_agents: Optional[List[str]],
                   agent_flag: Optional[str] = None, pattern: Optional[str] = None,
                   tools: Optional[List[str]] = None, is_orchestrator: bool = False) -> None:
    """
    Validate all input parameters.
    Args:
        agent_name: Name of the agent (non-empty string)
        agent_inputs: List of input parameters (non-empty list)
        agent_description: Description of the agent's functionality (non-empty string)
        agent_instruction: Instructions for the agent (non-empty string)
        agent_tags: List of tags for categorization (non-empty list)
        agent_url: URL for external agent (if applicable, string or None)
        sub_agents: List of sub-agent names (if applicable, list or None)
        agent_flag: Flag for the agent (string or None)
        pattern: Pattern string for multi-agent coordination (string or None)
        tools: List of tools for the agent (list or None)
        is_orchestrator: Whether the agent is an orchestrator (bool)
    """
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
    # Validate agent type specific inputs
    if agent_url is not None and sub_agents is not None:
        raise ValueError("Cannot specify both agent_url and sub_agents")
    if agent_url is not None and not isinstance(agent_url, str):
        raise ValueError("agent_url must be a string")
    if sub_agents is not None and not isinstance(sub_agents, list):
        raise ValueError("sub_agents must be a list")
    if agent_flag is not None and not isinstance(agent_flag, str):
        raise ValueError("agent_flag must be a string if provided")
    if pattern is not None and not isinstance(pattern, str):
        raise ValueError("pattern must be a string if provided")
    if tools is not None and not isinstance(tools, list):
        raise ValueError("tools must be a list if provided")
    if not isinstance(is_orchestrator, bool):
        raise ValueError("is_orchestrator must be a boolean value")

def create_agent_directory(agent_name: str, overwrite: bool) -> Path:
    """Create the agent directory structure."""
    snake_name = to_snake_case(agent_name)
    agent_dir = Path("agents") / snake_name
    
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

def parse_pattern(pattern: str):
    """Parse the pattern string and return the pattern type and agent structure."""
    if not pattern or not isinstance(pattern, str):
        raise ValueError("pattern must be a non-empty string")
    
    # Remove whitespace for easier parsing
    pattern_clean = pattern.replace(' ', '')
    if '->' in pattern_clean and ',' in pattern_clean:
        # Hybrid pattern
        pattern_type = 'hybrid'
        # Split by '->' first, then by ',' within each segment
        segments = pattern_clean.split('->')
        structure = [seg.split(',') for seg in segments]
    elif '->' in pattern_clean:
        pattern_type = 'sequential'
        structure = pattern_clean.split('->')
    elif ',' in pattern_clean:
        pattern_type = 'parallel'
        structure = pattern_clean.split(',')
    else:
        # Single agent (treat as sequential with one agent)
        pattern_type = 'sequential'
        structure = [pattern_clean]
    return pattern_type, structure

def create_agent(
    agent_name: str,
    agent_inputs: List[str],
    agent_description: str,
    agent_instruction: str,
    agent_tags: List[str],
    agent_flag: Optional[str] = None, # if None then it should be snake_case of agent_name
    overwrite: bool = True,
    agent_url: Optional[str] = None,
    sub_agents: Optional[List[str]] = None,
    pattern: Optional[str] = None,
    tools: Optional[List[str]] = None,
    is_orchestrator: bool = False
) -> None:
    """
    Create a Google ADK agent based on the provided parameters.
    
    Args:
        agent_name: Name of the agent
        agent_inputs: List of input parameters
        agent_description: Description of the agent's functionality
        agent_instruction: Instructions for the agent
        agent_tags: List of tags for categorization
        agent_flag: Flag for the agent
        overwrite: Whether to overwrite existing agent directory
        agent_url: URL for external agent (if applicable)
        sub_agents: List of sub-agent names (if applicable)
        pattern: Pattern string for multi-agent coordination (if applicable)
        tools: List of tools for the agent (if applicable)
        is_orchestrator: Whether the agent is an orchestrator
    """
    if agent_flag is None:
        agent_flag = to_snake_case(agent_name)
    # Validate inputs
    validate_inputs(agent_name, agent_inputs, agent_description, agent_instruction,
                   agent_tags, agent_url, sub_agents, agent_flag, pattern, tools, is_orchestrator)
    
    # Parse pattern if provided
    pattern_type = None
    pattern_structure = None
    if pattern:
        pattern_type, pattern_structure = parse_pattern(pattern)
        print(f"Pattern type: {pattern_type}, structure: {pattern_structure}")
        # If sub_agents is not provided, infer from pattern
        if sub_agents is None:
            # Flatten structure to get all agent names
            if pattern_type == 'hybrid':
                sub_agents = [agent for group in pattern_structure for agent in group]
            elif pattern_type == 'sequential':
                sub_agents = pattern_structure if isinstance(pattern_structure, list) else [pattern_structure]
            elif pattern_type == 'parallel':
                sub_agents = pattern_structure if isinstance(pattern_structure, list) else [pattern_structure]

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
    lines = []
    lines.append("from google.adk import Agent")
    lines.append("from google.adk.models.lite_llm import LiteLlm")
    if agent_url is not None:
        lines.append("from google.adk.tools import FunctionTool")
        lines.append("import requests")
        lines.append("from typing import Dict, Any")
    pattern_type = None
    if pattern:
        pattern_type, _ = parse_pattern(pattern)
    if sub_agents is not None:
        # Import each subagent's factory function from its module
        for name in sub_agents:
            module_name = to_snake_case(name)
            factory_name = f"create_{module_name}"
            lines.append(f"from agents.{module_name}.agent import {factory_name}")
        # Import the appropriate agent class for the pattern
        if pattern_type == 'sequential':
            lines.append("from google.adk.agents import SequentialAgent")
        elif pattern_type == 'parallel':
            lines.append("from google.adk.agents import ParallelAgent")
        else:
            lines.append("from google.adk.agents import LlmAgent")
    lines.append("")
    if agent_url is not None:
        lines.append("def call_agent(inputs: Dict[str, Any]) -> str:")
        lines.append("    try:")
        lines.append(f"        response = requests.post(\"{agent_url}\", json=inputs)")
        lines.append("        response.raise_for_status()")
        lines.append("        return response.json()")
        lines.append("    except requests.exceptions.RequestException as e:")
        lines.append("        return f'API call failed: {{str(e)}}'")
        lines.append("")
    # Factory function for this agent
    factory_name = f"create_{to_snake_case(agent_name)}"
    lines.append(f"def {factory_name}():")
    lines.append(f"    \"\"\"Factory function to create a new instance of {to_snake_case(agent_name)} agent.\"\"\"")
    # Choose the agent class based on the pattern type
    if pattern_type == 'sequential':
        agent_class = 'SequentialAgent'
    elif pattern_type == 'parallel':
        agent_class = 'ParallelAgent'
    else:
        agent_class = 'Agent'
    lines.append(f"    return {agent_class}(")
    lines.append(f"        name='{to_snake_case(agent_name)}',")
    lines.append("        model=LiteLlm(model='openai/gpt-4.1'),")
    lines.append(f"        description='{agent_description}',")
    lines.append(f"        instruction='{agent_instruction}'")
    if agent_url is not None:
        lines.append("        ,tools=[FunctionTool(call_agent)]")
    if sub_agents is not None:
        if is_orchestrator:
            sub_agents_str = ", ".join(f"AgentTool(agent={f'create_{to_snake_case(name)}()'})" for name in sub_agents)
            lines.append(f"        ,tools=[{sub_agents_str}]")
        else:
            sub_agents_str = ", ".join(f"{f'create_{to_snake_case(name)}()'}" for name in sub_agents)
            lines.append(f"        ,sub_agents=[{sub_agents_str}]")
    lines.append("    )")
    lines.append("")
    # Only add __main__ runner for leaf agents (not orchestrators)
    lines.append("if __name__ == '__main__':")
    lines.append("    import os")
    lines.append("    import asyncio")
    lines.append("    import uuid")
    lines.append("    from dotenv import load_dotenv")
    lines.append("    from google.adk.runners import Runner")
    lines.append("    from google.adk.sessions import DatabaseSessionService")
    lines.append("    from google.adk.artifacts import InMemoryArtifactService")
    lines.append("    from google.adk.memory import InMemoryMemoryService")
    lines.append("    from google.genai import types")
    lines.append("    load_dotenv()")
    lines.append("    async def main():")
    lines.append("        session_service = DatabaseSessionService(db_url=os.getenv('DATABASE_URL', 'sqlite:///agent_sessions.db'))")
    lines.append("        artifact_service = InMemoryArtifactService()")
    lines.append("        memory_service = InMemoryMemoryService()")
    lines.append("        initial_state = {}  # You can customize initial state if needed")
    lines.append("        app_name = 'SampleApp'")
    lines.append("        user_id = 'example_user'")
    lines.append("        session_id = str(uuid.uuid4())")
    lines.append("        await session_service.create_session(app_name=app_name, user_id=user_id, session_id=session_id, state=initial_state)")
    lines.append(f"        runner = Runner(app_name=app_name, agent={factory_name}(), session_service=session_service, artifact_service=artifact_service, memory_service=memory_service)")
    lines.append("        user_message = types.Content(role='user', parts=[types.Part(text='Hello, world!')])")
    lines.append("        async for event in runner.run_async(user_id=user_id, session_id=session_id, new_message=user_message):")
    lines.append("            if event.content and event.content.role == 'agent':")
    lines.append("                print(f'Agent: {event.content.parts[0].text}')")
    lines.append("        await runner.close()")
    lines.append("    asyncio.run(main())")

    with open(agent_file, "w") as f:
        f.write("\n".join(lines))

if __name__ == "__main__":
    # Example 1: LLM Agent
    create_agent(
        agent_name="Sentiment Analyzer",
        agent_inputs=["text"],
        agent_description="Analyzes text to determine sentiment and emotional tone",
        agent_instruction="Provide accurate sentiment analysis and emotional insights from text",
        agent_tags=["nlp", "sentiment-analysis", "emotion-detection"],
        agent_flag=None,
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
        agent_flag=None,
        overwrite=True,
        agent_url="http://external.api/sentiment",
        sub_agents=None
    )

    # Example 3: Multi-Agent Sequential (uses Example 1 and 2 as subagents)
    create_agent(
        agent_name="Task Coordinator Sequential",
        agent_inputs=["text"],
        agent_description="Coordinates Sentiment Analyzer and External Sentiment API sequentially",
        agent_instruction="First analyze sentiment, then call external API for further analysis.",
        agent_tags=["coordination", "multi-agent", "workflow", "sequential"],
        agent_flag=None,
        overwrite=True,
        sub_agents=["Sentiment Analyzer", "External Sentiment API"],
        pattern="sentiment_analyzer->external_sentiment_api"
    )

    # Example 4: Multi-Agent Parallel (uses Example 1 and 2 as subagents)
    create_agent(
        agent_name="Task Coordinator Parallel",
        agent_inputs=["text"],
        agent_description="Coordinates Sentiment Analyzer and External Sentiment API in parallel",
        agent_instruction="Analyze sentiment using both internal and external APIs in parallel.",
        agent_tags=["coordination", "multi-agent", "workflow", "parallel"],
        agent_flag=None,
        overwrite=True,
        sub_agents=["Sentiment Analyzer", "External Sentiment API"],
        pattern="sentiment_analyzer, external_sentiment_api"
    )

    # Example 5: Orchestrator Agent
    create_agent(
        agent_name="Orchestrator",
        agent_inputs=["text"],
        agent_description="Orchestrates the execution of multiple agents",
        agent_instruction="Manage and coordinate sub-agents to complete tasks efficiently",
        agent_tags=["coordination", "multi-agent", "workflow", "orchestrator"],
        agent_flag=None,
        overwrite=True,
        sub_agents=["Sentiment Analyzer", "External Sentiment API"],
        is_orchestrator=True
    )