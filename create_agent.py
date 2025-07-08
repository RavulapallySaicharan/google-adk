import os
import re
from typing import List, Optional
from pathlib import Path
import requests
from google.adk import Agent
from google.adk.models.lite_llm import LiteLlm
from google.adk.tools import FunctionTool
from google.adk.agents import LlmAgent, SequentialAgent, ParallelAgent, LoopAgent
from itertools import count

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
    pattern_clean = pattern.replace(' ', '')
    # Hybrid: more than one type of operator
    if any(op in pattern_clean for op in ['->', ',', '::']) and sum(op in pattern_clean for op in ['->', ',', '::']) > 1:
        pattern_type = 'complex'
        # For complex, just return the cleaned string for recursive parsing
        structure = pattern_clean
    elif '->' in pattern_clean:
        pattern_type = 'sequential'
        structure = pattern_clean.split('->')
    elif '::' in pattern_clean:
        pattern_type = 'loop'
        structure = pattern_clean.split('::')
    elif ',' in pattern_clean:
        pattern_type = 'parallel'
        structure = pattern_clean.split(',')
    else:
        pattern_type = 'sequential'
        structure = [pattern_clean]
    return pattern_type, structure

# Enhanced recursive parser to support :: (loop), -> (sequential), , (parallel)
def parse_pattern_recursive(pattern: str):
    pattern = pattern.replace(' ', '')
    def helper(s, idx=0):
        result = []
        token = ''
        while idx < len(s):
            if s[idx] == '(':  # group
                group, idx = helper(s, idx + 1)
                result.append(group)
            elif s[idx] == ')':
                if token:
                    result.append(token)
                    token = ''
                return result, idx + 1
            elif s[idx:idx+2] == '->':
                if token:
                    result.append(token)
                    token = ''
                result.append('->')
                idx += 2
            elif s[idx:idx+2] == '::':
                if token:
                    result.append(token)
                    token = ''
                result.append('::')
                idx += 2
            elif s[idx] == ',':
                if token:
                    result.append(token)
                    token = ''
                idx += 1
            else:
                token += s[idx]
                idx += 1
        if token:
            result.append(token)
        return result, idx
    parsed, _ = helper(pattern)
    return parsed

# Utility to flatten nested lists (for subagent names)
def flatten(l):
    for el in l:
        if isinstance(el, list):
            yield from flatten(el)
        else:
            yield el

# Modular workflow builder with LoopAgent support
def build_agent_from_pattern(parsed, agent_inputs, agent_description, agent_instruction, agent_tags, overwrite, temp_counter, parent_type=None):
    temp_agents = []
    if isinstance(parsed, str):
        return parsed, temp_agents
    # Check for loop, sequential, parallel
    if '::' in parsed:
        # Loop: split by '::', recursively build each part
        parts = []
        current = []
        for item in parsed:
            if item == '::':
                if current:
                    parts.append(current)
                    current = []
            else:
                current.append(item)
        if current:
            parts.append(current)
        subagent_names = []
        for part in parts:
            if len(part) == 1 and isinstance(part[0], str):
                sub_name, sub_temps = build_agent_from_pattern(part[0], agent_inputs, agent_description, agent_instruction, agent_tags, overwrite, temp_counter, parent_type='loop')
            else:
                sub_name, sub_temps = build_agent_from_pattern(part, agent_inputs, agent_description, agent_instruction, agent_tags, overwrite, temp_counter, parent_type='loop')
            subagent_names.append(sub_name)
            temp_agents.extend(sub_temps)
        temp_name = f"_tmp_loop{next(temp_counter)}"
        temp_agents.append((temp_name, 'loop', subagent_names))
        return temp_name, temp_agents
    elif '->' in parsed:
        # Sequential
        parts = []
        current = []
        for item in parsed:
            if item == '->':
                if current:
                    parts.append(current)
                    current = []
            else:
                current.append(item)
        if current:
            parts.append(current)
        subagent_names = []
        for part in parts:
            if len(part) == 1 and isinstance(part[0], str):
                sub_name, sub_temps = build_agent_from_pattern(part[0], agent_inputs, agent_description, agent_instruction, agent_tags, overwrite, temp_counter, parent_type='sequential')
            else:
                sub_name, sub_temps = build_agent_from_pattern(part, agent_inputs, agent_description, agent_instruction, agent_tags, overwrite, temp_counter, parent_type='sequential')
            subagent_names.append(sub_name)
            temp_agents.extend(sub_temps)
        temp_name = f"_tmp_seq{next(temp_counter)}"
        temp_agents.append((temp_name, 'sequential', subagent_names))
        return temp_name, temp_agents
    elif ',' in parsed:
        # Parallel
        subagent_names = []
        for item in parsed:
            if isinstance(item, str):
                sub_name, sub_temps = build_agent_from_pattern(item, agent_inputs, agent_description, agent_instruction, agent_tags, overwrite, temp_counter, parent_type='parallel')
            else:
                sub_name, sub_temps = build_agent_from_pattern(item, agent_inputs, agent_description, agent_instruction, agent_tags, overwrite, temp_counter, parent_type='parallel')
            subagent_names.append(sub_name)
            temp_agents.extend(sub_temps)
        temp_name = f"_tmp_par{next(temp_counter)}"
        temp_agents.append((temp_name, 'parallel', subagent_names))
        return temp_name, temp_agents
    else:
        # Single agent
        if isinstance(parsed, list) and len(parsed) == 1:
            return build_agent_from_pattern(parsed[0], agent_inputs, agent_description, agent_instruction, agent_tags, overwrite, temp_counter, parent_type)
        return parsed, temp_agents

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
    
    # Complex pattern support
    if pattern:
        pattern_type, _ = parse_pattern(pattern)
        if pattern_type == 'complex':
            # Recursively parse and build temp agents
            parsed = parse_pattern_recursive(pattern)
            temp_counter = count(1)
            top_agent, temp_agents = build_agent_from_pattern(parsed, agent_inputs, agent_description, agent_instruction, agent_tags, overwrite, temp_counter)
            # Instead of creating files for temp agents, collect their definitions to be written in the top-level agent.py
            temp_agent_defs = []
            for temp_name, temp_type, temp_subs in temp_agents:
                temp_desc = f"Temporary {temp_type} agent for complex pattern"
                temp_instr = f"Auto-generated {temp_type} agent for complex pattern"
                temp_tag = ["complex-temp"]
                # Flatten subagent names
                flat_subs = list(flatten(temp_subs))
                sub_factories = ', '.join([f"create_{to_snake_case(sub)}()" for sub in flat_subs])
                if temp_type == 'sequential':
                    agent_class = 'SequentialAgent'
                elif temp_type == 'parallel':
                    agent_class = 'ParallelAgent'
                elif temp_type == 'loop':
                    agent_class = 'LoopAgent'
                else:
                    agent_class = 'Agent'
                temp_factory = f"def create_{to_snake_case(temp_name)}():\n" \
                              f"    \"\"\"Factory for {temp_name}\"\"\"\n" \
                              f"    return {agent_class}(\n" \
                              f"        name='{to_snake_case(temp_name)}',\n" \
                              f"        model=LiteLlm(model='openai/gpt-4.1'),\n" \
                              f"        description='{temp_desc}',\n" \
                              f"        instruction='{temp_instr}',\n" \
                              f"        sub_agents=[{sub_factories}]\n" \
                              f"    )\n"
                temp_agent_defs.append(temp_factory)
            # Now, set up the final agent to use the top_agent and any trailing agents (if top_agent is not the final agent_name)
            if top_agent != agent_name:
                sub_agents = [top_agent]
            else:
                sub_agents = None
            # Create directory structure
            agent_dir = create_agent_directory(agent_name, overwrite)
            # Create __init__.py
            init_file = agent_dir / "__init__.py"
            init_file.touch()
            # Create agent.py with all temp agent factories and the top-level agent
            agent_file = agent_dir / "agent.py"
            lines = []
            lines.append("from google.adk import Agent")
            lines.append("from google.adk.models.lite_llm import LiteLlm")
            lines.append("from google.adk.agents import SequentialAgent, ParallelAgent, LoopAgent")
            lines.append("")
            # Write all temp agent factories
            for temp_factory in temp_agent_defs:
                lines.append(temp_factory)
                lines.append("")
            # Write the top-level agent factory
            factory_name = f"create_{to_snake_case(agent_name)}"
            lines.append(f"def {factory_name}():")
            lines.append(f"    \"\"\"Factory function to create a new instance of {to_snake_case(agent_name)} agent.\"\"\"")
            lines.append(f"    return SequentialAgent(")
            lines.append(f"        name='{to_snake_case(agent_name)}',")
            lines.append("        model=LiteLlm(model='openai/gpt-4.1'),")
            lines.append(f"        description='{agent_description}',")
            lines.append(f"        instruction='{agent_instruction}',")
            if sub_agents:
                sub_factories = ', '.join([f"create_{to_snake_case(sub)}()" for sub in sub_agents])
                lines.append(f"        sub_agents=[{sub_factories}]")
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
            return
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
        elif pattern_type == 'loop':
            lines.append("from google.adk.agents import LoopAgent")
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
    elif pattern_type == 'loop':
        agent_class = 'LoopAgent'
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
    from datetime import datetime
    
    start_time = datetime.now()
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
    end_time = datetime.now()
    print(f"Time taken for Example 1: {end_time - start_time}")
    start_time = datetime.now()
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
    end_time = datetime.now()
    print(f"Time taken for Example 2: {end_time - start_time}")
    start_time = datetime.now()
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
    end_time = datetime.now()
    print(f"Time taken for Example 3: {end_time - start_time}")
    start_time = datetime.now()
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
    end_time = datetime.now()
    print(f"Time taken for Example 4: {end_time - start_time}")
    start_time = datetime.now()
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
    end_time = datetime.now()
    print(f"Time taken for Example 5: {end_time - start_time}")
    start_time = datetime.now()
    # Example 6: Complex Pattern Agent
    # Pattern: ((a2->a3->a4),(a5->a6->a7))->a8
    # We'll use existing agents for demonstration, e.g., Sentiment Analyzer, External Sentiment API, and create temp names for others
    create_agent(
        agent_name="Complex Coordinator",
        agent_inputs=["text"],
        agent_description="Coordinates a complex workflow: two sequential groups in parallel, then a final agent.",
        agent_instruction="Run two sequential groups in parallel, then pass results to a final agent.",
        agent_tags=["coordination", "multi-agent", "workflow", "complex"],
        agent_flag=None,
        overwrite=True,
        pattern="((sentiment_analyzer->external_sentiment_api->task_coordinator_sequential),(task_coordinator_parallel->external_sentiment_api->sentiment_analyzer))->orchestrator"
    )
    end_time = datetime.now()
    print(f"Time taken for Example 6: {end_time - start_time}")

    # --- Example: LoopAgent Only ---
    start_time = datetime.now()
    create_agent(
        agent_name="Loop Only Agent",
        agent_inputs=["text"],
        agent_description="A loop agent with two subagents.",
        agent_instruction="Loop over AgentX and AgentY.",
        agent_tags=["loop", "test"],
        pattern="agent_x::agent_y"
    )
    end_time = datetime.now()
    print(f"Time taken for Example 7: {end_time - start_time}")
    start_time = datetime.now()

    # --- Example: Complex with LoopAgent ---
    create_agent(
        agent_name="Complex Loop Agent",
        agent_inputs=["text"],
        agent_description="Complex agent with sequential, loop, and parallel flows.",
        agent_instruction="Mix of sequential, loop, and parallel.",
        agent_tags=["complex", "loop", "test"],
        pattern="agent1->agent2::agent3,agent4"
    )
    end_time = datetime.now()
    print(f"Time taken for Example 8: {end_time - start_time}")
    start_time = datetime.now()

    # --- Example: Nested Loops ---
    create_agent(
        agent_name="Nested Loop Agent",
        agent_inputs=["text"],
        agent_description="Nested loop agent.",
        agent_instruction="Nested looping.",
        agent_tags=["loop", "nested", "test"],
        pattern="agenta::agentb::agentc"
    )
    end_time = datetime.now()
    print(f"Time taken for Example 9: {end_time - start_time}")
    start_time = datetime.now()

