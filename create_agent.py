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

    # --- Generate FastAPI-based a2a_server.py ---
    a2a_lines = []
    a2a_lines.append("from fastapi import FastAPI, Request")
    a2a_lines.append("from fastapi.responses import JSONResponse")
    a2a_lines.append("import uvicorn")
    a2a_lines.append("import uuid")
    a2a_lines.append("from pathlib import Path")
    a2a_lines.append(f"from .agent import {factory_name}")
    a2a_lines.append("")
    a2a_lines.append("app = FastAPI(title=\"A2A Agent Server\")")
    a2a_lines.append("")
    # Agent card (minimal, can be extended)
    a2a_lines.append("AGENT_CARD = {")
    a2a_lines.append(f"    'name': '{to_snake_case(agent_name)}',")
    a2a_lines.append(f"    'description': '{agent_description}',")
    a2a_lines.append(f"    'url': 'http://localhost:8000',  # Update as needed")
    a2a_lines.append(f"    'version': '1.0.0',")
    a2a_lines.append("    'capabilities': { 'streaming': False, 'pushNotifications': False },")
    a2a_lines.append("    'defaultInputModes': ['text'],")
    a2a_lines.append("    'defaultOutputModes': ['text'],")
    a2a_lines.append("    'skills': [")
    a2a_lines.append("        {")
    a2a_lines.append(f"            'id': '{to_snake_case(agent_name)}_main',")
    a2a_lines.append(f"            'name': '{agent_name}',")
    a2a_lines.append(f"            'description': '{agent_description}',")
    a2a_lines.append(f"            'tags': {agent_tags},")
    a2a_lines.append(f"            'examples': ['Say hello']")
    a2a_lines.append("        }")
    a2a_lines.append("    ]")
    a2a_lines.append("}")
    a2a_lines.append("")
    a2a_lines.append("@app.get('/.well-known/agent.json')")
    a2a_lines.append("async def get_agent_card():")
    a2a_lines.append("    return JSONResponse(AGENT_CARD)")
    a2a_lines.append("")
    a2a_lines.append("@app.post('/tasks/send')")
    a2a_lines.append("async def handle_task(request: Request):")
    a2a_lines.append("    data = await request.json()")
    a2a_lines.append("    task_id = data.get('id', str(uuid.uuid4()))")
    a2a_lines.append("    user_message = ''")
    a2a_lines.append("    try:")
    a2a_lines.append("        user_message = data['message']['parts'][0].get('text', '')")
    a2a_lines.append("    except Exception:")
    a2a_lines.append("        pass")
    a2a_lines.append(f"    agent = {factory_name}()")
    a2a_lines.append("    # TODO: Replace with actual agent logic. For now, echo the user message.")
    a2a_lines.append("    agent_reply = f'Agent received: {user_message}'")
    a2a_lines.append("    response = {")
    a2a_lines.append("        'id': task_id,")
    a2a_lines.append("        'status': {'state': 'completed'},")
    a2a_lines.append("        'messages': [")
    a2a_lines.append("            data.get('message', {}),")
    a2a_lines.append("            {")
    a2a_lines.append("                'role': 'agent',")
    a2a_lines.append("                'parts': [{'text': agent_reply}]\n            }")
    a2a_lines.append("        ]")
    a2a_lines.append("    }")
    a2a_lines.append("    return JSONResponse(response)")
    a2a_lines.append("")
    a2a_lines.append("if __name__ == '__main__':")
    a2a_lines.append("    uvicorn.run(app, host='0.0.0.0', port=8000)")

    a2a_file = agent_dir / "a2a_server.py"
    with open(a2a_file, "w") as f:
        f.write("\n".join(a2a_lines))

if __name__ == "__main__":
    from datetime import datetime

    # --- Finance Domain Multi-Agent Workflow Examples for State Street ---

    # Example 1: Sequential Workflow - Trade Settlement Process
    # TradeCapture -> ComplianceCheck -> TradeSettlement -> Reporting
    start_time = datetime.now()
    create_agent(
        agent_name="Trade Capture Agent",
        agent_inputs=["trade_data"],
        agent_description="Captures and validates incoming trade data for further processing.",
        agent_instruction="Ingest trade data and perform initial validation.",
        agent_tags=["trade", "capture", "finance"],
        overwrite=True
    )
    create_agent(
        agent_name="Compliance Check Agent",
        agent_inputs=["trade_data"],
        agent_description="Performs compliance checks on trade data to ensure regulatory adherence.",
        agent_instruction="Check trade data for compliance with internal and external regulations.",
        agent_tags=["compliance", "check", "finance"],
        overwrite=True
    )
    create_agent(
        agent_name="Trade Settlement Agent",
        agent_inputs=["trade_data"],
        agent_description="Handles the settlement of trades, ensuring proper transfer of securities and cash.",
        agent_instruction="Settle trades by coordinating with counterparties and clearing systems.",
        agent_tags=["settlement", "trade", "finance"],
        overwrite=True
    )
    create_agent(
        agent_name="Reporting Agent",
        agent_inputs=["settlement_data"],
        agent_description="Generates regulatory and client reports post-settlement.",
        agent_instruction="Produce and distribute settlement and compliance reports.",
        agent_tags=["reporting", "regulatory", "finance"],
        overwrite=True
    )
    create_agent(
        agent_name="Trade Settlement Workflow",
        agent_inputs=["trade_data"],
        agent_description="Sequential workflow for trade capture, compliance, settlement, and reporting.",
        agent_instruction="Process trades through capture, compliance, settlement, and reporting steps.",
        agent_tags=["workflow", "sequential", "trade", "finance"],
        sub_agents=[
            "Trade Capture Agent",
            "Compliance Check Agent",
            "Trade Settlement Agent",
            "Reporting Agent"
        ],
        pattern="trade_capture_agent->compliance_check_agent->trade_settlement_agent->reporting_agent"
    )
    end_time = datetime.now()
    print(f"Time taken for Sequential Workflow: {end_time - start_time}")

    # Example 2: Parallel Workflow - Asset Servicing and NAV Calculation
    # AssetServicing and NAVCalculation run in parallel, then results are reconciled
    start_time = datetime.now()
    create_agent(
        agent_name="Asset Servicing Agent",
        agent_inputs=["asset_events"],
        agent_description="Processes corporate actions, dividends, and other asset servicing events.",
        agent_instruction="Handle all asset servicing events and update records accordingly.",
        agent_tags=["asset", "servicing", "finance"],
        overwrite=True
    )
    create_agent(
        agent_name="NAV Calculation Agent",
        agent_inputs=["fund_data"],
        agent_description="Calculates Net Asset Value (NAV) for funds based on latest market and asset data.",
        agent_instruction="Compute NAV using validated fund and market data.",
        agent_tags=["NAV", "calculation", "fund", "finance"],
        overwrite=True
    )
    create_agent(
        agent_name="Reconciliation Agent",
        agent_inputs=["servicing_data", "nav_data"],
        agent_description="Reconciles asset servicing and NAV calculation results for consistency.",
        agent_instruction="Compare and reconcile outputs from asset servicing and NAV calculation.",
        agent_tags=["reconciliation", "finance"],
        overwrite=True
    )
    create_agent(
        agent_name="Asset Servicing & NAV Workflow",
        agent_inputs=["asset_events", "fund_data"],
        agent_description="Parallel workflow for asset servicing and NAV calculation, followed by reconciliation.",
        agent_instruction="Process asset servicing and NAV calculation in parallel, then reconcile results.",
        agent_tags=["workflow", "parallel", "asset", "NAV", "finance"],
        sub_agents=[
            "Asset Servicing Agent",
            "NAV Calculation Agent",
            "Reconciliation Agent"
        ],
        pattern="asset_servicing_agent,nav_calculation_agent->reconciliation_agent"
    )
    end_time = datetime.now()
    print(f"Time taken for Parallel Workflow: {end_time - start_time}")

    # Example 3: Loop Workflow - Daily Compliance Checks
    # ComplianceCheck <-> Reporting (loop until all issues resolved)
    start_time = datetime.now()
    create_agent(
        agent_name="Daily Compliance Loop Workflow",
        agent_inputs=["trade_data"],
        agent_description="Loop workflow for daily compliance checks and reporting until all issues are resolved.",
        agent_instruction="Iterate between compliance checks and reporting until no compliance issues remain.",
        agent_tags=["workflow", "loop", "compliance", "reporting", "finance"],
        sub_agents=[
            "Compliance Check Agent",
            "Reporting Agent"
        ],
        pattern="compliance_check_agent::reporting_agent"
    )
    end_time = datetime.now()
    print(f"Time taken for Loop Workflow: {end_time - start_time}")

    # Example 4: Complex Workflow - Fund Accounting End-to-End
    # TradeCapture -> (AssetServicing, NAVCalculation) -> Reconciliation -> FundAccounting -> Reporting
    start_time = datetime.now()
    create_agent(
        agent_name="Fund Accounting Agent",
        agent_inputs=["reconciled_data"],
        agent_description="Performs fund accounting based on reconciled asset and NAV data.",
        agent_instruction="Execute fund accounting tasks and prepare final figures for reporting.",
        agent_tags=["fund", "accounting", "finance"],
        overwrite=True
    )
    create_agent(
        agent_name="Fund Accounting Workflow",
        agent_inputs=["trade_data", "asset_events", "fund_data"],
        agent_description="Complex workflow for end-to-end fund accounting, combining sequential, parallel, and reconciliation steps.",
        agent_instruction="Capture trades, process asset servicing and NAV in parallel, reconcile, perform fund accounting, and report.",
        agent_tags=["workflow", "complex", "fund", "accounting", "finance"],
        sub_agents=[
            "Trade Capture Agent",
            "Asset Servicing Agent",
            "NAV Calculation Agent",
            "Reconciliation Agent",
            "Fund Accounting Agent",
            "Reporting Agent"
        ],
        pattern="trade_capture_agent->(asset_servicing_agent,nav_calculation_agent)->reconciliation_agent->fund_accounting_agent->reporting_agent"
    )
    end_time = datetime.now()
    print(f"Time taken for Complex Workflow: {end_time - start_time}")

    # Example 5: External Pricing API Agent (simulates external data fetch for NAV)
    start_time = datetime.now()
    create_agent(
        agent_name="External Pricing API Agent",
        agent_inputs=["security_ids"],
        agent_description="Fetches real-time security prices from an external pricing API for NAV calculation.",
        agent_instruction="Call the external pricing API and return the latest prices for the given securities.",
        agent_tags=["external", "pricing", "API", "NAV", "finance"],
        agent_url="http://external.pricing.api/prices",  # Simulated endpoint
        overwrite=True
    )
    end_time = datetime.now()
    print(f"Time taken for External Pricing API Agent: {end_time - start_time}")

    # Example 6: Orchestrator Agent - Dynamic Task Routing
    # The Orchestrator can choose which subagent(s) to invoke based on input or workflow state
    start_time = datetime.now()
    create_agent(
        agent_name="Finance Orchestrator Agent",
        agent_inputs=["task_type", "input_data"],
        agent_description="Orchestrates finance operations by dynamically routing tasks to the appropriate agent (e.g., compliance, settlement, NAV, reporting, pricing).",
        agent_instruction="Based on the task_type, invoke the relevant subagent(s) to process input_data. For NAV calculation, fetch prices using the External Pricing API Agent.",
        agent_tags=["orchestrator", "dynamic", "routing", "finance"],
        sub_agents=[
            "Trade Capture Agent",
            "Compliance Check Agent",
            "Trade Settlement Agent",
            "Reporting Agent",
            "Asset Servicing Agent",
            "NAV Calculation Agent",
            "Reconciliation Agent",
            "Fund Accounting Agent",
            "External Pricing API Agent"
        ],
        is_orchestrator=True,
        overwrite=True
    )
    end_time = datetime.now()
    print(f"Time taken for Orchestrator Agent: {end_time - start_time}")