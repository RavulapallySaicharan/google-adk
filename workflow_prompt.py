WORKFLOW_PROMPT = '''
You are an expert workflow and UI designer. Given the following goal and available agents, suggest a workflow to achieve the goal using one of the following workflow types: sequential, parallel, loop, or complex. Choose the most appropriate workflow type for the goal, and output the workflow as a React Flow JSON object.

Workflow Types:
- Sequential: Agents are executed one after another in a straight line.
- Parallel: The workflow branches so that multiple agents can operate in parallel, possibly merging later.
- Loop: There is a cyclical flow between agents (e.g., Agent1 ⇄ Agent2).
- Complex: The workflow combines sequential, parallel, and/or loop patterns (e.g., branching, merging, and cycles).

Instructions:
- Only include agents that are necessary for the goal.
- Select the most suitable workflow type (sequential, parallel, loop, or complex) and clearly reflect it in the structure.
- Output the workflow as a React Flow JSON object with two keys: "nodes" and "edges".
- Each node should have: id, type (e.g., "default", "input", "output"), data (with "label"), and position ({x, y}).
- Lable of each node should be the agent_flag.
- Each edge should have: id, source, target, and type (e.g., "step" or "default").
- Arrange nodes to visually represent the workflow type (e.g., vertical for sequential, branches for parallel, cycles for loops).
- For parallel or complex workflows, ensure branches and merges are clear in the node/edge structure.
- For loop workflows, include at least one cycle in the edges.
- For complex workflows, combine at least two patterns (e.g., a loop followed by a parallel branch).
- All workflows must start with a 'start' node (type: "input") and end with an 'end' node (type: "output").
- Return only the JSON object, nothing else.

Goal:
{goal}

Available Agents:
{agent_desc_text}

Example Outputs:

Sequential:
{
  "nodes": [
    {"id": "start", "type": "input", "data": {"label": "Start"}, "position": {"x": 0, "y": 0}},
    {"id": "agent1", "type": "default", "data": {"label": "Agent1"}, "position": {"x": 0, "y": 150}},
    {"id": "agent2", "type": "default", "data": {"label": "Agent2"}, "position": {"x": 0, "y": 300}},
    {"id": "agent3", "type": "default", "data": {"label": "Agent3"}, "position": {"x": 0, "y": 450}},
    {"id": "end", "type": "output", "data": {"label": "End"}, "position": {"x": 0, "y": 600}}
  ],
  "edges": [
    {"id": "e-start-agent1", "source": "start", "target": "agent1", "type": "step"},
    {"id": "e-agent1-agent2", "source": "agent1", "target": "agent2", "type": "step"},
    {"id": "e-agent2-agent3", "source": "agent2", "target": "agent3", "type": "step"},
    {"id": "e-agent3-end", "source": "agent3", "target": "end", "type": "step"}
  ]
}

Parallel:
{
  "nodes": [
    {"id": "start", "type": "input", "data": {"label": "Start"}, "position": {"x": 0, "y": 150}},
    {"id": "agent1", "type": "default", "data": {"label": "Agent1"}, "position": {"x": 200, "y": 50}},
    {"id": "agent2", "type": "default", "data": {"label": "Agent2"}, "position": {"x": 200, "y": 250}},
    {"id": "agent3", "type": "default", "data": {"label": "Agent3"}, "position": {"x": 400, "y": 150}},
    {"id": "end", "type": "output", "data": {"label": "End"}, "position": {"x": 600, "y": 150}}
  ],
  "edges": [
    {"id": "e-start-agent1", "source": "start", "target": "agent1", "type": "step"},
    {"id": "e-start-agent2", "source": "start", "target": "agent2", "type": "step"},
    {"id": "e-agent1-agent3", "source": "agent1", "target": "agent3", "type": "step"},
    {"id": "e-agent2-agent3", "source": "agent2", "target": "agent3", "type": "step"},
    {"id": "e-agent3-end", "source": "agent3", "target": "end", "type": "step"}
  ]
}

Loop:
{
  "nodes": [
    {"id": "start", "type": "input", "data": {"label": "Start"}, "position": {"x": 0, "y": 100}},
    {"id": "agent1", "type": "default", "data": {"label": "Agent1"}, "position": {"x": 200, "y": 100}},
    {"id": "agent2", "type": "default", "data": {"label": "Agent2"}, "position": {"x": 400, "y": 100}},
    {"id": "end", "type": "output", "data": {"label": "End"}, "position": {"x": 600, "y": 100}}
  ],
  "edges": [
    {"id": "e-start-agent1", "source": "start", "target": "agent1", "type": "step"},
    {"id": "e-agent1-agent2", "source": "agent1", "target": "agent2", "type": "step"},
    {"id": "e-agent2-agent1", "source": "agent2", "target": "agent1", "type": "step"},
    {"id": "e-agent2-end", "source": "agent2", "target": "end", "type": "step"}
  ]
}

Complex:
{
  "nodes": [
    {"id": "start", "type": "input", "data": {"label": "Start"}, "position": {"x": 0, "y": 200}},
    {"id": "agent1", "type": "default", "data": {"label": "Agent1"}, "position": {"x": 200, "y": 200}},
    {"id": "agent2", "type": "default", "data": {"label": "Agent2"}, "position": {"x": 400, "y": 100}},
    {"id": "agent3", "type": "default", "data": {"label": "Agent3"}, "position": {"x": 400, "y": 300}},
    {"id": "agent4", "type": "default", "data": {"label": "Agent4"}, "position": {"x": 600, "y": 50}},
    {"id": "agent5", "type": "default", "data": {"label": "Agent5"}, "position": {"x": 600, "y": 200}},
    {"id": "agent6", "type": "default", "data": {"label": "Agent6"}, "position": {"x": 600, "y": 350}},
    {"id": "end", "type": "output", "data": {"label": "End"}, "position": {"x": 800, "y": 200}}
  ],
  "edges": [
    {"id": "e-start-agent1", "source": "start", "target": "agent1", "type": "step"},
    {"id": "e-agent1-agent2", "source": "agent1", "target": "agent2", "type": "step"},
    {"id": "e-agent2-agent3", "source": "agent2", "target": "agent3", "type": "step"},
    {"id": "e-agent3-agent2", "source": "agent3", "target": "agent2", "type": "step"},
    {"id": "e-agent2-agent4", "source": "agent2", "target": "agent4", "type": "step"},
    {"id": "e-agent2-agent5", "source": "agent2", "target": "agent5", "type": "step"},
    {"id": "e-agent3-agent6", "source": "agent3", "target": "agent6", "type": "step"},
    {"id": "e-agent4-end", "source": "agent4", "target": "end", "type": "step"},
    {"id": "e-agent5-end", "source": "agent5", "target": "end", "type": "step"},
    {"id": "e-agent6-end", "source": "agent6", "target": "end", "type": "step"}
  ]
}
''' 


if __name__ == "__main__":
    # Example usage
    goal = "Process a customer support ticket using available agents."
    agent_desc_text = "Agent1: Intake agent, Agent2: Triage agent, Agent3: Resolution agent"
    prompt = WORKFLOW_PROMPT.format(goal=goal, agent_desc_text=agent_desc_text)
    print(prompt) 

