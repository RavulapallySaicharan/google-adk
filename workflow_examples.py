# ReactFlow-compatible workflow examples with explicit start and end nodes
# Each workflow is a Python dict with 'nodes' and 'edges' keys

sequential_workflow = {
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

parallel_workflow = {
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

loop_workflow = {
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

complex_workflow = {
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

workflows_by_type = {
    "sequential": sequential_workflow,
    "parallel": parallel_workflow,
    "loop": loop_workflow,
    "complex": complex_workflow
} 