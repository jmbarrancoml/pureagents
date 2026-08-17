---
sidebar_position: 14
---

# Chaining

Run agents in sequence. Output of each agent feeds into the next.

## Basic usage

```python
from pure_agents import Agent, chain

researcher = Agent(template="researcher")
summariser = Agent(system="Summarise the following text concisely.")
formatter = Agent(system="Format as bullet points.")

result = await chain(
    [researcher, summariser, formatter], "The history of machine learning"
)
```

## How it works

```
prompt → Agent 1 → output → Agent 2 → output → Agent 3 → final result
```

Each agent receives the previous agent's output as its prompt.

## Use cases

**Research pipeline:**
```python
gatherer = Agent(system="Find information about the topic.")
analyser = Agent(system="Analyse the information critically.")
writer = Agent(system="Write a report based on the analysis.")

report = await chain([gatherer, analyser, writer], "Climate change in 2024")
```

**Content pipeline:**
```python
drafter = Agent(system="Write a first draft.")
editor = Agent(system="Improve clarity and fix errors.")
formatter = Agent(system="Format for publication.")

article = await chain([drafter, editor, formatter], "AI in healthcare")
```

**Code pipeline:**
```python
planner = Agent(system="Plan the implementation approach.")
coder = Agent(template="coder")
reviewer = Agent(system="Review the code for bugs and improvements.")

code = await chain([planner, coder, reviewer], "REST API for user management")
```

## Sync version

```python
from pure_agents import chain_sync

result = chain_sync([agent1, agent2], "Your prompt")
```

## Tips

- Keep each agent focused on one task
- Order matters: think about what each agent needs as input
- Use `debug=True` on agents to see intermediate outputs
