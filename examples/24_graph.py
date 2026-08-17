"""
Graph: multi-agent workflows with conditional routing.

Similar to LangGraph but simpler.
"""

import asyncio

from pure_agents import END, Agent, Graph


async def main():
    # Create specialised agents
    researcher = Agent(
        system_prompt="You are a researcher. Find key facts about the topic.",
    )

    writer = Agent(
        system_prompt="You are a writer. Write a clear summary based on the research.",
    )

    reviewer = Agent(
        system_prompt=(
            "You are a reviewer. Check the content for quality. "
            "If it needs improvement, say 'NEEDS_REVISION: ' followed by feedback. "
            "If it's good, say 'APPROVED: ' followed by the final content."
        ),
    )

    # Build graph
    graph = Graph()

    # Add nodes
    graph.add_node("research", researcher)
    graph.add_node("write", writer)
    graph.add_node("review", reviewer)

    # Add edges
    graph.add_edge("research", "write")
    graph.add_edge("write", "review")

    # Conditional edge: revise or finish
    def check_review(state: dict) -> str:
        review = state.get("review", "")
        if "NEEDS_REVISION" in review:
            state["revision_count"] = state.get("revision_count", 0) + 1
            if state["revision_count"] >= 3:
                return END  # Stop after 3 revisions
            return "write"  # Go back to writer
        return END

    graph.add_conditional_edge("review", check_review)

    # Set entry
    graph.set_entry("research")

    # Run
    print("Running graph...\n")
    result = await graph.run("The impact of AI on software development")

    print(f"Final output:\n{result['output']}")
    print("\nNodes visited: research -> write -> review")
    if result.get("revision_count"):
        print(f"Revisions: {result['revision_count']}")


async def simple_example():
    """Simpler example with functions instead of agents."""
    print("\n=== Simple function-based graph ===\n")

    graph = Graph()

    # Nodes can be functions that modify state
    def step1(state: dict) -> dict:
        state["step1"] = f"Processed: {state['input']}"
        state["output"] = state["step1"]
        print(f"Step 1: {state['step1']}")
        return state

    def step2(state: dict) -> dict:
        state["step2"] = f"Enhanced: {state['output']}"
        state["output"] = state["step2"]
        print(f"Step 2: {state['step2']}")
        return state

    def step3(state: dict) -> dict:
        state["output"] = f"Final: {state['output']}"
        print(f"Step 3: {state['output']}")
        return state

    graph.add_node("step1", step1)
    graph.add_node("step2", step2)
    graph.add_node("step3", step3)

    graph.add_edge("step1", "step2")
    graph.add_edge("step2", "step3")

    graph.set_entry("step1")

    result = graph.run_sync("Hello World")
    print(f"\nFinal state: {result}")


if __name__ == "__main__":
    asyncio.run(simple_example())
    # asyncio.run(main())  # Uncomment to run with real agents
