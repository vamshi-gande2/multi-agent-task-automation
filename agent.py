from typing import TypedDict, Literal
from langgraph.graph import StateGraph, START, END


# -----------------------------
# State
# -----------------------------
class AgentState(TypedDict, total=False):
    user_query: str
    route: str
    tool_result: str
    final_answer: str
    steps: list[str]


# -----------------------------
# Tools
# -----------------------------
def calculator(expression: str) -> str:
    """Safely evaluate simple arithmetic expressions."""
    allowed = set("0123456789+-*/(). ")
    if not all(char in allowed for char in expression):
        return "Invalid expression."

    try:
        result = eval(expression, {"__builtins__": {}}, {})
        return str(result)
    except Exception:
        return "Could not calculate the expression."


def research_tool(query: str) -> str:
    """Mock research tool for the MVP."""
    knowledge = {
        "rag": "RAG stands for Retrieval-Augmented Generation. It retrieves relevant information before generating an answer.",
        "langgraph": "LangGraph is a framework for building stateful workflows and agent systems using graphs.",
        "multi agent": "A multi-agent system uses multiple specialized agents that cooperate to solve a task."
    }

    query_lower = query.lower()

    for key, value in knowledge.items():
        if key in query_lower:
            return value

    return f"Research result for '{query}': No detailed local result found."


# -----------------------------
# Supervisor
# -----------------------------
def supervisor(state: AgentState) -> AgentState:
    query = state["user_query"].lower()

    if any(word in query for word in ["calculate", "compute", "+", "-", "*", "/"]):
        route = "calculator"
    else:
        route = "research"

    steps = state.get("steps", [])
    steps.append(f"Supervisor routed task to: {route}")

    return {
        **state,
        "route": route,
        "steps": steps,
    }


# -----------------------------
# Calculator Agent
# -----------------------------
def calculator_agent(state: AgentState) -> AgentState:
    query = state["user_query"]

    # Extract a simple arithmetic expression from the request.
    expression = query.lower()

    for prefix in [
        "calculate ",
        "compute ",
        "what is ",
    ]:
        expression = expression.replace(prefix, "")

    result = calculator(expression)

    steps = state.get("steps", [])
    steps.append(f"Calculator executed: {expression}")

    return {
        **state,
        "tool_result": result,
        "steps": steps,
    }


# -----------------------------
# Research Agent
# -----------------------------
def research_agent(state: AgentState) -> AgentState:
    query = state["user_query"]

    result = research_tool(query)

    steps = state.get("steps", [])
    steps.append("Research tool executed")

    return {
        **state,
        "tool_result": result,
        "steps": steps,
    }


# -----------------------------
# Finalizer
# -----------------------------
def finalizer(state: AgentState) -> AgentState:
    result = state.get("tool_result", "No result available.")

    answer = f"Final Answer: {result}"

    steps = state.get("steps", [])
    steps.append("Finalizer generated response")

    return {
        **state,
        "final_answer": answer,
        "steps": steps,
    }


# -----------------------------
# Conditional Routing
# -----------------------------
def route_to_specialist(
    state: AgentState,
) -> Literal["calculator_agent", "research_agent"]:
    if state["route"] == "calculator":
        return "calculator_agent"

    return "research_agent"


# -----------------------------
# Build Graph
# -----------------------------
workflow = StateGraph(AgentState)

workflow.add_node("supervisor", supervisor)
workflow.add_node("calculator_agent", calculator_agent)
workflow.add_node("research_agent", research_agent)
workflow.add_node("finalizer", finalizer)

workflow.add_edge(START, "supervisor")

workflow.add_conditional_edges(
    "supervisor",
    route_to_specialist,
)

workflow.add_edge("calculator_agent", "finalizer")
workflow.add_edge("research_agent", "finalizer")

workflow.add_edge("finalizer", END)

graph = workflow.compile()


# -----------------------------
# Run
# -----------------------------
def run_agent(query: str) -> AgentState:
    result = graph.invoke(
        {
            "user_query": query,
            "steps": [],
        }
    )

    return result


if __name__ == "__main__":
    print("\n=== Multi-Agent Task Automation ===\n")

    query = input("Enter your task: ")

    result = run_agent(query)

    print("\nExecution Steps:")
    for step in result["steps"]:
        print(f"- {step}")

    print("\n" + result["final_answer"])
