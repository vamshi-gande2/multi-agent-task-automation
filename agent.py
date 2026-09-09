import ast
import operator as op
import sys
from typing import TypedDict, Literal

from langgraph.graph import StateGraph, START, END


# ============================================================
# STATE
# ============================================================

class AgentState(TypedDict, total=False):
    user_query: str
    route: str
    tool_result: str
    final_answer: str
    steps: list[str]


# ============================================================
# SAFE CALCULATOR TOOL
# ============================================================

_ALLOWED_OPERATORS = {
    ast.Add: op.add,
    ast.Sub: op.sub,
    ast.Mult: op.mul,
    ast.Div: op.truediv,
    ast.Mod: op.mod,
    ast.Pow: op.pow,
    ast.USub: op.neg,
    ast.UAdd: op.pos,
}


def safe_calculate(expression: str):
    """Safely evaluate basic arithmetic expressions."""
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError:
        raise ValueError("Invalid arithmetic expression.")

    def evaluate(node):
        if isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float)):
                return node.value
            raise ValueError("Only numbers are allowed.")

        if isinstance(node, ast.BinOp):
            operator = _ALLOWED_OPERATORS.get(type(node.op))
            if operator is None:
                raise ValueError("Operator not supported.")
            return operator(evaluate(node.left), evaluate(node.right))

        if isinstance(node, ast.UnaryOp):
            operator = _ALLOWED_OPERATORS.get(type(node.op))
            if operator is None:
                raise ValueError("Operator not supported.")
            return operator(evaluate(node.operand))

        raise ValueError("Unsupported expression.")

    return evaluate(tree.body)


def calculator_tool(query: str) -> str:
    """Extract and calculate an arithmetic expression."""
    expression = query.strip()

    prefixes = [
        "calculate ",
        "compute ",
        "what is ",
        "solve ",
    ]

    lowered = expression.lower()

    for prefix in prefixes:
        if lowered.startswith(prefix):
            expression = expression[len(prefix):].strip()
            break

    try:
        result = safe_calculate(expression)
        return str(result)
    except Exception as exc:
        return f"Calculator error: {exc}"


# ============================================================
# RESEARCH TOOL
# ============================================================

def research_tool(query: str) -> str:
    """
    Local research tool for the MVP.
    Later this can be replaced with a real API or RAG retriever.
    """
    knowledge_base = {
        "rag": (
            "RAG means Retrieval-Augmented Generation. "
            "It retrieves relevant information from external knowledge "
            "before an LLM generates a grounded response."
        ),
        "langgraph": (
            "LangGraph is a framework for building stateful, "
            "graph-based agent workflows."
        ),
        "multi-agent": (
            "A multi-agent system uses multiple specialized agents "
            "that collaborate or are routed to solve different subtasks."
        ),
        "agent": (
            "An AI agent can interpret a task, choose actions or tools, "
            "execute them, observe results, and continue toward a goal."
        ),
    }

    query_lower = query.lower()

    for keyword, answer in knowledge_base.items():
        if keyword in query_lower:
            return answer

    return (
        f"No local research result found for: '{query}'. "
        "The research tool is currently using a small local knowledge base."
    )


# ============================================================
# SUPERVISOR AGENT
# ============================================================

def supervisor(state: AgentState) -> AgentState:
    query = state["user_query"].lower()

    calculation_words = [
        "calculate",
        "compute",
        "solve",
    ]

    arithmetic_symbols = ["+", "-", "*", "/", "%"]

    is_calculation = (
        any(word in query for word in calculation_words)
        or any(symbol in query for symbol in arithmetic_symbols)
    )

    route = "calculator" if is_calculation else "research"

    steps = list(state.get("steps", []))
    steps.append(f"SUPERVISOR -> {route.upper()} SPECIALIST")

    return {
        **state,
        "route": route,
        "steps": steps,
    }


# ============================================================
# CALCULATOR SPECIALIST
# ============================================================

def calculator_agent(state: AgentState) -> AgentState:
    query = state["user_query"]

    result = calculator_tool(query)

    steps = list(state.get("steps", []))
    steps.append("CALCULATOR SPECIALIST -> calculator_tool")

    return {
        **state,
        "tool_result": result,
        "steps": steps,
    }


# ============================================================
# RESEARCH SPECIALIST
# ============================================================

def research_agent(state: AgentState) -> AgentState:
    import requests

    query = state["user_query"]

    try:
        response = requests.get(
            "https://en.wikipedia.org/api/rest_v1/page/summary/"
            + query.replace(" ", "_"),
            timeout=10,
        )

        if response.status_code == 200:
            data = response.json()
            result = data.get("extract", "No result found.")
        else:
            from rag import build_context
            result = build_context(query)

    except requests.RequestException:
        from rag import build_context
        result = build_context(query)

    steps = list(state.get("steps", []))
    steps.append("RESEARCH SPECIALIST -> EXTERNAL API / RAG FALLBACK")

    return {
        **state,
        "tool_result": result,
        "steps": steps,
    }   
    from rag import build_context

    query = state["user_query"]

    context = build_context(query)

    steps = list(state.get("steps", []))
    steps.append("RAG RETRIEVER -> CONTEXT RETURNED")
    return {
        **state,
        "tool_result": context,
        "steps": steps,
    }    

    query = state["user_query"]

    result = research_tool(query)

    steps = list(state.get("steps", []))
    steps.append("RESEARCH SPECIALIST -> research_tool")

    return {
        **state,
        "tool_result": result,
        "steps": steps,
    }


# ============================================================
# FINALIZER
# ============================================================

def finalizer(state: AgentState) -> AgentState:
    result = state.get("tool_result", "No result available.")

    steps = list(state.get("steps", []))
    steps.append("FINALIZER -> final response")

    return {
        **state,
        "final_answer": result,
        "steps": steps,
    }


# ============================================================
# CONDITIONAL ROUTING
# ============================================================

def route_to_specialist(
    state: AgentState,
) -> Literal["calculator_agent", "research_agent"]:

    if state["route"] == "calculator":
        return "calculator_agent"

    return "research_agent"


# ============================================================
# BUILD LANGGRAPH
# ============================================================

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


# ============================================================
# RUN AGENT
# ============================================================

def run_agent(query: str) -> AgentState:
    return graph.invoke(
        {
            "user_query": query,
            "steps": [],
        }
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    if len(sys.argv) > 1:
        user_query = " ".join(sys.argv[1:])
    else:
        user_query = input("Enter your task: ").strip()

    result = run_agent(user_query)

    print("\nUSER:")
    print(user_query)

    print("\nAGENT:")
    print("Agent trace:")
    print(" -> ".join(result["steps"]))

    print("\nFINAL ANSWER:")
    print(result["final_answer"])
