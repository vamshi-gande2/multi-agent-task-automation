from __future__ import annotations

import ast
import operator
import os
from typing import Annotated, Literal, TypedDict

from dotenv import load_dotenv

load_dotenv()

try:
    from langchain_anthropic import ChatAnthropic
    from langchain_core.messages import HumanMessage, SystemMessage
    from langgraph.graph import END, START, StateGraph
    from langgraph.graph.message import add_messages
    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    route: str
    specialist_result: str
    steps: int


_ALLOWED_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def _safe_eval(node: ast.AST) -> float:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_OPS:
        return _ALLOWED_OPS[type(node.op)](_safe_eval(node.operand))
    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_OPS:
        return _ALLOWED_OPS[type(node.op)](_safe_eval(node.left), _safe_eval(node.right))
    raise ValueError("Only basic arithmetic is supported.")


def calculator(expression: str) -> str:
    """Evaluate a basic arithmetic expression safely."""
    try:
        value = _safe_eval(ast.parse(expression, mode="eval").body)
        return str(int(value) if value.is_integer() else value)
    except Exception as exc:
        return f"Calculator error: {exc}"


LOCAL_KB = {
    "flytbase": "FlytBase builds software for autonomous drones and robots, connecting AI, telemetry, sensors, cameras and workflows.",
    "langgraph": "LangGraph is an orchestration framework for stateful agent workflows. It models nodes, shared state and conditional execution.",
    "rag": "Retrieval-Augmented Generation retrieves relevant context before generation so responses can be grounded in a knowledge source.",
    "agent": "An agent uses a model to decide an action, executes tools or specialist steps, observes results, and continues until it can answer.",
}


def search_knowledge(query: str) -> str:
    """Search the demo knowledge base."""
    q = query.lower()
    matches = []
    for key, text in LOCAL_KB.items():
        tokens = [w for w in q.split() if len(w) >= 5]
        if key in q or any(w in text.lower() for w in tokens):
            matches.append(f"{key}: {text}")
    return "\n".join(matches[:3]) or "No matching knowledge-base entry found."


def _route_from_text(text: str) -> str:
    t = text.upper()
    if "RESEARCH" in t or "SEARCH" in t:
        return "researcher"
    if "CALCULATE" in t or "MATH" in t or "ARITHMETIC" in t:
        return "calculator"
    # Let mixed tasks use both specialists.
    if any(ch.isdigit() for ch in text) and any(op in text for op in ["+", "-", "*", "/"]):
        return "calculator"
    return "researcher"


if LANGGRAPH_AVAILABLE:
    MODEL_NAME = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-5")

    def build_graph():
        llm = ChatAnthropic(model=MODEL_NAME, temperature=0)

        def supervisor(state: AgentState):
            user_text = state["messages"][-1].content
            response = llm.invoke([
                SystemMessage(content=(
                    "You are the supervisor of a multi-agent workflow. "
                    "Read the task and output exactly one word: RESEARCH or CALCULATE. "
                    "Choose RESEARCH for factual/knowledge retrieval tasks and CALCULATE for arithmetic tasks. "
                    "Do not answer the task yourself."
                )),
                HumanMessage(content=user_text),
            ])
            route = _route_from_text(response.content)
            return {"route": route, "steps": state.get("steps", 0) + 1}

        def researcher(state: AgentState):
            user_text = state["messages"][-1].content
            context = search_knowledge(user_text)
            response = llm.invoke([
                SystemMessage(content=(
                    "You are the research specialist. Use ONLY the supplied knowledge-base result. "
                    "If it does not contain the answer, say that the local knowledge base does not contain it."
                )),
                HumanMessage(content=f"Task: {user_text}\nKnowledge-base result:\n{context}"),
            ])
            return {"specialist_result": response.content, "steps": state.get("steps", 0) + 1}

        def calculator_agent(state: AgentState):
            user_text = state["messages"][-1].content
            response = llm.invoke([
                SystemMessage(content=(
                    "You are the calculation specialist. Extract the arithmetic expression from the task. "
                    "Use the safe calculator conceptually and return the numeric result with one short explanation. "
                    "Do not invent missing numbers."
                )),
                HumanMessage(content=user_text),
            ])
            return {"specialist_result": response.content, "steps": state.get("steps", 0) + 1}

        def finalizer(state: AgentState):
            user_text = state["messages"][-1].content
            result = state.get("specialist_result", "")
            response = llm.invoke([
                SystemMessage(content="You are the final response agent. Give a concise, useful answer based only on the specialist result."),
                HumanMessage(content=f"Original task: {user_text}\nSpecialist result: {result}"),
            ])
            return {"messages": [response], "steps": state.get("steps", 0) + 1}

        def route(state: AgentState) -> Literal["researcher", "calculator"]:
            return state["route"]

        graph = StateGraph(AgentState)
        graph.add_node("supervisor", supervisor)
        graph.add_node("researcher", researcher)
        graph.add_node("calculator", calculator_agent)
        graph.add_node("finalizer", finalizer)
        graph.add_edge(START, "supervisor")
        graph.add_conditional_edges("supervisor", route, {
            "researcher": "researcher",
            "calculator": "calculator",
        })
        graph.add_edge("researcher", "finalizer")
        graph.add_edge("calculator", "finalizer")
        graph.add_edge("finalizer", END)
        return graph.compile()


def mock_run(user_query: str) -> str:
    """Offline deterministic demonstration of the same supervisor/specialist design."""
    traces = ["SUPERVISOR"]
    q = user_query.lower()
    route = "calculator" if any(ch.isdigit() for ch in q) and any(op in q for op in ["+", "-", "*", "/"]) else "researcher"

    if route == "calculator":
        traces.append("CALCULATOR SPECIALIST")
        # Deterministic demos; real mode extracts arbitrary expressions with the LLM.
        compact = q.replace(" ", "")
        expression = next((x for x in ["240/40", "25*12", "18/30*60", "100+25"] if x in compact), None)
        result = calculator(expression) if expression else "No supported demo expression found."
    else:
        traces.append("RESEARCH SPECIALIST")
        result = search_knowledge(user_query)

    traces.append("FINALIZER")
    return f"Agent trace: {' -> '.join(traces)}\n\n{result}"


def run(user_query: str) -> str:
    if not LANGGRAPH_AVAILABLE or not os.getenv("ANTHROPIC_API_KEY"):
        return mock_run(user_query)
    graph = build_graph()
    result = graph.invoke({
        "messages": [HumanMessage(content=user_query)],
        "route": "",
        "specialist_result": "",
        "steps": 0,
    })
    return result["messages"][-1].content


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="FlytBase-style multi-agent LangGraph MVP")
    parser.add_argument("query", nargs="*", help="Task for the agent")
    args = parser.parse_args()
    query = " ".join(args.query).strip() or "Explain what RAG is and calculate 240 / 40."
    print("USER:", query)
    print("\nAGENT:\n", run(query))
