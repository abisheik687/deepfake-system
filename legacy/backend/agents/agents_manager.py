"""
KAVACH-AI — Master Agency Orchestrator (LangGraph)
Coordinates the specialized agents in a unified 'Mission Control' workflow.
"""

from typing import Dict, TypedDict, Annotated, List, Union, Any
from loguru import logger
from langgraph.graph import StateGraph, END

# Define the state of our Agency workflow
class AgencyState(TypedDict):
    media_data: Dict[str, Any]
    forensic_results: Dict[str, Any]
    verdict: str
    is_deepfake: bool
    findings: List[str]
    report_path: str
    public_summary: str

# ─── Nodes (Agent Functions) ──────────────────────────────────────────────────

async def fact_check_node(state: AgencyState) -> AgencyState:
    logger.info("[🛡️ Agency] Initializing Neural Inquest: Fact-Checker Agent")
    from backend.agents.fact_checker import FactCheckerAgent
    agent = FactCheckerAgent()
    verification = await agent.verify(state["media_data"])
    
    state["findings"].append(f"Fact-Checker: Media cross-referenced against Pinecone Global Matrix. Outcome: {verification['findings']}")
    return state

async def forensic_analysis_node(state: AgencyState) -> AgencyState:
    logger.info("[🔬 Agency] Initiating Forensic Synthesis Cycle")
    from backend.agents.forensic_reporter import ForensicReporterAgent
    agent = ForensicReporterAgent()
    
    # Generate the signed PDF bundle
    report_path = agent.generate_report(state["forensic_results"])
    state["report_path"] = report_path
    state["findings"].append(f"Forensic Analyst: Biometric signals synthesized into Signed PDF Bundle [ID: {report_path.split('/')[-1]}]")
    state["findings"].append("Forensic Analyst: SHA-256 Chain-of-Evidence established.")
    return state

async def communication_node(state: AgencyState) -> AgencyState:
    logger.info("[📝 Agency] Crafting Semantic Mission Briefing")
    from backend.agents.journalist import JournalistAgent
    agent = JournalistAgent()
    
    summary = agent.generate_summary(state["forensic_results"])
    state["public_summary"] = summary
    state["findings"].append(f"Journalist: Public briefing protocols finalized. Outcome: {summary[:50]}...")
    return state

# ─── Graph Construction ──────────────────────────────────────────────────────

def create_agency_graph():
    workflow = StateGraph(AgencyState)

    # Add nodes
    workflow.add_node("fact_checker", fact_check_node)
    workflow.add_node("forensic_analyst", forensic_analysis_node)
    workflow.add_node("journalist", communication_node)

    # Define edges: Sequential flow for Phase 3
    workflow.set_entry_point("fact_checker")
    workflow.add_edge("fact_checker", "forensic_analyst")
    workflow.add_edge("forensic_analyst", "journalist")
    workflow.add_edge("journalist", END)

    return workflow.compile()

# Global master orchestrator instance
agency_app = create_agency_graph()

async def run_agency_orchestration(media_metadata: dict, detection_results: dict):
    """KAVACH-AI Mission Control Entry Point."""
    initial_state = {
        "media_data": media_metadata,
        "forensic_results": detection_results,
        "verdict": detection_results.get("verdict", "UNKNOWN"),
        "is_deepfake": detection_results.get("verdict") == "FAKE",
        "findings": [],
        "report_path": "",
        "public_summary": "",
    }
    logger.info("🛡️ Starting KAVACH-AI Master Agency Orchestration...")
    final_state = await agency_app.ainvoke(initial_state)
    logger.success("🛡️ Mission Control cycle complete.")
    return final_state


async def dispatch(task_type: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Dispatch to the correct agent by task_type. Returns agent result or structured error.
    Logs all invocations with timestamp and duration.
    """
    import time
    t0 = time.perf_counter()
    try:
        if task_type == "fact_check":
            from backend.agents.fact_checker import FactCheckerAgent
            agent = FactCheckerAgent()
            out = await agent.verify(payload.get("media_metadata", payload))
        elif task_type == "forensic_report":
            from backend.agents.forensic_reporter import ForensicReporterAgent
            agent = ForensicReporterAgent()
            out = {"report_path": agent.generate_report(
                payload.get("detection_result", payload),
                detection_id=payload.get("detection_id"),
                file_hash_sha256=payload.get("file_hash"),
            )}
        elif task_type == "journalist":
            from backend.agents.journalist import JournalistAgent
            agent = JournalistAgent()
            out = {"summary": agent.generate_summary(payload.get("detection_data", payload))}
        else:
            out = {"error": f"Unknown task_type: {task_type}"}
        duration_ms = (time.perf_counter() - t0) * 1000
        logger.info(f"[AgentsManager] {task_type} completed in {duration_ms:.0f}ms")
        return {"ok": True, "result": out, "duration_ms": round(duration_ms, 1)}
    except Exception as e:
        duration_ms = (time.perf_counter() - t0) * 1000
        logger.error(f"[AgentsManager] {task_type} failed after {duration_ms:.0f}ms: {e}")
        return {"ok": False, "error": str(e), "duration_ms": round(duration_ms, 1)}
