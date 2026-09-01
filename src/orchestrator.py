#!/usr/bin/env python3
"""
FinOps-Security-Agent — LangGraph Multi-Agent Orchestrator Interface
Provides backward-compatible interface pointing to the LangGraph StateGraph engine.
"""

from src.langgraph_orchestrator import LangGraphOrchestrator

class DecisionOrchestrator(LangGraphOrchestrator):
    """
    Multi-Agent Decision Orchestrator powered by LangGraph StateGraph Workflows.
    Synthesizes signals from ML Engine, FinOps Agent, and Security Agent into a 3-way verdict
    and appends every decision to an immutable SHA-256 Cryptographic Hash Chain.
    """
    pass

orchestrator = DecisionOrchestrator()
