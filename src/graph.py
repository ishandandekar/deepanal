from langgraph.graph import StateGraph
from langgraph.graph.state import CompiledStateGraph
from src.schemas import DeepAnalState
from src.researchers import (
    financial_analyst,
    news_analyst,
    company_analyst,
    industry_analyst,
    controversy_analyst,
)
from src.processors import grounding, relevance_evaluator, briefer, editor

workflow: CompiledStateGraph = (
    StateGraph(DeepAnalState)
    # Add processors
    .add_node("grounding", grounding)
    .add_node("curator", relevance_evaluator)
    .add_node("briefer", briefer)
    .add_node("editor", editor)
    # Add researchers
    .add_node("financial_analyst", financial_analyst)
    .add_node("news_analyst", news_analyst)
    .add_node("company_analyst", company_analyst)
    .add_node("industry_analyst", industry_analyst)
    .add_node("controversy_analyst", controversy_analyst)
    # Setting up edges
    .set_entry_point("grounding")
    .add_edge("grounding", "financial_analyst")
    .add_edge("grounding", "news_analyst")
    .add_edge("grounding", "company_analyst")
    .add_edge("grounding", "industry_analyst")
    .add_edge("grounding", "controversy_analyst")
    .add_edge("financial_analyst", "curator")
    .add_edge("news_analyst", "curator")
    .add_edge("company_analyst", "curator")
    .add_edge("industry_analyst", "curator")
    .add_edge("controversy_analyst", "curator")
    .add_edge("curator", "briefer")
    .add_edge("briefer", "editor")
    .set_finish_point("editor")
    .compile()
)
workflow.name = "DeepAnal"
