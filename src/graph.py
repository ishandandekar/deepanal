from langgraph.graph import StateGraph
from src.schemas import DeepAnalState
from src.researchers import (
    financial_analyst,
    news_analyst,
    company_analyst,
    industry_analyst,
    risk_analyst,
)
from src.processors import grounding, collector, curator, breifer, editor

graph = StateGraph(DeepAnalState)

