from logging import Logger
from typing import Annotated, Any, TypedDict

from langchain_core.callbacks import UsageMetadataCallbackHandler
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AnyMessage
from langgraph.graph import add_messages
from pydantic import BaseModel, Field
from rich.console import Console
from tavily import TavilyClient


class DeepAnalState(TypedDict):
    company: str
    location: str
    industry: str
    llm: BaseChatModel
    tavily_client: TavilyClient
    logger: Logger
    company_analyst_node_result: dict[str, Any]
    financial_analyst_node_result: dict[str, Any]
    news_analyst_node_result: dict[str, Any]
    controversy_analyst_node_result: dict[str, Any]
    industry_analyst_node_result: dict[str, Any]
    messages: Annotated[list[AnyMessage], add_messages]
    console: Console
    llm_usage_callback: UsageMetadataCallbackHandler
    query_generator_llm: BaseChatModel


class SearchQueries(BaseModel):
    """Queries for search engines based on desciption and role"""

    queries: list[str] = Field(
        description="Search queries based on description", max_length=6, min_length=6
    )
