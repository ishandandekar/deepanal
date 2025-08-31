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
    company_url: str
    llm: BaseChatModel
    tavily_client: TavilyClient
    logger: Logger
    foo_analyst_message: dict[str, Any]
    messages: Annotated[list[AnyMessage], add_messages]
    console: Console
    llm_usage_callback: UsageMetadataCallbackHandler
    query_generator_llm: BaseChatModel
    site_scrape: str


class SearchQueries(BaseModel):
    """Queries for search engines based on desciption and role"""

    queries: list[str] = Field(
        description="Search queries based on description", max_length=6, min_length=6
    )
