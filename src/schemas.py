from typing import Annotated, Any, TypedDict
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AnyMessage
from logging import Logger
from tavily import TavilyClient
from langgraph.graph import add_messages


class DeepAnalState(TypedDict):
    company: str
    location: str
    industry: str
    company_url: str | None
    llm: BaseChatModel
    tavily_client: TavilyClient
    logger: Logger
    foo_analyst_message: dict[str, Any]
    messages: Annotated[list[AnyMessage], add_messages]
