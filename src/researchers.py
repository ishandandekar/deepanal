import sys
from typing import Any
import time

from langchain.prompts import PromptTemplate
from rich.console import Console
from rich.progress import Progress
import asyncio

from src.schemas import DeepAnalState


async def fetch_tavily_for_search_queries(
    tavily_client, queries_from_llm: list[str], topic: str
):
    results = await asyncio.gather(
        *[
            tavily_client.search(query, return_exception=True, topic=topic)
            for query in queries_from_llm
        ]
    )
    return results


def reseacher_pipeline(
    state: DeepAnalState,
    node_name: str,
    prompt: PromptTemplate,
    logger_prefix: str,
    tavily_search_topic: str,
) -> dict[str, Any]:
    company = state["company"]
    industry = state["industry"]
    location = state["location"]
    query_generator_llm = state["query_generator_llm"]
    tavily_client = state["tavily_client"]
    logger = state["logger"]
    cns = state["console"]
    logger.info(logger_prefix + "Starting analysis")
    node_result = {}

    query_generation_chain = prompt | query_generator_llm

    logger.info(logger_prefix + "Generating search queries")

    try:
        search_queries = query_generation_chain.invoke(
            {"company": company, "industry": industry, "location": location}
        ).queries
        logger.info(logger_prefix + "Generated search queries: " +
                    str(search_queries))
    except Exception:
        logger.error(
            logger_prefix + "Error while getting queries from chain", exc_info=True
        )
        sys.exit(1)
    node_result["queries"] = search_queries

    logger.info(logger_prefix + "Searching for queries using Tavily")
    try:
        search_results = asyncio.run(
            fetch_tavily_for_search_queries(
                tavily_client=tavily_client,
                queries_from_llm=search_queries,
                topic=tavily_search_topic,
            )
        )
    except Exception:
        logger.error(
            logger_prefix + "Error while searching for queries using Tavily",
            exc_info=True,
        )
        sys.exit(1)
    node_result["raw_search_results"] = search_results
    successful_results = len(search_results)
    processed_search_results = []
    for query, search_result in zip(search_queries, search_results):
        if isinstance(search_result, Exception):
            logger.info(
                logger_prefix +
                "Tavily raised an exception for query: " + str(query)
            )
            successful_results -= 1
            continue
        for tavily_result in search_result["results"]:
            processed_search_result = {}
            processed_search_result["query"] = query
            processed_search_result["title"] = tavily_result["title"]
            processed_search_result["url"] = tavily_result["url"]
            processed_search_result["content"] = tavily_result["content"]
            processed_search_result["score"] = tavily_result["score"]
            processed_search_results.append(processed_search_result)
    logger.info(
        logger_prefix
        + "Successful searches: "
        + str(successful_results)
        + " Failed searches: "
        + str(len(search_queries) - successful_results)
    )
    node_result["processed_search_results"] = processed_search_results
    logger.info(
        logger_prefix
        + "Number of processed results: "
        + str(len(processed_search_results))
    )
    node_result["text_content"] = "\n".join(
        [item["content"] for item in processed_search_results]
    )
    return node_result


def financial_analyst(state: DeepAnalState):
    logger = state["logger"]
    financial_analysis_start = time.perf_counter()
    financial_analyst_node_result = {}

    financial_query_generation_prompt = PromptTemplate.from_template("""
You are a senior company researcher with 15 years of experience in corporate research, equity analysis, and business strategy assessment across multiple industries. Your expertise lies in uncovering a company’s fundamentals, business drivers, leadership, and long-term positioning in its sector.
You have been hired to research {company}, which operates in the {industry} sector. Your job is not to provide analysis yet, but to prepare search queries that will help gather intelligence about this company’s core operations and strategy.
Your queries should be:
- Tailored to the company’s industry context, not just generic information checks.
- Designed to surface reliable data on its business model, offerings, leadership, and growth path.
- Concise, clear, and directly usable in search engines.
- Spread across different dimensions of company research.
- When generating queries, ensure they span these areas, but frame them in terms of the company within its industry:
- Core Products & Services – offerings, solutions, and customer base.
- Company History & Milestones – founding, expansions, IPOs, acquisitions.
- Leadership Team – executives, board members, founders.
- Business Model & Strategy – revenue sources, competitive advantage, growth vision.

Generate 6 highly relevant, industry-specific search queries that you, as a company researcher, would run to investigate this company. Each query should reference the {company} to make it focused and precise.
        """)
    research_result = reseacher_pipeline(
        state=state,
        node_name="Financial analyst",
        prompt=financial_query_generation_prompt,
        logger_prefix="Financial analyst: ",
        tavily_search_topic="finance",
    )
    return {"financial_analyst_output": research_result}


def industry_analyst(state: DeepAnalState): ...


def company_analyst(state: DeepAnalState):
    logger = state["logger"]
    company_analysis_start = time.perf_counter()
    company_analyst_node_result = {}

    company_query_generation_prompt = PromptTemplate.from_template("""
You are a senior company researcher with 15 years of experience in corporate research, equity analysis, and business strategy assessment across multiple industries. Your expertise lies in uncovering a company’s fundamentals, business drivers, leadership, and long-term positioning in its sector.
You have been hired to research {company}, which operates in the {industry} sector. Your job is not to provide analysis yet, but to prepare search queries that will help gather intelligence about this company’s core operations and strategy.
Your queries should be:
- Tailored to the company’s industry context, not just generic information checks.
- Designed to surface reliable data on its business model, offerings, leadership, and growth path.
- Concise, clear, and directly usable in search engines.
- Spread across different dimensions of company research.
- When generating queries, ensure they span these areas, but frame them in terms of the company within its industry:
- Core Products & Services – offerings, solutions, and customer base.
- Company History & Milestones – founding, expansions, IPOs, acquisitions.
- Leadership Team – executives, board members, founders.
- Business Model & Strategy – revenue sources, competitive advantage, growth vision.

Generate 6 highly relevant, industry-specific search queries that you, as a company researcher, would run to investigate this company. Each query should reference the {company} to make it focused and precise.
        """)
    research_result = reseacher_pipeline(
        state=state,
        node_name="Company analyst",
        prompt=company_query_generation_prompt,
        logger_prefix="Company analyst: ",
        tavily_search_topic="finance",
    )

    return {"company_analyst_node_result": research_result}


def news_analyst(state: DeepAnalState): ...
def risk_analyst(state: DeepAnalState): ...
