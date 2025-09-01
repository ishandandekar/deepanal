import sys
from typing import Any
import time

from langchain.prompts import PromptTemplate
import asyncio

from src.schemas import DeepAnalState


async def fetch_tavily_for_search_queries(
    tavily_client, queries_from_llm: list[str], topic: str
):
    results = await asyncio.gather(
        *[
            tavily_client.search(
                query, return_exception=True, topic=topic, include_raw_content="text"
            )
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
        logger.info(logger_prefix + "Generated search queries: " + str(search_queries))
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
                logger_prefix + "Tavily raised an exception for query: " + str(query)
            )
            successful_results -= 1
            continue
        for tavily_result in search_result["results"]:
            processed_search_result = {}
            processed_search_result["query"] = query
            processed_search_result["title"] = tavily_result["title"]
            processed_search_result["url"] = tavily_result["url"]
            processed_search_result["content"] = tavily_result["content"]
            processed_search_result["raw_content"] = tavily_result["raw_content"]
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
    node_result["node_name"] = node_name
    return node_result


def financial_analyst(state: DeepAnalState):
    logger = state["logger"]
    financial_analysis_start = time.perf_counter()

    financial_query_generation_prompt = PromptTemplate.from_template("""
You are a senior financial analyst with 15 years of experience in corporate finance, valuation, and investment analysis across multiple industries. Your expertise lies in examining a company’s fundraising history, financial performance, and profitability drivers to understand its financial health and growth prospects.
You have been hired to research {company}, which operates in the {industry} sector. Your job is not to provide analysis yet, but to prepare search queries that will help gather intelligence about this company’s financial position.
Your queries should be:

- Tailored to the company’s industry context, not just generic finance checks.
- Designed to surface reliable data on fundraising rounds, valuation trends, and financial performance.
- Concise, clear, and directly usable in search engines.
- Spread across different dimensions of financial analysis.
- When generating queries, ensure they span these areas, but frame them in terms of the company within its industry:
- Fundraising History & Valuation – capital raises, investors, IPO/SPAC, valuation changes.
- Financial Statements & Key Metrics – balance sheet, income statement, cash flow, profitability ratios.
- Revenue & Profit Sources – revenue streams, cost structure, margins, and earnings drivers.

Generate 6 highly relevant, industry-specific search queries that you, as a financial analyst, would run to investigate this company. Each query should reference both {company} and {industry} to make it focused and precise.
""")

    research_result = reseacher_pipeline(
        state=state,
        node_name="Financial analyst",
        prompt=financial_query_generation_prompt,
        logger_prefix="Financial analyst: ",
        tavily_search_topic="finance",
    )
    financial_analysis_end = time.perf_counter()
    logger.info(
        "Financial analyst: "
        + "Time taken for node: "
        + time.strftime(
            "%M mins %S secs",
            time.gmtime(financial_analysis_end - financial_analysis_start),
        )
    )
    # import json
    #
    # with open("financial_node_output.json", "w") as f_out:
    #     json.dump(research_result, f_out)
    return {"financial_analyst_node_result": research_result}


def industry_analyst(state: DeepAnalState):
    logger = state["logger"]
    industry_analysis_start = time.perf_counter()

    industry_query_generation_prompt = PromptTemplate.from_template("""
You are a senior industry analyst with 15 years of experience in market research, competitive intelligence, and sectoral strategy assessment across multiple industries. Your expertise lies in understanding how companies are positioned within their industry, identifying competitors, and analyzing long-term market trends and challenges.
You have been hired to research {company}, which operates in the {industry} sector. Your job is not to provide analysis yet, but to prepare search queries that will help gather intelligence about the company’s position within its industry and the dynamics of the sector as a whole.
Your queries should be:
- Tailored to the company’s industry context, not just generic market checks.
- Designed to surface reliable data on competitive positioning, market trends, and sector growth.
- Concise, clear, and directly usable in search engines.
- Spread across different dimensions of industry analysis.
- When generating queries, ensure they span these areas, but frame them in terms of the company within its industry:
- Market Position – how {company} is positioned within the {industry} sector.
- Competitors – key rivals, market share comparisons, industry competition.
- {industry} Industry Trends & Challenges – emerging opportunities, risks, disruptions, and regulatory pressures.
- Market Size & Growth – current market size, forecasts, and long-term growth potential.

Generate 6 highly relevant, industry-specific search queries that you, as an industry analyst, would run to investigate this company. Each query should reference both {company} and {industry} to make it focused and precise.
""")
    research_result = reseacher_pipeline(
        state=state,
        node_name="Industry analyst",
        prompt=industry_query_generation_prompt,
        logger_prefix="Industry analyst: ",
        tavily_search_topic="general",
    )

    industry_analysis_end = time.perf_counter()

    logger.info(
        "Industry analyst: "
        + "Time taken for node: "
        + time.strftime(
            "%M mins %S secs",
            time.gmtime(industry_analysis_end - industry_analysis_start),
        )
    )
    return {"industry_analyst_node_result": research_result}


def company_analyst(state: DeepAnalState):
    logger = state["logger"]
    company_analysis_start = time.perf_counter()

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
        tavily_search_topic="general",
    )

    company_analysis_end = time.perf_counter()
    logger.info(
        "Company analyst: "
        + "Time taken for node: "
        + time.strftime(
            "%M mins %S secs",
            time.gmtime(company_analysis_end - company_analysis_start),
        )
    )
    return {"company_analyst_node_result": research_result}


def news_analyst(state: DeepAnalState):
    logger = state["logger"]
    news_analysis_start = time.perf_counter()

    news_query_generation_prompt = PromptTemplate.from_template("""
You are a senior news and media analyst with 15 years of experience in corporate communications tracking, press monitoring, and market-moving news analysis. Your expertise lies in identifying recent announcements, partnerships, and press releases that may impact a company’s perception, reputation, and business outlook.
You have been hired to research {company}, which operates in the {industry} sector. Your job is not to provide analysis yet, but to prepare search queries that will help gather the most recent updates and news flow about the company.
Your queries should be:
- Focused on recent developments, not historical background.
- Designed to surface official announcements, partnerships, and media coverage.
- Concise, clear, and directly usable in search engines.
- Spread across different dimensions of news monitoring.
- When generating queries, ensure they span these areas, but frame them in terms of the company within its industry:
- Recent Company Announcements – updates, initiatives, and corporate developments.
- Press Releases – official statements, product launches, financial disclosures.
- New Partnerships – alliances, collaborations, and joint ventures.

Generate 6 highly relevant, company-specific search queries that you, as a news analyst, would run to investigate this company. Each query should reference both {company} and {industry} to make it focused and precise.
""")
    research_result = reseacher_pipeline(
        state=state,
        node_name="News analyst",
        prompt=news_query_generation_prompt,
        logger_prefix="News analyst: ",
        tavily_search_topic="news",
    )

    news_analysis_end = time.perf_counter()
    logger.info(
        "News analyst: "
        + "Time taken for node: "
        + time.strftime(
            "%M mins %S secs",
            time.gmtime(news_analysis_end - news_analysis_start),
        )
    )
    return {"news_analyst_node_result": research_result}


def controversy_analyst(state: DeepAnalState):
    logger = state["logger"]
    controversy_analysis_start = time.perf_counter()

    controversy_query_generation_prompt = PromptTemplate.from_template("""
You are a senior sentiment and controversy analyst with 15 years of experience in media monitoring, reputation analysis, and corporate risk intelligence. Your expertise lies in identifying public perception issues, controversies, and negative events that could affect a company’s reputation and stakeholder trust.
You have been hired to research {company}, which operates in the {industry} sector. Your job is not to provide analysis yet, but to prepare search queries that will help gather intelligence about the company’s reputation, controversies, and sentiment in the market.
Your queries should be:
- Focused on uncovering risks, controversies, or negative sentiment, not just neutral updates.
- Designed to surface lawsuits, scandals, negative press, and public criticism.
- Concise, clear, and directly usable in search engines.
- Spread across different dimensions of reputation monitoring.
- When generating queries, ensure they span these areas, but frame them in terms of the company within its industry:
- Public Perception & Media Sentiment – reputation, customer reviews, and press coverage tone.
- Legal & Regulatory Controversies – lawsuits, fines, compliance failures.
- Social & Ethical Issues – labor disputes, ESG controversies, activist criticism.
- Negative News & Scandals – fraud, governance failures, scandals, or crises.

Generate 6 highly relevant, industry-specific search queries that you, as a sentiment analyst, would run to investigate this company. Each query should reference both {company} and {industry} to make it focused and precise.
""")
    research_result = reseacher_pipeline(
        state=state,
        node_name="Controversy analyst",
        prompt=controversy_query_generation_prompt,
        logger_prefix="Controversy analyst: ",
        tavily_search_topic="general",
    )

    controversy_analysis_end = time.perf_counter()
    logger.info(
        "Controversy analyst: "
        + "Time taken for node: "
        + time.strftime(
            "%M mins %S secs",
            time.gmtime(controversy_analysis_end - controversy_analysis_start),
        )
    )
    return {"controversy_analyst_node_result": research_result}
