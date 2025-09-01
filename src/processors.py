import os
import sys
import asyncio
import time
from langchain.chat_models import init_chat_model
from langchain_core.callbacks import UsageMetadataCallbackHandler
from tavily import AsyncTavilyClient

from langchain.prompts import PromptTemplate
from src.schemas import DeepAnalState, SearchQueries


def grounding(state: DeepAnalState):
    grounding_start = time.perf_counter()
    company_url = state["company_url"]
    logger = state["logger"]
    cns = state["console"]

    logger_prefix = "Grounding node: "

    gemini_llm = init_chat_model(
        "gemini-2.0-flash", model_provider="google_genai", temperature=0
    )
    ollama_llm = init_chat_model("ollama:deepseek-r1:7b", reasoning=False)
    usage_callback = UsageMetadataCallbackHandler()

    llm = gemini_llm.with_fallbacks([ollama_llm]).with_config(
        {"callbacks": [usage_callback]}
    )

    query_generator_llm = llm.with_structured_output(SearchQueries)
    tavily_client = AsyncTavilyClient(api_key=os.environ["TAVILY_API_KEY"])

    logger.info(logger_prefix + "Instantiating LLM and Tavily client")
    logger.info(
        logger_prefix + "Extracting info from company's website. URL: " + company_url
    )
    try:
        company_site_url = asyncio.run(
            tavily_client.extract(
                urls=company_url,
                include_images=False,
                extract_depth="advanced",
                format="markdown",
                include_favicon=False,
            )
        )
        extract_response = company_site_url.get("results", None)
        if extract_response is None:
            logger.error(
                logger_prefix
                + "Could not extract text from company's URL. Stopping execution"
            )
            sys.exit(1)
        site_info = extract_response[0].get("raw_content", None)
        if site_info is None:
            logger.error(
                logger_prefix
                + "Could not extract text from company's URL. Stopping execution"
            )
            sys.exit(1)
    except Exception:
        logger.error(
            logger_prefix
            + "Could not extract text from company's URL. Stopping execution"
        )
        sys.exit(1)

    grounding_end = time.perf_counter()
    time_taken = grounding_end - grounding_start
    logger.info(
        logger_prefix
        + "Grouding finished. Time taken: "
        + time.strftime("%M mins %S secs", time.gmtime(time_taken))
    )
    return {
        "llm": llm,
        "tavily_client": tavily_client,
        "query_generator_llm": query_generator_llm,
        "llm_usage_callback": usage_callback,
        "site_scrape": site_info,
    }


def relevance_evaluator(state: DeepAnalState):
    cns = state["console"]
    logger = state["logger"]
    relevancy_score = 0.5

    logger_prefix = "Relevance evaluator: "
    company_node_result = state["company_analyst_node_result"]
    financial_node_result = state["financial_analyst_node_result"]
    industry_node_result = state["industry_analyst_node_result"]
    news_node_result = state["news_analyst_node_result"]
    controversy_node_result = state["controversy_analyst_node_result"]
    logger.info(logger_prefix + "Starting relevancy filtering")
    for node_result in (
        company_node_result,
        financial_node_result,
        industry_node_result,
        news_node_result,
        controversy_node_result,
    ):
        relevant_docs = list(
            filter(
                lambda doc: (doc["score"] >= relevancy_score)
                and (doc["raw_content"] is not None),
                node_result["processed_search_results"],
            )
        )
        if len(relevant_docs) > 0:
            logger.info(
                logger_prefix
                + "For "
                + str(node_result["node_name"])
                + " filtered "
                + str(len(relevant_docs))
                + " documents out of "
                + str(len(node_result["processed_search_results"]))
            )
        else:
            logger.warning(
                logger_prefix
                + "For "
                + str(node_result["node_name"])
                + " no relevant documents were found amongst "
                + str(len(node_result["processed_search_results"]))
            )
        node_result["relevant_docs"] = relevant_docs
    return {
        "company_analyst_node_result": company_node_result,
        "financial_analyst_node_result": financial_node_result,
        "industry_analyst_node_result": industry_node_result,
        "news_analyst_node_result": news_node_result,
        "controversy_analyst_node_result": controversy_node_result,
    }


def briefer(state: DeepAnalState):
    company = state["company"]
    industry = state["industry"]
    location = state["location"]
    cns = state["console"]
    llm = state["llm"]
    logger = state["logger"]
    max_content_length_per_section = 8000
    logger_prefix = "Briefer: "
    company_node_result = state["company_analyst_node_result"]
    financial_node_result = state["financial_analyst_node_result"]
    industry_node_result = state["industry_analyst_node_result"]
    news_node_result = state["news_analyst_node_result"]
    controversy_node_result = state["controversy_analyst_node_result"]

    briefing_prompts = {
        "Industry analyst": """Create a focused industry briefing for {company}, a {industry} company based in {location}.
Key requirements:
1. Structure into these categories using bullet points:
    a. Market Position
        Relative market standing
        Share of market
    b. Competitors
        Key rivals
        Competitive strengths/weaknesses
    c. {industry} Trends & Challenges
        Emerging opportunities
        Industry headwinds
    d. Market Size & Growth
        Current size
        Forecast growth rates
2. Sort newest to oldest where applicable
3. One fact per bullet point
4. Do not mention "no information found" or "no data available"
5. Never use ### headers, only bullet points
6. Provide only the briefing. Do not provide explanations or commentary.
7. Use only the given information""",
        "Company analyst": """Create a focused company fundamentals briefing for {company}, a {industry} company based in {location}.
Key requirements:
1. Structure into these categories using bullet points:
    - Core Products & Services
        Main offerings
        Key customer segments
    - Business Model
        Revenue streams
        Monetization approach
    - Leadership
        CEO and top executives
        Board highlights
    - History & Milestones
        Founding details
        Major events, expansions, or acquisitions
2. Sort items newest to oldest where applicable
3. One fact per bullet point
4. Do not mention "no information found" or "no data available"
5. Never use ### headers, only bullet points
6. Use only the given information
7. Provide only the briefing. Do not provide explanations or commentary.""",
        "Financial analyst": """Create a focused financial briefing for {company}, a {industry} company based in {location}.
Key requirements:
1. Structure into these categories using bullet points:
    - Fundraising & Valuation
        - Capital raises
        - Valuation changes
    - Key Metrics
        - Revenue
        - Profit
        - Margins
        - Growth
    - Revenue & Profit Sources
        - Segment breakdown
        - Geographic distribution
2. Sort newest to oldest where applicable
3. One fact per bullet point
4. Do not mention "no information found" or "no data available"
5. Never use ### headers, only bullet points
6. Use only the given information
7. Provide only the briefing. Do not provide explanations or commentary.""",
        "News analyst": """Create a focused news briefing for {company}, a {industry} company based in {location}.
Key requirements:
1. Structure into these categories using bullet points:
    - Major Announcements
        - Product / service launches
        - New initiatives
    - Partnerships
        - Integrations
        - Collaborations
    - Recognition
        - Awards
        - Press coverage
2. Sort newest to oldest
3. One event per bullet point
4. Do not mention "no information found" or "no data available"
5. Never use ### headers, only bullet points
6. Use only the given information
7. Provide only the briefing. Do not provide explanations or commentary.""",
        "Controversy analyst": """Create a focused sentiment and controversy briefing for {company}, a {industry} company based in {location}.
Key requirements:
1. Structure into these categories using bullet points:
    - Public Sentiment
        - Media tone
        - Customer perception
    - Legal & Regulatory
        - Lawsuits
        - Fines
        - Compliance issues
    - ESG & Ethical Issues
        - Environmental concerns
        - Social or labor controversies
        - Governance failures
    - Negative Events
        - Scandals
        - Crises
2. Sort newest to oldest where applicable
3. One fact per bullet point
4. Do not mention "no information found" or "no data available"
5. Never use ### headers, only bullet points
6. Use only the given information
7. Provide only the briefing. Do not provide explanations or commentary.""",
    }

    logger.info(logger_prefix + "Creating briefings for researchers")
    for node_result in (
        company_node_result,
        financial_node_result,
        industry_node_result,
        news_node_result,
        controversy_node_result,
    ):
        if len(node_result["relevant_docs"]) == 0:
            logger.warning(
                logger_prefix
                + "No relevant documents found for "
                + node_result["node_name"]
                + ". Skipping briefing"
            )
            node_result["brief"] = "No information provided"
            continue
        relevant_docs = node_result["relevant_docs"]
        sorted_relevant_docs = sorted(
            relevant_docs,
            key=lambda x: float(x["score"]),
            reverse=True,
        )

        doc_texts = []
        total_length = 0
        for doc in sorted_relevant_docs:
            title = doc.get("title", "")
            content = doc.get("raw_content")
            if content is None:
                cns.print(
                    "NODE: "
                    + node_result["node_name"]
                    + " Title: "
                    + doc["title"]
                    + " URL: "
                    + doc["url"]
                )
                logger.warning(logger_prefix + node_result["node_name"])
            if len(content) > max_content_length_per_section:
                content = (
                    content[:max_content_length_per_section] + "... [content truncated]"
                )
            doc_entry = f"Title: {title}\n\nContent: {content}"
            if total_length + len(doc_entry) < 100000:  # Keep under limit
                doc_texts.append(doc_entry)
                total_length += len(doc_entry)
            else:
                break

        separator = "\n" + "-" * 40 + "\n"
        briefing_prompt = (
            briefing_prompts[node_result["node_name"]].format(
                company=company, industry=industry, location=location
            )
            + "\n"
            + "Analyze the following documents and extract key information."
            "Provide only the briefing, no explanations or commentary:"
            + separator
            + separator.join(doc_texts)
            + separator
        )
        # cns.print(briefing_prompt)
        # briefing_prompt = PromptTemplate.from_template(
        #     briefing_prompt.format(
        #         company=company, industry=industry, location=location
        #     )
        # )
        briefing_chain = llm
        try:
            node_brief = briefing_chain.invoke(briefing_prompt)
        except Exception:
            logger.error(
                logger_prefix
                + " Error occured while creating briefing for "
                + node_result["node_name"],
                exc_info=True,
            )
            sys.exit(1)

        logger.info(logger_prefix + "Created briefing for " + node_result["node_name"])
        node_result["brief"] = node_brief
        cns.print(node_brief)

    return {
        "company_analyst_node_result": company_node_result,
        "financial_analyst_node_result": financial_node_result,
        "industry_analyst_node_result": industry_node_result,
        "news_analyst_node_result": news_node_result,
        "controversy_analyst_node_result": controversy_node_result,
    }


def editor(state: DeepAnalState): ...
