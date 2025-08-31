import os
import sys
import asyncio
import time
from langchain.chat_models import init_chat_model
from langchain_core.callbacks import UsageMetadataCallbackHandler
from tavily import AsyncTavilyClient

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


def collector(state: DeepAnalState): ...
def curator(state: DeepAnalState): ...
def briefer(state: DeepAnalState): ...
def editor(state: DeepAnalState): ...
