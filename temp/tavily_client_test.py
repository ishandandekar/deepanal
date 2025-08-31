import asyncio
import os

from catppuccin.extras.rich_ctp import mocha
from langchain.chat_models import init_chat_model
from langchain_core.callbacks import UsageMetadataCallbackHandler
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from rich.console import Console
from rich.progress import Progress
from tavily import AsyncTavilyClient, TavilyClient
from langchain.prompts import PromptTemplate

company = "Affinity Water"
industry = "Water utility and supply"

cns = Console(theme=mocha)
tavily_client = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])


class SearchQueries(BaseModel):
    queries: list[str] = Field(
        description="Search queries based on description", max_length=6, min_length=6
    )


gemini_llm = init_chat_model(
    "gemini-2.0-flash", model_provider="google_genai", temperature=0
)
ollama_llm = init_chat_model("ollama:deepseek-r1:7b", reasoning=False)
llm = gemini_llm.with_fallbacks([ollama_llm])
search_query_pydantic_parser = PydanticOutputParser(
    pydantic_object=SearchQueries)

usage_callback = UsageMetadataCallbackHandler()
query_generator_llm = gemini_llm.with_structured_output(SearchQueries)

query_generation_prompt = PromptTemplate.from_template("""
You are a senior risk analyst with 15 years of experience in corporate risk assessment across multiple industries. Your expertise lies in identifying potential financial, regulatory, operational, ESG, reputational, and geopolitical risks for companies.
You have been hired to research {company}, which operates in the {industry} sector. Your task is not to analyze the company yet, but to prepare search queries that will help you gather intelligence on its risk profile.
Your queries should be:
- Focused on uncovering risks and vulnerabilities, not just generic company info.
- Specific to the company and its industry context.
- Concise and actionable — something an analyst could directly search online.
- Cover a diverse set of risk categories.

When generating queries, ensure they span these areas, but frame them in terms of the industry:
- Regulatory & Legal – industry-specific laws, licenses, penalties.
- Financial & Market – exposure to industry price cycles, subsidies, demand shocks.
- Operational & Supply Chain – industry bottlenecks, resource constraints, logistics issues.
- Environmental, Social & Governance (ESG) – sustainability concerns tied to the sector, labor practices, certifications.
- Reputation & Controversies – industry-related scandals, NGO/activist reports, negative media.
- Geopolitical & Strategic – trade barriers, territorial disputes, political risks unique to the industry.
- Technology & Cyber – digital adoption, IT reliance, cyber vulnerabilities within the industry.

Generate only 6 highly relevant, industry-specific search queries that you, as a risk analyst, would run to investigate this company. Each query should reference both {company} and {industry} to make it focused and precise.
""")

cns.print(query_generation_prompt.config_specs)

query_generation_chain = query_generation_prompt | query_generator_llm
queries_from_llm = query_generation_chain.invoke(
    {"company": company, "industry": industry}, config={"callbacks": [usage_callback]}
)

cns.print(usage_callback.usage_metadata)

atavily_client = AsyncTavilyClient(api_key=os.environ["TAVILY_API_KEY"])


async def do_work(task_id, progress, query):
    res = await atavily_client.search(query, return_exception=True)
    progress.update(task_id, advance=1)
    return res


async def fetch_tavily_for_search_query():
    with Progress() as progress:
        task1 = progress.add_task("Fetching search results", total=6)

        results = await asyncio.gather(
            *[do_work(task1, progress, query) for query in queries_from_llm.queries]
        )
        print("\nAll tasks finished")
    return results


tavily_results = asyncio.run(fetch_tavily_for_search_query())
