from pydantic_ai import Agent
from pydantic_ai.models.gemini import GeminiModel

model = GeminiModel("gemini-2.0-flash", provider="google-gla")
agent = Agent(
    model,
    # tools=[tavily_search_tool(api_key=os.environ["TAVILY_API_KEY"])],
    # system_prompt="Search Tavily for the given query and return the results",
)
if __name__ == "__main__":
    from rich.console import Console
    from rich.markdown import Markdown

    answer = agent.run_sync("Hi!")

    cons = Console(log_path=False, log_time=False)
    cons.print(Markdown(answer.data))
