**DeepAnal** is AI-powered company research tool. "Tool" because the idea is to help you in work, but not do all of it. I know there are a lot of things like DeepAnal in the market, but this is my attempt to learn AI-agents thoroughly, nothing special this is a purely a learning project
Very nascent stage. Like there's literally no code.
Any help is appreciated, you can see the TODOs below, just pick anything. There will be TODOs scattered around in the code files as well, please have a look through and contribute

TODOs:
- [ ] Output should be in markdown, do not print the entire report in terminal
- [ ] Extensive logging using logger, don't use print statements
- [ ] Agent State should have fields like these
      ```python
      class DeepAnalState(TypedDict):
        company_name:str
        location:str
        industry:str
        company_url: Optional[str]
        llm: LangChainLLM
        tavily_client: TavilyClient
        logger: Logger
        foo_analyst_message: dict
      ```
- [ ] `.envrc` to store keys
- [ ] As an extension, make a _risk_analyst_, for regulatory and legal risk, reputation, ESG risk, operational
- [ ] KISS
- [ ] Have a clear balance between feature and fancy
- [ ] Don't be afraid to use Gemini, as a fallback have Ollama
- [ ] There should be statistics post usage, such as time taken, token usage, URLs searched
- [ ] Use LangGraph more for graph based usage and not as such for LLM stuff, I mean they've already designed it that way
- [ ] Usage should only be via CLI args, no "task file"s. No `thread-id` idgaf if you get an error
- [ ] Usage will only support one company at a time
- [ ] Be super-verbose, log everything
- [ ] Don't use `pandas` or anything tabular data library, use simple data-models
- [ ] Functions over classes, keep each node as function only and have a try-except within it 
