# DeepAnal
**DeepAnal** is AI-powered company research tool. "Tool" because the idea is to help you in work, but not do all of it.  
It does analysis for company fundamentals, its place in the industry, its finances, recent news and also a controversy-risk analysis.  
I know there are a lot of things like DeepAnal in the market, but this is my attempt to learn AI-agents thoroughly, nothing special, this is a purely a learning project  

### Working
**DeepAnal** employs [Tavily](https://www.tavily.com/) to run search queries for each vertical of a company. Based on these documents, collates the information and then presents it in a nicely formatted report.  

### Motivation
I saw [Company Research](https://companyresearcher.tavily.com/), and thought it's a very simple and easy to use tool to get you so much information about a company, where you go 0 to 100, in very minimal effort. Gave myself a challenge to go through the source code and recreate it.  
Also, yeah I mean I was itching to use LangGraph

### Usage
```bash

❯ uv run ./deepanal.py \
-c "Marsh McLennan" \
-l "United States of America" \
-i "Risk consulting and insurance brokerage"

██████╗ ███████╗███████╗██████╗  █████╗ ███╗   ██╗ █████╗ ██╗
██╔══██╗██╔════╝██╔════╝██╔══██╗██╔══██╗████╗  ██║██╔══██╗██║
██║  ██║█████╗  █████╗  ██████╔╝███████║██╔██╗ ██║███████║██║
██║  ██║██╔══╝  ██╔══╝  ██╔═══╝ ██╔══██║██║╚██╗██║██╔══██║██║
██████╔╝███████╗███████╗██║     ██║  ██║██║ ╚████║██║  ██║███████╗
╚═════╝ ╚══════╝╚══════╝╚═╝     ╚═╝  ╚═╝╚═╝  ╚═══╝╚═╝  ╚═╝╚══════╝
```
