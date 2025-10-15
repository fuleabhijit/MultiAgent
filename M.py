from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.models.groq import Groq
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.tools.yfinance import YFinanceTools

import os
from dotenv import load_dotenv
load_dotenv()

_grok_key = os.getenv("GROQ_API_KEY")
if _grok_key:
    os.environ["GROQ_API_KEY"] = _grok_key
else:
    raise ValueError("GROQ_API_KEY is not set in environment or .env file")

web_agent=Agent(
    name="Web Agent",
    role="search the web for information",
    model=Groq(id="llama-3.3-70b-versatile"),
    tools=[DuckDuckGoTools()],
    instructions="""
Always include sources at the end.
Use markdown formatting for all output.
For facts and summaries, use bulleted or numbered lists.
Separate each section with a markdown heading (##).
Be concise and structured in your answers.
""",
    show_tool_calls=False,
    markdown=True,
)

finance_agent = Agent(
    name="Finance Agent",
    role="Get financial data",
    model=Groq(id="llama-3.3-70b-versatile"),
    tools=[YFinanceTools(stock_price=True, analyst_recommendations=True,stock_fundamentals=True,company_info=True)],
    instructions="""
Use markdown tables with clear headers for all financial data.
Always include sources at the end.
For analysis, use numbered or bulleted lists.
Separate each section with a markdown heading (##).
Be concise and structured in your answers.
""",
    show_tool_calls=False,
    markdown=True,
)

agent_team=Agent(
    team=[web_agent,finance_agent],
    model=Groq(id="llama-3.3-70b-versatile"),
    instructions=[
        "Always include sources at the end.",
        "For financial data, use markdown tables with clear headers.",
        "For analysis, use numbered or bulleted lists.",
        "Separate each section with a markdown heading (##).",
        "Be concise and structured in your answers.",
        "Use markdown formatting for all output."
    ],
    show_tool_calls=False,
    markdown=True,
)

##agent_team.print_response("Analyze companies like Tesla,Apple and suggest which to buy for long term")
##agent_team.print_response("Show me the stock price trend for Tesla over the past 6 months")
##agent_team.print_response("What do analysts say about investing in AI companies like Nvidia and Palantir, and how are they performing on the stock market?")
agent_team.print_response("Compare Apple and Microsoft in terms of stock fundamentals and public sentiment from the last month")