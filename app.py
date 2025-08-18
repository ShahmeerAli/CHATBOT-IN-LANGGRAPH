from typing import TypedDict,Annotated,Optional
from langgraph.graph import StateGraph,END,add_messages
from dotenv import  load_dotenv,find_dotenv
from langchain_groq import ChatGroq
import os
from langchain_community.tools.tavily_search import TavilySearchResults
from  langgraph.checkpoint.memory import MemorySaver
from uuid import uuid4
from langchain_core.messages import AIMessage,HumanMessage,ToolMessage
import json


load_dotenv()


GROQ_API_KEY=os.environ['GROQAPI_KEY']

llm=ChatGroq(
     model="llama3-70b-8192",
    groq_api_key=os.environ.get("GROQAPI_KEY")
)



search_tool=TavilySearchResults()



tools=[search_tool]

checkpointer=MemorySaver()

llm_eith_tools=llm.bind_tools(tools=tools)


class AgentState(TypedDict):
    messages:Annotated[list,add_messages]


    