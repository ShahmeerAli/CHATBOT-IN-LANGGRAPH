from typing import TypedDict,Annotated,Optional
from langgraph.graph import StateGraph,END,add_messages
from dotenv import  load_dotenv,find_dotenv
from langchain_groq import ChatGroq
import os
from langchain_tavily import TavilySearch
from  langgraph.checkpoint.memory import MemorySaver
from uuid import uuid4
from langchain_core.messages import AIMessage,HumanMessage,ToolMessage
import json
import asyncio


load_dotenv()


GROQ_API_KEY=os.environ['GROQAPI_KEY']

llm=ChatGroq(
     model="llama3-70b-8192",
    groq_api_key=os.environ.get("GROQAPI_KEY")
)



search_tool=TavilySearch()



tools=[search_tool]

memoy=MemorySaver()

llm_with_tools=llm.bind_tools(tools=tools)


class AgentState(TypedDict):
    messages:Annotated[list,add_messages]


#first node
async def model(state:AgentState):
    result=await llm_with_tools.ainvoke(state["messages"])
    return {
        "messages":result

    }



#tool router
async def tools_router(state:AgentState):
    last_message=state["messages"][-1]

    if(hasattr(last_message,"tool_calls")and len(last_message.tool_calls)>0):
        return "tool_node"
    else:
        return END
    

#node that handles tool calls from LLMs
async def tool_node(state):
    """Custom tool node that handles tool calls from the LLM."""
    #Get the tool calls from the last message
    tool_calling=state["messages"][-1].tool_calls
    tool_messages=[]
    for tool_call in tool_calling:
        tool_name=tool_call['name']
        tool_args=tool_call['args']
        tool_id=tool_call["id"]

        #handle the search toool
        if tool_name=="tavily_search_results_json":
            #execute the search tool with the provided aruguments
            search_results=await search_tool.ainvoke(tool_args)
            tool_message=ToolMessage(
                content=str(search_results),
                tool_call_id=tool_id,
                name=tool_name
            ) 
            tool_messages.append(tool_message)
    return {"messages": tool_messages}


system_prompt = HumanMessage(content=
    "You are a helpful assistant. "
    "Use the search tool *only if the user asks about factual, external knowledge* "
    "that you are not confident about. "
    "For casual conversation (like greetings, introductions, chit-chat), "
    "do NOT use any tools."
)

graph=StateGraph(AgentState)

graph.add_node("model",model)
graph.add_node("tool_node",tool_node)

graph.set_entry_point("model")
graph.add_conditional_edges("model",tools_router)
graph.add_edge("tool_node","model")



config={
    "configurable":{
        "thread_id":3
    }
}

graph_builder=graph.compile(checkpointer=memoy)

async def run():
   response=await graph_builder.ainvoke({
    "messages":[system_prompt,HumanMessage(content="What is my name?")],
   },config=config)

   print(response["messages"][-1].content)


asyncio.run(run())   
