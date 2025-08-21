from typing import TypedDict,Annotated,Optional
from langgraph.graph import StateGraph,END,add_messages
from dotenv import  load_dotenv,find_dotenv
from langchain_groq import ChatGroq
import os
from fastapi import FastAPI,Query
from langchain_community.tools.tavily_search import TavilySearchResults
from  langgraph.checkpoint.memory import MemorySaver
from uuid import uuid4
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from langchain_core.messages import AIMessageChunk,AIMessage,SystemMessage,HumanMessage,ToolMessage
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
import aiosqlite 
import json
import sqlite3
import asyncio


load_dotenv()


#added checkpoints in the form of sqlite
GROQ_API_KEY=os.environ['GROQAPI_KEY']

llm=ChatGroq(
     model="llama3-70b-8192",
    groq_api_key=os.environ.get("GROQAPI_KEY")
)

api=FastAPI()

search_tool=TavilySearchResults()



tools=[search_tool]

sql_connection=aiosqlite.connect("memory.sqlite",check_same_thread=False)

memory=AsyncSqliteSaver(conn=sql_connection)


llm_with_tools=llm.bind_tools(tools=tools)


class AgentState(TypedDict):
    messages:Annotated[list,add_messages]



#first node
async def model(state:AgentState):
    result=await llm_with_tools.ainvoke(state["messages"])
    return {
        "messages":[result]

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


system_prompt =SystemMessage(content=
    "You are a helpful assistant. "
    "Use the search tool *only if the user asks about factual, external knowledge* "
    "that you are not confident about. "
    "For casual conversation (like greetings, introductions, chit-chat), "
    "do NOT use any tools."
    "for user details go to the memory file.If the user tells his/her name remember it"
)

graph=StateGraph(AgentState)

graph.add_node("model",model)
graph.add_node("tool_node",tool_node)

graph.set_entry_point("model")
graph.add_conditional_edges("model",tools_router)
graph.add_edge("tool_node","model")




graph_builder=graph.compile(checkpointer=memory)



api.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["Content-Type"]    

)

def serialize_ai_message_chunk(chunk):
    if isinstance(chunk, AIMessageChunk):
      
        if chunk.content:
            return chunk.content
        
        if hasattr(chunk, "delta") and chunk.delta:
            if isinstance(chunk.delta, list):
                return "".join(
                    d.get("content", "") for d in chunk.delta if isinstance(d, dict)
                )
            elif isinstance(chunk.delta, dict):
                return chunk.delta.get("content", "")
        
        return ""  
    return str(chunk)



async def generate_chat_responses(message: str, checkpoint_id: Optional[str] = None):
   
    config = {"configurable": {"thread_id": checkpoint_id}}

    if checkpoint_id is None:
    
        new_checkpoint_id = str(uuid4())
        config["configurable"]["thread_id"] = new_checkpoint_id
        
        
        inputs = {"messages": [system_prompt, HumanMessage(content=message)]}
        yield f"data: {{\"type\":\"checkpoint\",\"checkpoint_id\":\"{new_checkpoint_id}\"}}\n\n"
    else:
    
        inputs = {"messages": [HumanMessage(content=message)]}

    events = graph_builder.astream_events(
        inputs,
        config=config,
        version="v2"
    )

    async for event in events:
        event_type = event['event']

        if event_type == "on_chat_model_stream":
            chunk_content = serialize_ai_message_chunk(event['data']['chunk'])
            if not chunk_content:
                continue
            safe_content = json.dumps(chunk_content)[1:-1]
            yield f"data: {{\"type\": \"content\", \"content\": \"{safe_content}\"}}\n\n"

        elif event_type == "on_chat_model_end":
            tool_calls = getattr(event['data']['output'], "tool_calls", [])
            search_calls = [call for call in tool_calls if call['name'] == "tavily_search_results_json"]

            if search_calls:
                search_query = search_calls[0]["args"].get("query", "")
                safe_query = json.dumps(search_query)[1:-1]
                yield f"data: {{\"type\": \"search_start\", \"query\": \"{safe_query}\"}}\n\n"

        elif event_type == "on_tool_end" and event['name'] == "tavily_search_results_json":
            output = event['data']['output']
            results = []
            if isinstance(output, list):
                for item in output:
                    if isinstance(item, dict):
                        results.append({
                            "title": item.get("title", ""),
                            "url": item.get("url", ""),
                            "snippet": item.get("content", "")
                        })
            yield f"data: {{\"type\": \"search_results\", \"results\": {json.dumps(results)}}}\n\n"

    yield f"data: {{\"type\": \"end\"}}\n\n"


@api.get("/chat_stream/{message}")
async def chat_stream(message:str,checkpoint_id:Optional[str]=Query(None)):
   return StreamingResponse(
       generate_chat_responses(message,checkpoint_id),
       media_type="text/event-stream"
   )

