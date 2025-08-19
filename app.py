from typing import TypedDict,Annotated,Optional
from langgraph.graph import StateGraph,END,add_messages
from dotenv import  load_dotenv,find_dotenv
from langchain_groq import ChatGroq
import os
from fastapi import FastAPI,Query
from langchain_tavily import TavilySearch
from  langgraph.checkpoint.memory import MemorySaver
from uuid import uuid4
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from langchain_core.messages import AIMessageChunk,AIMessage,SystemMessage,HumanMessage,ToolMessage
import json
import asyncio


load_dotenv()


GROQ_API_KEY=os.environ['GROQAPI_KEY']

llm=ChatGroq(
     model="llama3-70b-8192",
    groq_api_key=os.environ.get("GROQAPI_KEY")
)

api=FastAPI()

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



async def main():
    async for event in graph_builder.astream_events(
        {"messages": [system_prompt, HumanMessage(content="tell me about tesla optimus robot")]},
        config=config,
        version='v2'
    ):
      if event['event']=="on_chat_model_end":
          print(event['data']['output'].content)

if __name__ == "__main__":
    asyncio.run(main())



#FAST API IMPLEMENTATION
#CREATING CORS MIDDLEWARW TO 

api.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["Content-Type"]    

)

def serialize_ai_message_chunk(chunk):
    if(isinstance(chunk,AIMessageChunk)):
        return chunk.content
    else:
        raise TypeError(
            f"Object of type {type(chunk).__name__} is not correct formatted for serialization"
        )



async def generate_chat_responses(message:str,checkpoint_id:Optional[str]=None):
    is_new_connection=checkpoint_id is None

    if is_new_connection:
        #generating nwe checkpointer id for the message
        new_checkpoint_id=str(uuid4())

        config={
            "configurable":{
                "thread_id":new_checkpoint_id
            }
        }

        #initialize the first message
        events=graph_builder.astream_events(
            {
                "messages":[HumanMessage(content=message)]
            },
            version='v2',
            config=config
        )
        #first send the checkpoint ID
        yield f"data{{\"type\":\"checkpoint\",\"checkpoint_id\" : \"{new_checkpoint_id}\" }}\n\n"
    else:
        config={
            "configurable":{
                "thread_id":checkpoint_id
            }
        }    
        #continue existing the conversation

        events=graph_builder.astream_events(
            {"messages": [HumanMessage(content=message)]},
            version="v2",
            config=config
        )
    #executyion remaining
    



@api.get(".chat_stream/{message}")
async def chat_stream(message:str,checkpoint_id:Optional[str]=Query(None)):
   return StreamingResponse(
       generate_chat_responses(message,checkpoint_id),
       media_type="text/event-stream"
   )



