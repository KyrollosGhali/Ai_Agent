from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated, Sequence, Literal
from langchain_core.messages import SystemMessage, BaseMessage
from langgraph.prebuilt import ToolNode
from langgraph.graph.message import add_messages
import os
import replicate
from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_groq.chat_models import ChatGroq

# Load environment variables from .env before reading API keys.
load_dotenv()

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
token = os.getenv("REPLICATE_API_TOKEN")
if not token:
    raise RuntimeError(
        "REPLICATE_API_TOKEN is not set. Add it to your .env file or environment variables."
    )
client = replicate.Client(api_token=token)
@tool
def generate_image(prompt : str)-> str :
    """Generates an image based on the given prompt using the Google Imagen model."""
    payload = {
        "prompt": prompt,
        "aspect_ratio": "16:9",
        "safety_filter_level": "block_medium_and_above"
    }
    raw_output = client.run(
        "google/imagen-4",
        input=payload
    )
    outputs = raw_output if isinstance(raw_output, list) else [raw_output]

    first = outputs[0]
    if isinstance(first, dict):
        return first.get("url", "")
    if isinstance(first, str):
        return first

    # Replicate may return FileOutput objects with a `.url` attribute.
    file_url = getattr(first, "url", None)
    if isinstance(file_url, str) and file_url:
        return file_url

    # Fallback for FileOutput-like objects: stringify the output.
    value = str(first).strip()
    if value:
        return value

    raise RuntimeError("Image generation returned an unsupported output format.")
tools = [generate_image]
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise RuntimeError(
        "GROQ_API_KEY is not set. Add it to your .env file or environment variables."
    )
llm = ChatGroq(
    model="llama-3.1-8b-instant",
    api_key=api_key
).bind_tools(tools)


def model_call(state: AgentState) -> AgentState:
    prompt = """You are a helpful assistant that can generate images using the provided tool.
- When the user asks for an image, call the `generate_image` tool.
- Respond clearly with the image URL returned by the tool.
- If the user asks for something unrelated, explain that you can help with image generation.
- if the user wants to stop , return a message saying "Goodbye!" and do not call any tools.
"""
    system_message = SystemMessage(content=prompt)
    response = llm.invoke([system_message, *state["messages"]])
    print(f"model response: {response.content}")
    return {"messages": [response]}


def should_continue(state: AgentState) -> Literal["continue", "end"]:
    if state["messages"][-1].tool_calls:
        return "continue"
    return "end"

tools_node = ToolNode(tools=tools)
graph = StateGraph(AgentState)
graph.add_node("model_call", model_call)
graph.add_node("tools_node", tools_node)
graph.add_edge(START, "model_call")
graph.add_conditional_edges(
    "model_call",
    should_continue,
    {
        "continue": "tools_node",
        "end": END,
    },
)
graph.add_edge("tools_node", "model_call")

app = graph.compile()