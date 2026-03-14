# agent.py
from typing import List, Optional, TypedDict
from langchain_core.messages import (
    SystemMessage,
    ToolMessage,
    HumanMessage,
    AnyMessage
)
from langgraph.graph import add_messages
from langchain_google_genai import ChatGoogleGenerativeAI  # Gemini API client
from langchain_core.tools import Tool
import base64
from io import BytesIO
from PIL import Image
# -------------------------
# Conversation / Agent State
# -------------------------
class ImageGenState(TypedDict):
    generation_output: Optional[List[str]]  # base64 strings of images
    messages: Optional[List[AnyMessage]]

# -------------------------
# Gemini Image Generator Tool
# -------------------------
class GeminiImageGenerator:
    def __init__(self, api_key: Optional[str] = None, model: str = "gemini-2.0-flash"):
        self.api_key = api_key
        self.client = ChatGoogleGenerativeAI(api_key=api_key, model=model)

    def generate_image(self, prompt: str, width=512, height=512) -> str:
        """Call Gemini AI to generate an image from prompt."""
        result = self.client.generate_image(prompt=prompt, width=width, height=height)
        # Gemini typically returns base64-encoded images
        img_base64 = result.image_base64
        return img_base64

# -------------------------
# Agent Creation
# -------------------------
def create_agent(api_key: Optional[str] = None, model: str = "gemini-2.0-flash"):
    image_generator = GeminiImageGenerator(api_key=api_key, model=model)

    # Wrap as a callable tool for LangGraph
    def generate_image_tool(prompt: str) -> str:
        print(f"Generating image for prompt: {prompt}")
        return image_generator.generate_image(prompt)

    # System prompt instructing LLM to generate story and images
    system_prompt = """
    You are a friendly assistant that generates images based on user requests.
    First, generate a story and extract 3-5 key scenes.
    Then, for each scene, call the "generate_image_tool".
    """

    # Gemini LLM
    llm_agent = ChatGoogleGenerativeAI(
        api_key=api_key,
        model=model,
        temperature=0.0,
    )
    llm_agent = llm_agent.bind_tools([generate_image_tool])

    # -------------------------
    # Assistant Function
    # -------------------------
    def assistant(state: ImageGenState) -> ImageGenState:
        messages = state.get("messages", [])
        if not messages or (messages[-1].name != "generate_image_tool" and len(messages) == 1):
            messages = add_messages(
                [SystemMessage(content=system_prompt)],
                [HumanMessage(content=messages[-1].content if messages else "Generate a story")]
            )

        response = llm_agent.invoke(messages)
        response.pretty_print()

        state["messages"] = add_messages(messages, [response])
        return state

    def extract_images(state: ImageGenState) -> ImageGenState:
        generation_output = state.get("generation_output", [])
        if generation_output is None:
            generation_output = []

        for msg in state.get("messages", []):
            if isinstance(msg, ToolMessage) and msg.name == "generate_image_tool":
                generation_output.append(msg.content)
                print(f"Image generated and added to state.")

        state["generation_output"] = generation_output
        return state

    # -------------------------
    # Routing function
    # -------------------------
    def routing(state: ImageGenState) -> str:
        ai_message = state["messages"][-1]
        if hasattr(ai_message, "tool_calls") and len(ai_message.tool_calls) > 0:
            return "tool"
        return "done"

    return assistant, extract_images, routing