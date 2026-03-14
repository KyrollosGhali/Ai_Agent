import os
import base64
import time
from io import BytesIO
from PIL import Image
from dotenv import load_dotenv
from langchain_google_genai.chat_models import ChatGoogleGenerativeAIError

from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage

# Load .env before importing modules that may need environment variables.
load_dotenv()

from agent import create_agent, ImageGenState

GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError(
        "GOOGLE_API_KEY is not set. Add it to your .env file or environment variables."
    )

GEMINI_MODEL = os.getenv("GOOGLE_MODEL", "gemini-2.0-flash")

# -------------------------
# Create agent components
# -------------------------
assistant, extract_images, routing = create_agent(GEMINI_API_KEY, model=GEMINI_MODEL)

# -------------------------
# Build LangGraph workflow
# -------------------------
workflow = StateGraph(ImageGenState)

workflow.add_node("assistant", assistant)
workflow.add_node("extract_images", extract_images)

workflow.set_entry_point("assistant")

workflow.add_conditional_edges(
    "assistant",
    routing,
    {
        "tool": "assistant",
        "done": "extract_images"
    }
)

workflow.add_edge("extract_images", END)

app = workflow.compile()

# -------------------------
# Run pipeline
# -------------------------
def run(topic: str):

    print(f"\nTopic: {topic}\n")

    state = {
        "messages": [HumanMessage(content=topic)],
        "generation_output": []
    }

    max_retries = int(os.getenv("GEMINI_MAX_RETRIES", "0"))
    retry_wait_seconds = int(os.getenv("GEMINI_RETRY_WAIT_SECONDS", "30"))

    result = None
    for attempt in range(max_retries + 1):
        try:
            result = app.invoke(state)
            break
        except ChatGoogleGenerativeAIError as exc:
            error_text = str(exc)
            is_quota_error = "RESOURCE_EXHAUSTED" in error_text or "429" in error_text
            is_zero_quota = "limit: 0" in error_text

            if is_quota_error and not is_zero_quota and attempt < max_retries:
                print(
                    f"Gemini quota/rate limit reached. Retrying in {retry_wait_seconds} seconds..."
                )
                time.sleep(retry_wait_seconds)
                continue

            print("\nGemini API request failed.")
            if is_zero_quota:
                print(
                    "Your Gemini key/project currently has zero available quota (limit: 0). "
                    "Enable billing or use another API key/project with available quota."
                )
            elif is_quota_error:
                print(
                    "Quota/rate limit exceeded. Wait and retry, or reduce request frequency."
                )
            else:
                print("API error details:")
                print(error_text)
            return
        except Exception as exc:
            print("\nUnexpected error while running the workflow:")
            print(str(exc))
            return

    if result is None:
        print("No result returned from workflow.")
        return

    images = result.get("generation_output", [])

    if not images:
        print("No images generated.")
        return

    print(f"\nSaving {len(images)} images...\n")

    for i, img_base64 in enumerate(images):

        img_data = base64.b64decode(img_base64)
        image = Image.open(BytesIO(img_data))

        filename = f"scene_{i+1}.png"
        image.save(filename)

        print(f"Saved {filename}")

    print("\nDone!\n")


# -------------------------
# Entry
# -------------------------
if __name__ == "__main__":

    topic = "A futuristic city floating in the sky with flying cars"

    run(topic)