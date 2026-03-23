# Image Generation Agent (LangGraph + Groq + Replicate)

This project is a Streamlit app that lets you type an image prompt and generate an image URL using an AI agent workflow.

The workflow is built with LangGraph:
- A chat model (Groq) decides when to call a tool.
- A tool function calls Replicate (Google Imagen) to generate the image.
- The app shows the model response and renders the generated image in the UI.

## Features

- Streamlit web interface for prompt input and result display
- Agent orchestration with LangGraph state machine
- Tool-calling via LangChain tools
- Image generation through Replicate model `google/imagen-4`
- Automatic image URL detection and preview in the app

## Project Structure

- `main.py`: Streamlit UI, prompt handling, app invocation, URL extraction, image display
- `agent.py`: Agent graph, model/tool wiring, environment loading, image generation tool
- `requirements.txt`: Python dependencies
- `.env.example`: Required environment variable names

## Requirements

- Python 3.10+ recommended
- A Groq API key
- A Replicate API token

## Environment Variables

Create a `.env` file in the project root (or copy from `.env.example`) and set:

- `GROQ_API_KEY=your_groq_api_key`
- `REPLICATE_API_TOKEN=your_replicate_api_token`

Notes:
- `agent.py` will raise an error at startup if either value is missing.
- Make sure there are no extra spaces around variable names.

## Installation

1. Create and activate a virtual environment (recommended).
2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Run The App

```bash
streamlit run main.py
```

Then open the local Streamlit URL shown in your terminal (usually `http://localhost:8501`).

## How It Works

1. You enter a text prompt in the Streamlit interface.
2. The app sends your message to the LangGraph app (`app.invoke(...)`).
3. The model decides to call `generate_image`.
4. `generate_image` calls Replicate with:
   - `prompt`: your text
   - `aspect_ratio`: `16:9`
   - `safety_filter_level`: `block_medium_and_above`
5. The generated URL is returned to the model response.
6. The UI extracts URLs from the response and displays the image.

## Troubleshooting

- Missing API keys:
  - Error about `GROQ_API_KEY` or `REPLICATE_API_TOKEN` means your `.env` file is missing or not configured.
- Dependency issues:
  - Reinstall dependencies with `pip install -r requirements.txt`.
- No image displayed:
  - Check if the model response contains a valid URL.
  - Confirm your Replicate token has permission and quota.

## Notes

- The default prompt in `main.py` is only a starter example and can be changed.
- The current agent is focused on image-generation-related tasks.