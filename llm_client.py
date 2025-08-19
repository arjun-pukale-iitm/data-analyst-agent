import os
from dotenv import load_dotenv
import google.generativeai as genai
from google.generativeai.types import content_types
from typing import List, Dict, Any


load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY is missing in the .env file")

# Configure the Gemini client
genai.configure(api_key=GEMINI_API_KEY)
#model = genai.GenerativeModel("gemini-2.5-flash-lite") # fast but poor results (good for testing)
model = genai.GenerativeModel("gemini-2.5-flash") # very good results. but a bit slower
#model = genai.GenerativeModel("gemini-2.5-pro")

def call_llm(messages: list, attachments: List[Dict[str, Any]] = None) -> str:
    """
    Call Gemini model with a conversation-style message list.
    messages: [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Tell me a joke."}
    ]
    """
    parts = []
    # Gemini does not have role awareness like OpenAI; we format the prompt ourselves
    prompt_text = "\n".join([f"{m['role'].capitalize()}: {m['content']}" for m in messages])
    parts.append({"text": prompt_text})
    # Add image attachments if present
    if attachments:
        for att in attachments:
            if att.get("is_image"):
                # Add image as inline input
                parts.append(
                    {
                        "inline_data": {
                            "mime_type": att["content_type"],
                            "data": att["content_bytes"]
                        }
                    }
                )


    response = model.generate_content(parts)
    return response.text.strip()
