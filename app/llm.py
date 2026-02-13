import os
from dotenv import load_dotenv

from google import genai


# Load environment and configure Gemini client/models in one place
load_dotenv()

GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash-001")
GEMINI_EMBED_MODEL = os.getenv("GEMINI_EMBED_MODEL", "gemini-embedding-001")

client = genai.Client()

