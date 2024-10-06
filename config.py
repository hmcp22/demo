from utils.utils import encode_image
from pathlib import Path
from langfuse.openai import openai
from langfuse import Langfuse
import json
from copy import deepcopy
from langfuse.decorators import observe, langfuse_context
from utils.utils import get_git_repository_info
from dotenv import load_dotenv
from os import getenv

load_dotenv()

MODELS = [
    "gpt-4o",
    "qwen2-vl-7b",
    "pixtral-12b",
    "claude-3.5-sonnet",
    "gpt-4o-mini",
    "llama-3.2-11b-vision-preview",
]

langfuse_client = Langfuse(
    host=getenv("LANGFUSE_HOST"),
    secret_key=getenv("LANGFUSE_SECRET_KEY"),
    public_key=getenv("LANGFUSE_PUBLIC_KEY"),
)

REPO_DIR = Path(__file__).parent


AUTOGEN_ALL_LLMS_CONFIG = {"config_list":[]}

for model in MODELS:
    AUTOGEN_ALL_LLMS_CONFIG["config_list"].append(
        {
            "model": model,
            "api_key": "anything",
            "base_url": getenv("LITELLM_HOST"),
        }
    )
