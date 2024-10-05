from utils.utils import encode_image
from pathlib import Path
from langfuse.openai import openai
from langfuse import Langfuse
import json
from copy import deepcopy
from langfuse.decorators import observe, langfuse_context
from utils.utils import get_git_repository_info

langfuse_client = Langfuse()

REPO_DIR = Path(__file__).parent

OPENAI_CLIENT = openai.OpenAI(api_key="anything", base_url="http://localhost:9000")