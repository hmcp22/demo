from utils.utils import encode_image
from pathlib import Path
from langfuse.openai import openai
from langfuse import Langfuse
import json
from copy import deepcopy
from langfuse.decorators import observe

langfuse_client = Langfuse()

REPO_DIR = Path(__file__).parent

client = openai.OpenAI(api_key="anything", base_url="http://localhost:9000")

@observe()
def openai_zero_shot_extractor(
    image_path: Path, langfuse_prompt_name: str, model: str = None
):

    prompt_obj = langfuse_client.get_prompt(langfuse_prompt_name)

    config = deepcopy(prompt_obj.config)

    json_schema = config.pop("json_schema")

    system_prompt = prompt_obj.compile()

    base64_image = encode_image(image_path)

    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{base64_image}"},
                },
            ],
        },
    ]

    chat_response = client.beta.chat.completions.parse(
        model=model,
        messages=messages,
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "change_in_account_value",
                "schema": json_schema,
                # "strict": True,
            },
        },
        **config,
    )

    return chat_response.choices[0].message.content


if __name__ == "__main__":

    table_images_dir = REPO_DIR / "data"

    for image_path in table_images_dir.glob(f"*.png"):
        result = openai_zero_shot_extractor(
            image_path, "extractor_system_prompt", "gpt-4o"
        )
        print(result)
