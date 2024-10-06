from utils.utils import encode_image
from pathlib import Path
import json
from copy import deepcopy
from langfuse.decorators import observe, langfuse_context
from utils.utils import get_git_repository_info

from config import langfuse_client, REPO_DIR
import openai
from os import getenv

OPENAI_CLIENT = openai.OpenAI(api_key="anything", base_url=getenv("LITELLM_HOST"))


@observe()
def openai_single_round_extractor_with_structured_outputs(
    image_path: Path, langfuse_prompt_name: str, model: str
):

    git_info = get_git_repository_info(REPO_DIR)
    langfuse_context.update_current_trace(version=git_info["commit_hash"])

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

    @observe(as_type="generation")
    def generate_openai_structured_output(model, messages, json_schema, **config):
        return OPENAI_CLIENT.beta.chat.completions.parse(
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

    chat_response = generate_openai_structured_output(
        model=model, messages=messages, json_schema=json_schema, **config
    )
    return chat_response.choices[0].message.content


@observe()
def non_openai_single_round_extractor(
    image_path: Path, langfuse_prompt_name: str, model: str = None
):
    git_info = get_git_repository_info(REPO_DIR)
    langfuse_context.update_current_trace(version=git_info["commit_hash"])

    prompt_obj = langfuse_client.get_prompt(langfuse_prompt_name)

    config = deepcopy(prompt_obj.config)
    json_schema = config.pop("json_schema")

    if "model" not in config and model is None:
        raise ValueError(
            "Model name is required in the prompt config or as an argument"
        )

    if model:
        config["model"] = model

    messages = prompt_obj.compile(schema=json.dumps(json_schema))

    base64_image = encode_image(image_path)

    new_messages = [
        messages[0],
        {
            "role": "user",
            "content": [
                {"type": "text", "text": messages[1]["content"]},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{base64_image}"},
                },
            ],
        },
    ]

    chat_response = OPENAI_CLIENT.chat.completions.create(
        langfuse_prompt=prompt_obj, messages=new_messages, **config
    )

    return chat_response.choices[0].message.content


if __name__ == "__main__":

    # from utils.utils import create_openai_extractor_langfuse_prompt, create_qwen_extractor_langfuse_prompt
    # create_qwen_extractor_langfuse_prompt()
    # create_openai_extractor_langfuse_prompt()

    table_images_dir = REPO_DIR / "data"

    for image_path in table_images_dir.glob(f"*.png"):
        result = non_openai_single_round_extractor(
            image_path, "qwen_extractor_prompt", model="qwen2-vl-7b"
        )
        print(result)
