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

client = openai.OpenAI(api_key="anything", base_url="http://localhost:9000")


# TODO: add examples to langfuse dataset
# TODO: add a scoring function
# TODO: use the runs feature in langfuse to run eval using langfuse dataset with custom scoring function
# TODO: so far its all single round, one shot, try with multi-round and multiple agents (multiple llms) if we have time

# @observe()
# def non_openai_zero_shot_extractor(
#     image_path: Path, langfuse_prompt_name: str, model: str = None
# ):

#     prompt_obj = langfuse_client.get_prompt(langfuse_prompt_name)

#     config = deepcopy(prompt_obj.config)
#     json_schema = config.pop("json_schema")

#     if "model" not in config and model is None:
#         raise ValueError(
#             "Model name is required in the prompt config or as an argument"
#         )

#     if model:
#         config["model"] = model

#     messages = prompt_obj.compile(schema=json.dumps(json_schema))

#     base64_image = encode_image(image_path)

#     messages.append(
#         {
#             "role": "user",
#             "content": [
#                 {
#                     "type": "image_url",
#                     "image_url": {"url": f"data:image/png;base64,{base64_image}"},
#                 },
#             ],
#         }
#     )

#     chat_response = client.chat.completions.create(
#         langfuse_prompt=prompt_obj, messages=messages, **config
#     )

#     return chat_response.choices[0].message.content


@observe()
def non_openai_zero_shot_extractor(
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
    # new_messages = [messages[0], ]

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
    # messages.append(
    #     {
    #         "role": "user",
    #         "content": [
    #             {
    #                 "type": "image_url",
    #                 "image_url": {"url": f"data:image/png;base64,{base64_image}"},
    #             },
    #         ],
    #     }
    # )

    chat_response = client.chat.completions.create(
        langfuse_prompt=prompt_obj, messages=new_messages, **config
    )

    return chat_response.choices[0].message.content


if __name__ == "__main__":

    table_images_dir = REPO_DIR / "data"

    for image_path in table_images_dir.glob(f"*.png"):
        result = non_openai_zero_shot_extractor(image_path, "qwen_extractor_prompt")
        print(result)
