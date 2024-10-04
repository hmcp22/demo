import base64
from langfuse import Langfuse
from utils.schema import ChangeInAccountValue
from git import Repo
from pathlib import Path
from typing import Union, Dict, Any


langfuse_client = Langfuse()


# Function to encode the image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def create_langfuse_text_prompt(
    prompt_name,
    json_schema,
    prompt,
    labels: list = ["production"],
    **model_kwargs,
):

    config = {"json_schema": json_schema}
    if model_kwargs:
        config["model_kwargs"] = model_kwargs

    langfuse_client.create_prompt(
        name=prompt_name,
        prompt=prompt,
        labels=labels,
        config={
            "json_schema": json_schema,
            **model_kwargs,
        },
    )


def create_langfuse_chat_prompt(
    prompt_name,
    json_schema,
    messages,
    labels: list = ["production"],
    **model_kwargs,
):

    config = {"json_schema": json_schema}
    if model_kwargs:
        config["model_kwargs"] = model_kwargs

    langfuse_client.create_prompt(
        name=prompt_name,
        prompt=messages,
        labels=labels,
        config={
            "json_schema": json_schema,
            **model_kwargs,
        },
        type="chat",
    )


def create_qwen_extractor_langfuse_prompt():

    qwen_system_prompt = """"
    You are an assistant in charge of looking at brokerage account statements for your clients.
    You will be provided with an image of the account statement. Please carefully look through the provided account statement and extract the relevant information.
    You should follow the format provided in the schema. If a field is not present in the statement, you can leave it as null.
    """

    qwen_message_prompt = """Here is the account statement image. Please extract the following information: {{schema}}"""

    messages = [
        {"role": "system", "content": qwen_system_prompt},
        {
            "role": "user",
            "content": qwen_message_prompt,
        },
    ]

    json_schema = ChangeInAccountValue.model_json_schema()

    create_langfuse_chat_prompt(
        "qwen_extractor_prompt", json_schema, messages, model="qwen2-vl-7b"
    )


def create_openai_extractor_langfuse_prompt():

    system_prompt = """
    You are an assistant in charge of looking at brokerage account statements for your clients.
    You will be provided with an image of the account statement. Please carefully look through the provided account statement and extract the relevant information.
    If a field is not present in the statement, you can leave it as null.
    """

    create_langfuse_text_prompt(
        "extractor_system_prompt",
        ChangeInAccountValue.model_json_schema(),
        system_prompt,
    )

def get_git_repository_info(
    repo_path: Union[Path, str],
) -> Dict[str, Any]:
    """
    Returns a dict with git repository info for the git repository specified
    by repo_path.

    Args:
        repo_path (Union[Path,str]): Path to a directory inside of a git
            repository.

    Returns:
        Dict[str, Any]: Returns dict with commit_hash, branch_name, remote_url,
        git_diff and untracked_files list.
    """

    repo = Repo(path=repo_path, search_parent_directories=True)
    return {
        "commit_hash": repo.head.commit.hexsha,
        "branch_name": repo.active_branch.name,
        "remote_url": repo.remotes[0].config_reader.get("url"),
        "git_diff": repo.git.diff(),
        "untracked_files": repo.untracked_files,
    }


if __name__ == "__main__":

    # create_qwen_extractor_langfuse_prompt()
    create_openai_extractor_langfuse_prompt()
