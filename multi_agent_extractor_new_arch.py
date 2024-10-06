import autogen
import os
from pathlib import Path
from utils.autogen_langfuse import (
    LangfuseMultimodalConversableAgent,
    LangfuseConversableAgent,
)
import asyncio
from config import REPO_DIR, AUTOGEN_ALL_LLMS_CONFIG
from langfuse.decorators import observe, langfuse_context
from dotenv import load_dotenv
from utils.utils import get_git_repository_info

load_dotenv()


from typing import Annotated, Literal

Operator = Literal["+", "-", "*", "/"]


def calculator(a: int, b: int, operator: Annotated[Operator, "operator"]) -> int:
    if operator == "+":
        return a + b
    elif operator == "-":
        return a - b
    elif operator == "*":
        return a * b
    elif operator == "/":
        return int(a / b)
    else:
        raise ValueError("Invalid operator")


def is_termination_msg(msg):

    if isinstance(msg, list):
        msg_content = msg[-1].get("content", "")
    else:
        msg_content = msg.get("content", "")
    if msg_content:
        return "TERMINATE" in msg.get("content", "")
    return False


@observe()
async def multiagent_extractor_new(image_path):

    git_info = get_git_repository_info(REPO_DIR)
    langfuse_context.update_current_trace(
        session_id="multiagent_extractor_new", version=git_info["commit_hash"]
    )

    models = [
        "gpt-4o",
        # "qwen2-vl-7b",
        # "pixtral-12b",
        # "claude-3.5-sonnet",
        "gpt-4o-mini",
        # "llama-3.2-11b-vision-preview",
    ]

    verifier_model = "gpt-4o"

    verifier_agent = LangfuseConversableAgent(
        name=f"verifier_{verifier_model}",
        human_input_mode="NEVER",
        max_consecutive_auto_reply=10,
        # llm_config=None,
        llm_config={
            **{
                "config_list": autogen.filter_config(
                    AUTOGEN_ALL_LLMS_CONFIG["config_list"],
                    filter_dict={
                        "model": [verifier_model],
                    },
                )
            },
            "temperature": 0,
            "cache_seed": None,
        },
        is_termination_msg=is_termination_msg,
        langfuse_prompt_name="verifier_system_prompt",
    )

    # Register the tool signature with the assistant agent.
    verifier_agent.register_for_llm(
        name="calculator", description="A simple calculator"
    )(calculator)

    verifier_multimodel_agent_chats = []

    for model in models:

        model_config = {
            "config_list": autogen.filter_config(
                AUTOGEN_ALL_LLMS_CONFIG["config_list"],
                filter_dict={
                    "model": [model],
                },
            )
        }

        multimodal_agent = LangfuseMultimodalConversableAgent(
            name=f"{model}_agent",
            langfuse_prompt_name="autogen_extractor_system_prompt",
            llm_config={
                **model_config,
                "temperature": 0,
                "cache_seed": None,
            },
            human_input_mode="NEVER",
            max_consecutive_auto_reply=10,
            is_termination_msg=is_termination_msg,
        )

        # Register the tool function with the user proxy agent.
        multimodal_agent.register_for_execution(name="calculator")(calculator)

        verifier_multimodel_agent_chats.append(
            {
                "chat_id": model,
                "recipient": multimodal_agent,
                "langfuse_prompt_name": "autogen_extractor_message_prompt",
                "langfuse_prompt_args": {"image_path": str(image_path)},
                "summary_method": "last_msg",
            }
        )

    chat_results = await verifier_agent.a_initiate_chats(
        verifier_multimodel_agent_chats
    )

    summary = verifier_agent.get_chat_results()

    return chat_results


if __name__ == "__main__":

    image_path = REPO_DIR / "data" / "abc.png"

    loop = asyncio.get_event_loop()

    chat_results = loop.run_until_complete(multiagent_extractor_new(image_path))

    loop.close()

    print(chat_results)
