import autogen
from utils.autogen_langfuse import (
    LangfuseMultimodalConversableAgent,
    LangfuseConversableAgent,
)
import asyncio
from config import REPO_DIR, AUTOGEN_ALL_LLMS_CONFIG
from langfuse.decorators import observe, langfuse_context
from utils.utils import get_git_repository_info

@observe()
async def multiagent_extractor(image_path):


    git_info = get_git_repository_info(REPO_DIR)
    langfuse_context.update_current_trace(
        session_id="multiagent_extractor", version=git_info["commit_hash"]
    )

    # model = "qwen2-vl-7b"
    models = [
        "gpt-4o",
        # "qwen2-vl-7b",
        # "pixtral-12b",
        # "claude-3.5-sonnet",
        "gpt-4o-mini",
        # "llama-3.2-11b-vision-preview",
    ]

    user_proxy = LangfuseConversableAgent(
        name="user",
        llm_config=None,
        human_input_mode="NEVER",
        max_consecutive_auto_reply=2,
    )

    multimodal_agent_chats = []

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
            name=f"{model} agent",
            langfuse_prompt_name="autogen_extractor_system_prompt",
            llm_config={
                **model_config,
                "temperature": 0,
                "cache_seed": None,
            },
            human_input_mode="NEVER",
            max_consecutive_auto_reply=2,
        )

        multimodal_agent_chats.append(
            {
                "chat_id": model,
                "recipient": multimodal_agent,
                "langfuse_prompt_name": "autogen_extractor_message_prompt",
                "langfuse_prompt_args": {"image_path": image_path},
                "summary_method": "last_msg",
            }
        )


    chat_results = await user_proxy.a_initiate_chats(multimodal_agent_chats)


    summary = user_proxy.get_chat_results()

    return chat_results




if __name__ == "__main__":

    # image_path = REPO_DIR / "data" / "fidelity.png"
    from pathlib import Path

    for image_path in Path(REPO_DIR / "data").glob("*.png"):

        loop = asyncio.get_event_loop()

        chat_results = loop.run_until_complete(multiagent_extractor(image_path))

        print(chat_results)


    loop.close()

