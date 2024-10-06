from typing import Any, Callable, Dict, List, Literal, Optional, Union
from autogen.agentchat.chat import ChatResult
from autogen.cache.cache import Cache
from autogen import ConversableAgent, Agent
from autogen.agentchat.contrib.multimodal_conversable_agent import (
    MultimodalConversableAgent,
)
from langfuse.decorators import observe, langfuse_context
import langfuse


class LangfuseConversableAgent(ConversableAgent):
    """
    A ConversableAgent with Langfuse custom logging.
    """

    @observe(as_type="generation")
    def __init__(
        self,
        name: str,
        langfuse_prompt_name: str | None = None,
        langfuse_prompt_args: Dict[str, Any] = None,
        is_termination_msg: Callable[[Dict], bool] | None = None,
        max_consecutive_auto_reply: int | None = None,
        human_input_mode: (
            Literal["ALWAYS"] | Literal["NEVER"] | Literal["TERMINATE"]
        ) = "TERMINATE",
        function_map: Dict[str, Callable[..., Any]] | None = None,
        code_execution_config: Dict | Literal[False] = False,
        llm_config: Dict | None | Literal[False] = None,
        default_auto_reply: str | Dict = "",
        description: str | None = None,
    ):

        self.langfuse_client = langfuse.Langfuse()

        system_message = None
        if langfuse_prompt_name:
            # the langfuse_client is used in this call
            system_message = self.log_langfuse_prompt(
                langfuse_prompt_name, langfuse_prompt_args, name
            )
        else:
            langfuse_context.update_current_observation(name=name)

        super().__init__(
            name=name,
            system_message=system_message,
            is_termination_msg=is_termination_msg,
            max_consecutive_auto_reply=max_consecutive_auto_reply,
            human_input_mode=human_input_mode,
            function_map=function_map,
            code_execution_config=code_execution_config,
            llm_config=llm_config,
            default_auto_reply=default_auto_reply,
            description=description,
        )

    def log_langfuse_prompt(
        self,
        prompt_name: str,
        prompt_args: Dict[str, Any] = None,
        obervation_name: str = None,
    ):
        
        prompt_obj = self.langfuse_client.get_prompt(prompt_name)

        p_args = {}
        if "json_schema" in prompt_obj.config:
            p_args = {"schema": prompt_obj.config["json_schema"]}
        
        if prompt_args:
            p_args.update(prompt_args)

        prompt = prompt_obj.compile(**p_args)
        
        langfuse_context.update_current_observation(
            prompt=prompt_obj, output=prompt, name=obervation_name
        )

        return prompt

    @observe(as_type="generation")
    def initiate_chat(
        self,
        recipient: "ConversableAgent",
        clear_history: bool = True,
        silent: Optional[bool] = False,
        cache: Optional[Cache] = None,
        max_turns: Optional[int] = None,
        summary_method: Optional[Union[str, Callable]] = None,
        summary_args: Optional[dict] = {},
        message: Optional[
            Union[
                Dict,
                str,
                Callable,
            ]
        ] = None,
        langfuse_prompt_name: str = None,
        langfuse_prompt_args: Dict[str, Any] = None,
        **context,
    ) -> ChatResult:

        message_to_send = message

        if langfuse_prompt_name:
            message_to_send = self.log_langfuse_prompt(
                langfuse_prompt_name,
                langfuse_prompt_args,
                f"{self.name} --> {recipient.name}",
            )

        return super().initiate_chat(
            recipient=recipient,
            clear_history=clear_history,
            silent=silent,
            cache=cache,
            max_turns=max_turns,
            summary_method=summary_method,
            summary_args=summary_args,
            message=message_to_send,
            **context,
        )

    @observe(as_type="generation")
    def a_initiate_chat(
        self,
        recipient: "ConversableAgent",
        clear_history: bool = True,
        silent: Optional[bool] = False,
        cache: Optional[Cache] = None,
        max_turns: Optional[int] = None,
        summary_method: Optional[Union[str, Callable]] = None,
        summary_args: Optional[dict] = {},
        message: Optional[
            Union[
                Dict,
                str,
                Callable,
            ]
        ] = None,
        langfuse_prompt_name: str = None,
        langfuse_prompt_args: Dict[str, Any] = None,
        **context,
    ) -> ChatResult:

        message_to_send = message

        if langfuse_prompt_name:
            message_to_send = self.log_langfuse_prompt(
                langfuse_prompt_name,
                langfuse_prompt_args,
                f"{self.name} --> {recipient.name}",
            )

        return super().a_initiate_chat(
            recipient=recipient,
            clear_history=clear_history,
            silent=silent,
            cache=cache,
            max_turns=max_turns,
            summary_method=summary_method,
            summary_args=summary_args,
            message=message_to_send,
            **context,
        )

    @observe(as_type="generation")
    def generate_reply(
        self,
        messages: List[Dict[str, Any]] | None = None,
        sender: Agent | None = None,
        **kwargs: Any,
    ) -> str | Dict | None:
        """
        Modified generate_reply method that logs the generated reply to langfuse
        under a new langfuse generation observation with the agent's name.
        Also adds generation usage stats to the logging context.

        Args:
            messages (List[Dict[str, Any]] | None): A list of messages to generate a reply from.
            sender (Agent | None): The sender agent of the messages.
            **kwargs (Any): Additional keyword arguments.

        Returns:
            str | Dict | None: The generated reply.

        """
        langfuse_context.update_current_observation(
            name=f"{self.name} --> {sender.name}", input=self.chat_messages[sender]
        )
        reply = super().generate_reply(messages, sender, **kwargs)
        usage = self.get_actual_usage()
        if usage:
            for model_name, usage_dict in usage.items():
                if model_name != "total_cost":
                    langfuse_context.update_current_observation(
                        usage=usage_dict, model=model_name
                    )
        return reply

    @observe(as_type="generation")
    async def a_generate_reply(
        self,
        messages: Optional[List[Dict[str, Any]]] = None,
        sender: Optional["Agent"] = None,
        **kwargs: Any,
    ) -> Union[str, Dict[str, Any], None]:

        langfuse_context.update_current_observation(
            name=f"{self.name} --> {sender.name}",
            input=self.chat_messages[sender],
        )
        langfuse_context.update_current_trace()
        reply = await super().a_generate_reply(messages, sender, **kwargs)
        usage = self.get_actual_usage()
        if usage:
            for model_name, usage_dict in usage.items():
                if model_name != "total_cost":
                    langfuse_context.update_current_observation(
                        usage=usage_dict, model=model_name
                    )
        return reply


class LangfuseMultimodalConversableAgent(
    LangfuseConversableAgent, MultimodalConversableAgent
):
    """
    A MultimodalConversableAgent with the same Langfuse custom logging as LangfuseConversableAgent.
    """
  