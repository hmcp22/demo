from config import langfuse_client
import json
from single_round_extractors import (
    openai_single_round_extractor_with_structured_outputs,
    non_openai_single_round_extractor,
)
from utils.utils import extract_jsons_from_message_content
from pathlib import Path

models = [
    # "gpt-4o",
    # "qwen2-vl-7b",
    "pixtral-12b",
    # "claude-3.5-sonnet",
    # "gpt-4o-mini",
    # "llama-3.2-11b-vision-preview",
]

dataset = langfuse_client.get_dataset("brokerage_statements_table_extraction")


def eval_exact_match(output_json, expected_output_json):

    expected_output_dict = json.loads(expected_output_json)
    try:
        output_dict = json.loads(output_json)
    except:
        return 0

    score = 0
    for key, value in expected_output_dict.items():
        if key in output_dict and output_dict[key] == value:
            score += 1
    return score / len(expected_output_dict)


for item in dataset.items:

    image_path = item.input["args"][0]

    image_name = Path(image_path).stem

    for model in models:

        if "gpt" in model or "o1" in model:
            llm_application = openai_single_round_extractor_with_structured_outputs
            prompt_name = "extractor_system_prompt"
            extract_json = False
        else:
            llm_application = non_openai_single_round_extractor
            extract_json = True
            prompt_name = "qwen_extractor_prompt"

        prompt = langfuse_client.get_prompt(prompt_name)
        with item.observe(
            run_name=f"{model}_eval",
            run_description=f"Eval {model}",
            run_metadata={
                "model": model,
                "prompt_version": prompt.version,
                "prompt_name": prompt_name,
            },
        ) as trace_id:
            try:
                output = llm_application(image_path, prompt_name, model=model)
                if extract_json :
                    output_json = extract_jsons_from_message_content(output)[0]
                else:
                    output_json = output
                langfuse_client.score(
                    trace_id=trace_id,
                    name="exact_match",
                    value=eval_exact_match(output_json, item.expected_output),
                    comment=f"Exact match model={model}, image={image_name}",
                )
            except Exception as e:
                print(f"Error: {e}")
                continue

langfuse_client.flush()
