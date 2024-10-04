from pydantic import BaseModel, Field
from typing import Optional
import base64
import openai
from pathlib import Path

# Function to encode the image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

class ChangeInAccountValue(BaseModel):
    reporting_period: Optional[str] = Field(
        None, description="The time period over which the account value change is being reported."
    )
    starting_value: Optional[float] = Field(
        None, description="The value of the account at the start of the period."
    )
    credits: Optional[float] = Field(
        None, description="Total credits (deposits, contributions, etc.) added to the account during the period."
    )
    debits: Optional[float] = Field(
        None, description="Total debits (withdrawals, distributions, etc.) taken from the account during the period."
    )
    transfer_of_securities: Optional[float] = Field(
        None, description="The value of securities transferred in or out of the account during the period."
    )
    income_reinvested: Optional[float] = Field(
        None, description="Income generated by the account that was reinvested during the period."
    )
    change_in_investment_value: Optional[float] = Field(
        None, description="The change in the value of investments due to market performance."
    )
    accrued_income: Optional[float] = Field(
        None, description="The total income accrued during the period, such as interest or dividends."
    )
    ending_value_with_accrued_income: Optional[float] = Field(
        None, description="The final value of the account at the end of the period, including accrued income."
    )
    ending_value: Optional[float] = Field(
        None, description="The value of the account at the end of the period, excluding accrued income."
    )
    total_change_in_value: Optional[float] = Field(
        None, description="The overall change in the account value during the period, including deposits and withdrawals."
    )
    percentage_change: Optional[float] = Field(
        None, description="The percentage change in the account value during the period."
    )
    total_withdrawals_and_deposits: Optional[float] = Field(
        None, description="The total amount of withdrawals and deposits made during the period."
    )

# Example usage:
change = ChangeInAccountValue(
    starting_value=1000000.0,
    credits=50000.0,
    debits=10000.0,
    transfer_of_securities=0.0,
    income_reinvested=2000.0,
    change_in_investment_value=15000.0,
    accrued_income=3000.0,
    ending_value_with_accrued_income=1058000.0,
    ending_value=1055000.0,
    total_change_in_value=58000.0,
    percentage_change=5.8,
    total_withdrawals_and_deposits=40000.0,
)


SYSTEM_PROMPT = """
You are an assistant in charge of looking at brokerage account statements for your clients.
You will be provided with an image of the account statement. Please carefully look through the provided account statement and extract the relevant information.
You should follow the format provided in the schema. If a field is not present in the statement, you can leave it as null.
"""

MESSAGE = """
Here is the account statement image. Please extract the following information:
{schema}
"""

client = openai.OpenAI(api_key="anything", base_url="http://localhost:9000")

# TODO: Add logging to langfuse
# TODO: include app version in the logs --> use git hash?
# TODO: add prompts to langfuse --> log json schema in the prompt config
# TODO: add examples to langfuse dataset
# TODO: add a scoring function
# TODO: use the runs feature in langfuse to run eval using langfuse dataset with custom scoring function
# TODO: so far its all single round, one shot, try with multi-round and multiple agents (multiple llms) if we have time
async def extractor_app(image_path: Path):
    base64_image = encode_image(image_path)

    chat_response = client.chat.completions.create(
        model="qwen2-vl-7b",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{base64_image}"
                        },
                    },
                    {"type": "text", "text": MESSAGE.format(schema=ChangeInAccountValue.model_json_schema())},
                ],
            },
        ],
    )

    return chat_response.choices[0].message.content


if __name__ == "__main__":

    table_images_dir = Path("/home/hmcp22/hugo-repos/figaro/document_extraction/data")

    for image_path in table_images_dir.glob(f"*.png"):

        # Getting the base64 string
        base64_image = encode_image(image_path)

        chat_response = client.chat.completions.create(
            model="qwen2-vl-7b",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{base64_image}"
                            },
                        },
                        {"type": "text", "text": MESSAGE.format(schema=ChangeInAccountValue.model_json_schema())},
                    ],
                },
            ],
        )
        print("Chat response:", chat_response)