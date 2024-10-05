from pydantic import BaseModel, Field
from typing import Optional

class ChangeInAccountValue(BaseModel):
    reporting_period_start_date: Optional[str] = Field(
        None,
        description="The start date for the time period over which the account value change is being reported. In the format YYYY-MM-DD.",
    )
    reporting_period_end_date: Optional[str] = Field(
        None,
        description="The end date for the time period over which the account value change is being reported. In the format YYYY-MM-DD.",
    )
    starting_value: Optional[float] = Field(
        None, description="The value of the account at the start of the period."
    )
    credits: Optional[float] = Field(
        None,
        description="Total credits (deposits, contributions, etc.) added to the account during the period.",
    )
    debits: Optional[float] = Field(
        None,
        description="Total debits (withdrawals, distributions, etc.) taken from the account during the period.",
    )
    transfer_of_securities: Optional[float] = Field(
        None,
        description="The value of securities transferred in or out of the account during the period.",
    )
    transaction_costs_fees_and_charges: Optional[float] = Field(
        None,
        description="The total transaction costs, fees, and charges incurred during the period.",
    )
    income_reinvested: Optional[float] = Field(
        None,
        description="Income generated by the account that was reinvested during the period.",
    )
    change_in_investment_value: Optional[float] = Field(
        None,
        description="The change in the value of investments due to market performance.",
    )
    accrued_income: Optional[float] = Field(
        None,
        description="The total income accrued during the period, such as interest or dividends.",
    )
    ending_value_with_accrued_income: Optional[float] = Field(
        None,
        description="The final value of the account at the end of the period, including accrued income.",
    )
    ending_value: Optional[float] = Field(
        None,
        description="The value of the account at the end of the period, excluding accrued income.",
    )
    total_change_in_value: Optional[float] = Field(
        None,
        description="The overall change in the account value during the period, including deposits, withdrawals and any accrued income.",
    )
