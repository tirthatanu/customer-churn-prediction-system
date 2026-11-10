"""Strict API contracts for the churn prediction service."""
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class CustomerInput(BaseModel):
    """Customer features available in the supplied dataset.

    Boundaries are basic input plausibility checks, not a guarantee that an
    out-of-distribution input will produce a meaningful prediction.
    """

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    age: int = Field(ge=18, le=100, examples=[30], description="Customer age in years")
    gender: Literal["Female", "Male"]
    tenure: int = Field(ge=0, le=120, examples=[12], description="Months as a subscriber")
    usage_frequency: int = Field(ge=0, le=365, examples=[15])
    support_calls: int = Field(ge=0, le=100, examples=[4])
    payment_delay: int = Field(ge=0, le=365, examples=[10], description="Payment delay in days")
    subscription_type: Literal["Basic", "Standard", "Premium"]
    contract_length: Literal["Monthly", "Quarterly", "Annual"]
    total_spend: float = Field(ge=0, le=10_000_000, allow_inf_nan=False, examples=[906.0])
    last_interaction: int = Field(ge=0, le=365, examples=[9], description="Days since last interaction")


class PredictionResponse(BaseModel):
    prediction: Literal["Churn", "No Churn"]
    probability: float = Field(ge=0, le=1, description="Estimated probability of churn (class 1)")
    risk_level: Literal["Low", "Medium", "High"]


class PredictionHistoryEntry(PredictionResponse):
    id: int
    created_at_utc: str


class HistoryResponse(BaseModel):
    count: int
    limit: int
    offset: int
    predictions: list[PredictionHistoryEntry]
