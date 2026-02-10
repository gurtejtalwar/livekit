from pydantic import BaseModel, ConfigDict
from enum import Enum

class PostCallAnalysis(BaseModel):
    intent: str
    emotion: str
    score: float
    objection_analysis: str
    summary: str
    follow_up_trigger: bool
    lead_type: str

    model_config = ConfigDict(for_orm=True)
