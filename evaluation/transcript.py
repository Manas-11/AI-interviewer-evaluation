"""Validation and analysis helpers for interview transcripts."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

Speaker = Literal["interviewer", "candidate"]


class InterviewTurn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    speaker: Speaker
    text: str = Field(min_length=1, max_length=20_000)

    @field_validator("text")
    @classmethod
    def normalize_text(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Turn text cannot be empty.")
        return value


class InterviewInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    role: str = Field(min_length=1, max_length=200)
    instructions: str = Field(min_length=1, max_length=5_000)
    transcript: list[InterviewTurn] = Field(min_length=1, max_length=1_000)

    @field_validator("role", "instructions")
    @classmethod
    def normalize_required_text(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("This field cannot be empty.")
        return value


def process_transcript(data: dict) -> dict:
    """Validate transcript input and return a stable, UI-friendly structure."""

    interview = InterviewInput.model_validate(data)
    turns = [
        {"turn_number": index, "speaker": turn.speaker, "text": turn.text}
        for index, turn in enumerate(interview.transcript, start=1)
    ]
    return {
        "role": interview.role,
        "instructions": interview.instructions,
        "turn_count": len(turns),
        "candidate_turn_count": sum(turn["speaker"] == "candidate" for turn in turns),
        "interviewer_turn_count": sum(turn["speaker"] == "interviewer" for turn in turns),
        "turns": turns,
    }
