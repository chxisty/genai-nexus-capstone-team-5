
"""Context analysis for Railway Passenger Help Desk.

Member 4: Context Management and Response Reliability.

Detects follow-up questions and provides relevant conversation
history without inventing missing railway information.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class ContextAnalysis:
    question: str
    is_follow_up: bool
    needs_clarification: bool
    clarification_question: str | None
    relevant_history: list[dict[str, str]]


class ContextHandler:
    """Analyze whether a passenger question depends on prior turns."""

    FOLLOW_UP_PATTERNS = (
        r"^(what about|how about)\b",
        r"^(and|also|then)\b",
        r"^(what if)\b",
        r"^(same|another|the other)\b",
        r"^(tomorrow|today|tonight|yesterday)\b",
        r"^(yes|no|okay|ok|sure)\b",
        r"^(it|its|that|those|they|them|there)\b",
        r"\b(this|that|same|other)\s+train\b",
    )

    def __init__(self, max_history_messages: int = 8) -> None:
        if max_history_messages < 0:
            raise ValueError(
                "max_history_messages cannot be negative"
            )

        self.max_history_messages = max_history_messages

    def analyze(
        self,
        question: str,
        history: Sequence[Mapping[str, Any]] | None = None,
    ) -> ContextAnalysis:
        """Classify a question and select recent conversation context."""

        cleaned = question.strip()

        if not cleaned:
            return ContextAnalysis(
                question="",
                is_follow_up=False,
                needs_clarification=True,
                clarification_question="Please enter your railway question.",
                relevant_history=[],
            )

        relevant_history: list[dict[str, str]] = []

        for message in history or []:
            role = str(message.get("role", "")).lower()

            if role not in {"user", "assistant"}:
                continue

            content = str(message.get("content", "")).strip()

            if not content:
                continue

            relevant_history.append({
                "role": role,
                "content": content,
            })

        if self.max_history_messages == 0:
            relevant_history = []
        else:
            relevant_history = relevant_history[
                -self.max_history_messages:
            ]

        normalized = re.sub(r"\s+", " ", cleaned).lower()

        is_follow_up = any(
            re.search(pattern, normalized)
            for pattern in self.FOLLOW_UP_PATTERNS
        )

        # Identify whether the passenger refers to a specific train.
        train_reference = bool(
            re.search(
                r"\b(this|that|same|other)\s+train\b",
                normalized,
            )
        )

        # Collect distinct five-digit train numbers from history.
        train_numbers: set[str] = set()

        for message in relevant_history:
            numbers = re.findall(
                r"\b\d{5}\b",
                message["content"],
            )

            train_numbers.update(numbers)

        # Check whether the passenger explicitly selected one train
        # after discussing multiple trains.
        selected_train = None

        for message in relevant_history:
            if message["role"] != "user":
                continue

            content = message["content"].strip()

            # Accept a clear, standalone train selection.
            match = re.fullmatch(
                r"(?:train\s*(?:number|no\.?|#)?\s*)?(\d{5})[.!]?",
                content,
                re.IGNORECASE,
            )

            if match:
                selected_train = match.group(1)

        # A later explicit selection takes priority over older
        # discussions containing multiple train numbers.
        has_single_train = (
            selected_train is not None
            or len(train_numbers) == 1
        )

        # Ask for clarification when:
        # 1. A follow-up has no previous conversation.
        # 2. A referenced train cannot be uniquely identified.
        needs_clarification = (
            (is_follow_up and not relevant_history)
            or (
                train_reference
                and not has_single_train
            )
        )

        clarification_question = None

        if needs_clarification:
            clarification_question = (
                "Could you provide more details about your "
                "railway question, such as the train, route, "
                "or journey you are referring to?"
            )

        return ContextAnalysis(
            question=cleaned,
            is_follow_up=is_follow_up,
            needs_clarification=needs_clarification,
            clarification_question=clarification_question,
            relevant_history=relevant_history,
        )
