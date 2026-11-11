"""Railway Passenger Help Desk — Helpdesk Package.

Member 1 (Christy Marrya George — AI Assistant Lead) exports:
- LLM integration: ``LLMClient``, ``LLMConfig``, ``LLMResponse``, ``ToolCall``, exceptions
- Prompts: ``build_assistant_system_prompt``, ``build_answer_system_prompt``
- Grounded answers: ``AnswerGenerator``, ``GroundedAnswer``, ``SourceCitation``
- Memory integration: ``MemoryAdapter``, ``MemoryBackendProtocol``
- AI assistant coordinator: ``AIAssistant``, ``ToolExecutorProtocol``
"""

from helpdesk.ai_assistant import AIAssistant, ToolExecutorProtocol
from helpdesk.answer import AnswerGenerator, GroundedAnswer, SourceCitation
from helpdesk.llm import (
    LLMAuthError,
    LLMClient,
    LLMConfig,
    LLMConfigurationError,
    LLMConnectionError,
    LLMError,
    LLMInvalidResponseError,
    LLMRateLimitError,
    LLMResponse,
    LLMTimeoutError,
    ToolCall,
)
from helpdesk.memory_adapter import MemoryAdapter, MemoryBackendProtocol
from helpdesk.prompts import build_answer_system_prompt, build_assistant_system_prompt

__all__ = [
    "AIAssistant",
    "AnswerGenerator",
    "GroundedAnswer",
    "LLMAuthError",
    "LLMClient",
    "LLMConfig",
    "LLMConfigurationError",
    "LLMConnectionError",
    "LLMError",
    "LLMInvalidResponseError",
    "LLMRateLimitError",
    "LLMResponse",
    "LLMTimeoutError",
    "MemoryAdapter",
    "MemoryBackendProtocol",
    "SourceCitation",
    "ToolCall",
    "ToolExecutorProtocol",
    "build_answer_system_prompt",
    "build_assistant_system_prompt",
]
