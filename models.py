"""
Pydantic models for Claude Code and trajectory data.

The message structures follow Anthropic's API format where applicable.
"""

from typing import Any, Optional, List
from pydantic import BaseModel, Field


# ============================================================================
# Claude Code API Request Model (cc_prompt.json)
# ============================================================================

class CacheControl(BaseModel):
    """Cache control configuration."""
    type: str


class SystemMessage(BaseModel):
    """System message content."""
    type: str = "text"
    text: str
    cache_control: Optional[CacheControl] = Field(default=None, alias="cache_control")

    class Config:
        populate_by_name = True


class Thinking(BaseModel):
    """Thinking configuration."""
    budget_tokens: int = Field(alias="budget_tokens")
    type: str

    class Config:
        populate_by_name = True


class InputSchemaProperty(BaseModel):
    """Input schema property definition."""
    type: str
    description: Optional[str] = None
    enum: Optional[List[str]] = None
    minimum: Optional[int] = None
    maximum: Optional[int] = None
    default: Optional[Any] = None
    min_length: Optional[int] = None
    max_items: Optional[int] = None
    min_items: Optional[int] = None
    format: Optional[str] = None
    items: Optional[dict] = None
    additional_properties: Optional[bool] = Field(default=None, alias="additionalProperties")
    required: Optional[List[str]] = None

    class Config:
        populate_by_name = True


class InputSchema(BaseModel):
    """Tool input schema."""
    type: str = "object"
    properties: dict[str, InputSchemaProperty]
    required: Optional[List[str]] = None
    additional_properties: Optional[bool] = Field(default=None, alias="additionalProperties")
    json_schema: Optional[str] = Field(default=None, alias="$schema")

    class Config:
        populate_by_name = True


class Tool(BaseModel):
    """Tool definition."""
    name: str
    description: str
    input_schema: InputSchema = Field(alias="input_schema")

    class Config:
        populate_by_name = True


class RequestBody(BaseModel):
    """Request body containing messages and metadata."""
    max_tokens: int = Field(alias="max_tokens")
    messages: List[dict]
    metadata: Optional[dict] = None
    model: str
    stream: bool
    system: Optional[List[SystemMessage]] = None
    thinking: Optional[Thinking] = None
    tools: List[Tool]

    class Config:
        populate_by_name = True


class Headers(BaseModel):
    """Request headers."""
    Accept: str = "application/json"
    Authorization: str
    Content_Type: str = Field(default="application/json", alias="Content-Type")
    User_Agent: str = Field(alias="User-Agent")

    class Config:
        populate_by_name = True


class CCRequest(BaseModel):
    """Claude Code API request model."""
    client_ip: str = Field(default="127.0.0.1", alias="client_ip")
    duration_ms: int = Field(alias="duration_ms")
    headers: Headers
    method: str = "POST"
    path: str
    query: str
    request_body: RequestBody = Field(alias="request_body")
    status_code: int = 200
    timestamp: str
    user_agent: str = Field(alias="user_agent")
    placeholders: Optional[dict] = None

    class Config:
        populate_by_name = True


# ============================================================================
# Trajectory/Turn Model (tb-dbg-config-persit-wrong.jsonl)
# ============================================================================

class MessageContent(BaseModel):
    """Message content item.

    This corresponds to Anthropic's ContentBlock type, with additional
    Claude Code specific fields like tool_use_id and input.
    """
    type: str
    text: Optional[str] = None
    id: Optional[str] = None
    name: Optional[str] = None
    input: Optional[dict] = None
    tool_use_id: Optional[str] = Field(default=None, alias="tool_use_id")
    content: Optional[Any] = None
    cache_control: Optional[dict] = Field(default=None, alias="cache_control")

    class Config:
        populate_by_name = True


class MessageUsage(BaseModel):
    """Token usage information.

    This corresponds to Anthropic's Usage type.
    """
    input_tokens: int = Field(alias="input_tokens")
    output_tokens: int = Field(alias="output_tokens")
    cache_read_input_tokens: Optional[int] = Field(default=None, alias="cache_read_input_tokens")

    class Config:
        populate_by_name = True


class Message(BaseModel):
    """Message content.

    This extends Anthropic's MessageParam format with additional fields
    like model, stop_reason, and usage that are part of the API response
    but not the request format.
    """
    role: str
    content: str | List[MessageContent]
    id: Optional[str] = None
    type: Optional[str] = None
    model: Optional[str] = None
    stop_reason: Optional[str] = Field(default=None, alias="stop_reason")
    stop_sequence: Optional[int] = Field(default=None, alias="stop_sequence")
    usage: Optional[MessageUsage] = None

    class Config:
        populate_by_name = True


class ThinkingMetadata(BaseModel):
    """Thinking metadata configuration (Claude Code specific)."""
    level: str
    disabled: bool
    triggers: List


class TodoItem(BaseModel):
    """Todo item (Claude Code specific)."""
    content: str
    status: str  # pending, in_progress, completed
    activeForm: str = Field(alias="activeForm")

    class Config:
        populate_by_name = True


class ToolUseResult(BaseModel):
    """Tool use result data (Claude Code specific)."""
    oldTodos: Optional[List[TodoItem]] = Field(default=None, alias="oldTodos")
    newTodos: Optional[List[TodoItem]] = Field(default=None, alias="newTodos")

    class Config:
        populate_by_name = True


class Turn(BaseModel):
    """A single turn in the conversation trajectory.

    This is a Claude Code specific container that wraps an Anthropic Message
    with additional metadata about the conversation turn.

    The 'message' field follows Anthropic's MessageParam format for the
    role and content structure, with additional response fields like
    stop_reason and usage.
    """
    parent_uuid: str = Field(alias="parentUuid")
    is_sidechain: bool = Field(alias="isSidechain")
    user_type: str = Field(alias="userType")
    cwd: str
    session_id: str = Field(alias="sessionId")
    version: str
    git_branch: str = Field(alias="gitBranch")
    slug: str
    type: str  # user, assistant, etc.
    message: Message  # Uses Anthropic-compatible message format
    uuid: str
    timestamp: str
    thinking_metadata: Optional[ThinkingMetadata] = Field(default=None, alias="thinkingMetadata")
    todos: Optional[List[TodoItem]] = None
    tool_use_result: Optional[ToolUseResult] = Field(default=None, alias="toolUseResult")

    class Config:
        populate_by_name = True
