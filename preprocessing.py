#!/usr/bin/env python3
"""
System Prompt Pre-processing Module

This module provides functions to load system prompt configuration
from Claude Code's API request format.

Note: Anthropic's API uses 'system' as a separate parameter, not a message role.
System prompts and tools are API configuration, not part of the conversation trajectory.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import typer


def load_system_prompt(prompt_path: Path) -> Optional[Dict[str, Any]]:
    """Load the system prompt from cc_prompt.json.

    The file contains system configuration including:
    - system: List of system messages
    - tools: List of tool definitions
    - thinking: Thinking configuration
    - Other metadata (status_code, timestamp, user_agent, placeholders)

    Args:
        prompt_path: Path to the cc_prompt.json file

    Returns:
        Dictionary with system configuration or None if loading fails
    """
    try:
        with open(prompt_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data
    except FileNotFoundError:
        typer.echo(f"Warning: System prompt file not found: {prompt_path}", err=True)
        return None
    except Exception as e:
        typer.echo(f"Warning: Failed to load system prompt: {e}", err=True)
        return None


def extract_system_config(data: Dict[str, Any]) -> Dict[str, Any]:
    """Extract system configuration from cc_prompt.json data.

    Returns the system prompt and tools in the format expected by
    Anthropic's Messages API.

    Args:
        data: Loaded JSON data from cc_prompt.json

    Returns:
        Dictionary with system configuration:
        {
            "system": str or List[dict],  # System prompt content
            "tools": List[dict],            # Tool definitions
            "thinking": Optional[dict]      # Thinking configuration
        }
    """
    # Extract system prompt (may be multiple system messages)
    system_parts = []
    if "system" in data:
        for sys_msg in data["system"]:
            if isinstance(sys_msg, dict) and "text" in sys_msg:
                system_parts.append(sys_msg["text"])

    # Join multiple system messages with newlines, or use as-is
    system_prompt = "\n\n".join(system_parts) if len(system_parts) > 1 else system_parts[0] if system_parts else ""

    # Extract tools - they're already in the right format
    tools = data.get("tools", [])

    # Extract thinking config if present
    thinking = data.get("thinking", None)

    return {
        "system": system_prompt,
        "tools": tools,
        "thinking": thinking
    }


def get_default_system_prompt_path() -> Path:
    """Get the default path to the system prompt file.

    Returns:
        Path to resource/cc_prompt.json relative to the script directory
    """
    # Get the directory where this script is located
    script_dir = Path(__file__).parent
    return script_dir / "resource" / "cc_prompt.json"
