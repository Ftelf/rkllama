import json
import logging
import math
from typing import Any, Dict, Optional, Tuple, Union, List
import re
import copy
import uuid
import time
from flask import jsonify
import cv2
import numpy as np
import os
import base64
import requests
from PIL import Image
import io

import rkllama.config

try:
    from pydantic import BaseModel, ValidationError, create_model
    from pydantic.fields import FieldInfo
    PYDANTIC_AVAILABLE = True
except ImportError:
    PYDANTIC_AVAILABLE = False
    # Use a simple fallback if Pydantic is not available
    class BaseModel:
        pass
    
    class ValidationError(Exception):
        pass
        
    def create_model(*args, **kwargs):
        return None

logger = logging.getLogger("rkllama.format_utils")

def get_pydantic_type(json_type_name: str):
    """Convert JSON schema type to Python/Pydantic type"""
    if not PYDANTIC_AVAILABLE:
        return Any
        
    if json_type_name == "string":
        return str
    elif json_type_name == "integer":
        return int
    elif json_type_name == "number":
        return float
    elif json_type_name == "boolean":
        return bool
    elif json_type_name == "array":
        return List[Any]
    elif json_type_name == "object":
        return Dict[str, Any]
    return Any

def create_pydantic_model(format_spec: Dict) -> Optional[type]:
    """Create a Pydantic model from a JSON schema"""
    if not PYDANTIC_AVAILABLE:
        logger.warning("Pydantic not available, format validation disabled")
        return None
        
    if not format_spec or not isinstance(format_spec, dict):
        return None
        
    try:
        # Get schema properties and required fields
        properties = format_spec.get("properties", {})
        required = format_spec.get("required", [])
        
        # Create field definitions for the Pydantic model
        fields = {}
        for prop_name, prop_spec in properties.items():
            prop_type = prop_spec.get("type", "string")
            python_type = get_pydantic_type(prop_type)
            
            # Make field optional if not required
            if prop_name not in required:
                fields[prop_name] = (Optional[python_type], None)
            else:
                fields[prop_name] = (python_type, ...)
        
        # Create dynamic model based on the schema
        model_name = format_spec.get("title", "DynamicResponseModel")
        model = create_model(model_name, **fields)
        return model
    except Exception as e:
        logger.error(f"Error creating Pydantic model from schema: {str(e)}")
        return None

def create_format_instruction(format_spec):
    """Create a format instruction based on the format specification"""
    if not format_spec:
        return ""
    
    instruction = "\n\n"
    
    # Handle different format types
    if isinstance(format_spec, dict):
        format_type = format_spec.get('type', '')
        
        if format_type == 'json':
            instruction += "You must respond with a valid JSON. Return only the JSON with no explanation text before or after it."
        
        elif format_type == 'object':
            # For object type, create a template based on properties
            properties = format_spec.get('properties', {})
            example = {}
            
            # Create example values for each property
            for prop, details in properties.items():
                prop_type = details.get('type', 'string')
                if prop_type == 'string':
                    example[prop] = ""
                elif prop_type == 'integer':
                    example[prop] = 0
                elif prop_type == 'number':
                    example[prop] = 0.0
                elif prop_type == 'boolean':
                    example[prop] = False
                elif prop_type == 'array':
                    example[prop] = []
                elif prop_type == 'object':
                    example[prop] = {}
            
            required = format_spec.get('required', [])
            if required:
                required_str = ", ".join(required)
                instruction += f"You must respond with a valid JSON object with exactly these required fields: {required_str}.\n\n"
            
            # Add example JSON structure
            instruction += "Format your entire response as a JSON object with ONLY these fields:\n"
            instruction += "```json\n"
            instruction += json.dumps(example, indent=2)
            instruction += "\n```\n\n"
            instruction += "Return ONLY the JSON object, with no explanations, comments or text before or after the JSON.\n"
            instruction += "Never use '_' prefix in your field names."
    
    # Handle simple string format specification like format="json"
    elif isinstance(format_spec, str):
        if format_spec.lower() == 'json':
            instruction += "You must respond with valid JSON. Return ONLY the JSON with no explanation or text before or after it.\n"
            instruction += "Format your entire response as a JSON object containing all the relevant information from your answer.\n"
            instruction += "Ensure the JSON is properly formatted and valid."
    
    return instruction

def get_example_value(type_name: str) -> str:
    """Return an example value for a given JSON schema type"""
    if type_name == "string":
        return '""'
    elif type_name == "integer":
        return "0"
    elif type_name == "number":
        return "0.0"
    elif type_name == "boolean":
        return "false"
    elif type_name == "array":
        return "[]"
    elif type_name == "object":
        return "{}"
    elif type_name == "null":
        return "null"
    return '""'  # default to string

def extract_json(text):
    """Extract JSON from text that might contain non-JSON content"""
    
    # First look for JSON in code blocks
    code_block_pattern = r'```(?:json)?\s*([\s\S]*?)\s*```'
    code_matches = re.findall(code_block_pattern, text)
    
    for potential_json in code_matches:
        try:
            parsed = json.loads(potential_json)
            return potential_json.strip(), parsed
        except json.JSONDecodeError:
            continue
    
    # If no valid JSON in code blocks, try to find JSON-like content directly
    json_pattern = r'(\{(?:[^{}]|(?:\{[^{}]*\}))*\})'
    json_matches = re.findall(json_pattern, text)
    
    for potential_json in json_matches:
        try:
            parsed = json.loads(potential_json)
            return potential_json.strip(), parsed
        except json.JSONDecodeError:
            continue
    
    # Try with more lenient pattern
    more_lenient_pattern = r'\{[\s\S]*?\}'
    lenient_matches = re.findall(more_lenient_pattern, text)
    
    for potential_json in lenient_matches:
        # Clean up the text
        cleaned = re.sub(r'[^\{\}\[\],:."\'0-9a-zA-Z_\s-]', '', potential_json)
        cleaned = cleaned.replace("'", '"')  # Replace single quotes with double quotes
        
        try:
            parsed = json.loads(cleaned)
            return cleaned.strip(), parsed
        except json.JSONDecodeError:
            continue
    
    # No valid JSON found
    return None, None

def validate_format_response(text, format_spec):
    """
    Validate that the model's response matches the requested format
    
    Args:
        text: The model's response text
        format_spec: The format specification (dict or string)
    
    Returns:
        tuple: (success, parsed_data, error_message, cleaned_json)
    """
    if not format_spec:
        return False, None, "No format specification provided", None
    
    # Extract JSON from the response text
    json_text, parsed_data = extract_json(text)
    
    if not json_text or not parsed_data:
        return False, None, "Could not extract valid JSON from response", None
    
    # For simple 'json' format, we just need valid JSON
    if format_spec == 'json' or (isinstance(format_spec, str) and format_spec.lower() == 'json') or \
       (isinstance(format_spec, dict) and format_spec.get('type') == 'json'):
        return True, parsed_data, None, json_text
    
    # For 'object' format with schema validation
    if isinstance(format_spec, dict) and format_spec.get('type') == 'object':
        properties = format_spec.get('properties', {})
        required = format_spec.get('required', [])
        
        # Verify all required fields are present
        missing_fields = []
        for field in required:
            if field not in parsed_data:
                missing_fields.append(field)
        
        if missing_fields:
            return False, None, f"Missing required field{'s' if len(missing_fields) > 1 else ''}: {', '.join(missing_fields)}", None
        
        # Check field types
        for field, value in parsed_data.items():
            if field in properties:
                expected_type = properties[field].get('type')
                
                # Validate type
                if expected_type == 'string' and not isinstance(value, str):
                    return False, None, f"Field '{field}' should be a string", None
                elif expected_type == 'number' and not isinstance(value, (int, float)):
                    return False, None, f"Field '{field}' should be a number", None
                elif expected_type == 'integer':
                    # Convert floats to ints if they are whole numbers
                    if isinstance(value, float) and value.is_integer():
                        parsed_data[field] = int(value)
                    elif not isinstance(value, int):
                        return False, None, f"Field '{field}' should be an integer", None
                elif expected_type == 'boolean' and not isinstance(value, bool):
                    return False, None, f"Field '{field}' should be a boolean", None
                elif expected_type == 'array' and not isinstance(value, list):
                    return False, None, f"Field '{field}' should be an array", None
                elif expected_type == 'object' and not isinstance(value, dict):
                    return False, None, f"Field '{field}' should be an object", None
        
        # Create a clean JSON with only the expected fields
        if properties:
            clean_data = {}
            for field in properties.keys():
                if field in parsed_data:
                    clean_data[field] = parsed_data[field]
            
            # Include any required fields not in properties
            for field in required:
                if field not in clean_data and field in parsed_data:
                    clean_data[field] = parsed_data[field]
            
            cleaned_json = json.dumps(clean_data, indent=2)
            return True, clean_data, None, cleaned_json
    
    return True, parsed_data, None, json_text


def openai_to_ollama_chat_request(openai_payload: dict) -> dict:
    """
    Translate an OpenAI /v1/chat/completions request payload to Ollama /api/chat format.

    Args:
        openai_payload (dict): OpenAI request payload.

    Returns:
        dict: Ollama-compatible request payload.
    """
    messages = openai_payload.get("messages", [])
    model = openai_payload.get("model", "llama3")
    stream = openai_payload.get("stream", False)

    # Base Ollama payload
    ollama_payload = {
        "model": model,
        "messages": messages,
        "stream": stream,
    }

    # Supported Ollama options from OpenAI fields
    supported_option_mappings = {
        "temperature": "temperature",
        "top_p": "top_p",
        "top_k": "top_k",
        "presence_penalty": "presence_penalty",
        "frequency_penalty": "frequency_penalty",
        "stop": "stop",
        "max_tokens": "max_new_tokens",
        "max_completion_tokens": "max_new_tokens",
        "seed": "seed",
        "logit_bias": "logit_bias",  # If supported
    }

    for openai_key, ollama_key in supported_option_mappings.items():
        if openai_key in openai_payload:
            ollama_payload.setdefault("options", {})[ollama_key] = openai_payload[openai_key]

    # Handle tool_choice, tools, functions
    # Ollama currently has no native tool/function support (like OpenAI tool-calling)
    # But we include them for forward compatibility if needed by custom handler
    for passthrough_key in ["tools", "tool_choice", "functions", "function_call", "response_format", "n"]:
        if passthrough_key in openai_payload:
            ollama_payload[passthrough_key] = openai_payload[passthrough_key]

    # Multimodal Support: handle images in messages
    for message in ollama_payload["messages"]:
        if message.get("role") in ["user"]:
            content = message.get("content", "")
            if isinstance(content, list):
                # If content is already a list, process each item
                images = []
                content_tmp = []
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "image_url" and "image_url" in item:
                        image_url = item["image_url"]["url"]
                        images.append(image_url)
                    elif isinstance(item, dict) and item.get("type") == "text" and "text" in item:
                        # Only keep non-image items in content
                        content_tmp.append(item["text"])
                if images:
                    message["images"] = images
                    message["content"] = ". ".join(content_tmp) if content_tmp else ""
            elif isinstance(content, dict) and content.get("type") == "image_url" and "image_url" in content:
                # Single image content
                image_url = content["image_url"]["url"]
                message["images"] = [image_url]
                message["content"] = ""
    
    return ollama_payload


def openai_to_ollama_generate_request(openai_payload: dict) -> dict:
    """
    Translate an OpenAI /v1/completions request payload to Ollama /api/generate format.
    Args:
        openai_payload (dict): OpenAI request payload for /v1/completions.
    Returns:
        dict: Ollama-compatible /api/generate request payload.
    """
    # Handle "prompt": could be a string or an array
    prompt = openai_payload.get("prompt", "")
    if isinstance(prompt, list):
        # Join multi-part array into a single string
        prompt = "\n".join([str(part) for part in prompt])
    
    model = openai_payload.get("model", "llama3")
    stream = openai_payload.get("stream", False)
    images = openai_payload.get("images", [])

    # Base Ollama payload
    ollama_payload = {
        "model": model,
        "prompt": prompt,
        "stream": stream,
    }

    # Supported Ollama option mappings
    supported_option_mappings = {
        "temperature": "temperature",
        "top_p": "top_p",
        "top_k": "top_k",
        "presence_penalty": "presence_penalty",
        "frequency_penalty": "frequency_penalty",
        "stop": "stop",
        "max_tokens": "max_new_tokens",  # OpenAI max_tokens => Ollama max_new_tokens
        "seed": "seed",
        "logit_bias": "logit_bias",  # If supported by your Ollama deployment
    }

    for openai_key, ollama_key in supported_option_mappings.items():
        if openai_key in openai_payload:
            ollama_payload.setdefault("options", {})[ollama_key] = openai_payload[openai_key]

    # Pass through extra non-standard fields for forward compatibility (optional)
    for passthrough_key in ["n", "best_of", "logprobs", "echo", "user"]:
        if passthrough_key in openai_payload:
            ollama_payload[passthrough_key] = openai_payload[passthrough_key]

    # Muktimdoal Support: handle images in prompt if any
    if images:
        if isinstance(images, list):
            # If content is already a list, process each item
            images_tmp = []
            for item in images:
                if isinstance(item, dict) and item.get("type") == "image_url" and "image_url" in item:
                    image_url = item["image_url"]["url"]
                    images_tmp.append(image_url)
            if images_tmp:
                ollama_payload["images"] = images_tmp
        elif isinstance(images, dict) and images.get("type") == "image_url" and "image_url" in images:
            # Single image content
            image_url = images["image_url"]["url"]
            ollama_payload["images"] = [image_url]

    return ollama_payload


def ollama_chat_to_openai_v1_chat_completion(ollama_response: dict) -> dict:
    """
    Convert Ollama's chat response to a fully OpenAI-compatible /v1/chat/completions response.
    
    Args:
        ollama_response (dict): Response from Ollama's /api/chat endpoint.
    
    Returns:
        dict: OpenAI-compatible response.
    """
    
    # Generate metadata
    completion_id = f"chatcmpl-{uuid.uuid4().hex}"
    created = int(time.time())
    model = ollama_response.get("model", "unknown-model")

    # Extract message
    message = ollama_response.get("message", {})
    role = message.get("role", "assistant")
    content = message.get("content", "")
    tool_calls = message.get("tool_calls", None)

    # Handle finish_reason
    finish_reason = "stop" if ollama_response.get("done", True) else None

    # Build the choice message
    choice_message = {
        "role": role,
        "content": content
    }

    if tool_calls:
        # OpenAI v1 supports `tool_calls`
        for tool in tool_calls:
            tool["id"] = f"call_{uuid.uuid4().hex}"
            tool["type"] = "function"
        choice_message["tool_calls"] = tool_calls
        finish_reason = ollama_response.get("done_reason")


    choice = {
        "index": 0,
        "message": choice_message,
        "finish_reason": finish_reason
    }

    # Handle token usage if present
    prompt_tokens = ollama_response.get("prompt_eval_count", 0)
    completion_tokens = ollama_response.get("eval_count", 0)
    eval_duration_ns = ollama_response.get("eval_duration", 0)
    prompt_eval_duration_ns = ollama_response.get("prompt_eval_duration", 0)
    total_duration_ns = ollama_response.get("total_duration", 0)
    load_duration_ns = ollama_response.get("load_duration", 0)
    eval_duration = eval_duration_ns / 1_000_000_000 if eval_duration_ns > 0 else 0
    prompt_eval_duration = prompt_eval_duration_ns / 1_000_000_000 if prompt_eval_duration_ns > 0 else 0
    total_duration = total_duration_ns / 1_000_000_000 if total_duration_ns > 0 else 0
    load_duration = load_duration_ns / 1_000_000_000 if load_duration_ns > 0 else 0
    usage = {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
        "tokens_per_second": round(completion_tokens / eval_duration, 2) if eval_duration > 0 else 0,
        "prompt_eval_duration": round(prompt_eval_duration, 4),
        "eval_duration": round(eval_duration, 4),
        "total_duration": round(total_duration, 4),
        "load_duration": round(load_duration, 4),
    }

    # Build full response
    openai_response = {
        "id": completion_id,
        "object": "chat.completion",
        "created": created,
        "model": model,
        "choices": [choice],
    }

    if usage:
        openai_response["usage"] = usage

    return openai_response



def ollama_generate_to_openai_v1_completion(ollama_response: dict) -> dict:
    """
    Convert Ollama's /api/generate response to a fully OpenAI-compatible /v1/completions response.

    Args:
        ollama_response (dict): Response from Ollama's /api/generate endpoint.

    Returns:
        dict: OpenAI-compatible /v1/completions response.
    """

    # Metadata
    completion_id = f"cmpl-{uuid.uuid4().hex}"
    created = int(time.time())
    model = ollama_response.get("model", "unknown-model")

    # Generated text
    content = ollama_response.get("response", "")

    # Finish reason
    finish_reason = ollama_response.get("done_reason", "stop" if ollama_response.get("done", True) else None)

    # Build choice
    choice = {
        "text": content,
        "index": 0,
        "logprobs": None,
        "finish_reason": finish_reason
    }

    # Usage
    usage = {}
    if "prompt_eval_count" in ollama_response:
        usage["prompt_tokens"] = ollama_response["prompt_eval_count"]
    if "eval_count" in ollama_response:
        usage["completion_tokens"] = ollama_response["eval_count"]
    if usage:
        usage["total_tokens"] = usage.get("prompt_tokens", 0) + usage.get("completion_tokens", 0)

    # Assemble OpenAI-style response
    openai_completion_response = {
        "id": completion_id,
        "object": "text_completion",
        "created": created,
        "model": model,
        "choices": [choice]
    }
    if usage:
        openai_completion_response["usage"] = usage

    return openai_completion_response


def ollama_embedding_to_openai_v1_embeddingns(ollama_response: dict) -> dict:
    """
    Convert Ollama's /api/embed response to a fully OpenAI-compatible /v1/embedding response.

    Args:
        ollama_response (dict): Response from Ollama's /api/embed endpoint.

    Returns:
        dict: OpenAI-compatible /v1/embedding response.
    """
  
    # Metadata
    model = ollama_response.get("model", "unknown-model")

    # Generated text
    embeddings = ollama_response.get("embeddings", "")

    # Build choice
    data = {
        "object": "embedding",
        "embedding": [item for embedding in embeddings for item in embedding],
        "index": 0
    }

    # Usage
    usage = {}
    if "prompt_eval_count" in ollama_response:
        usage["prompt_tokens"] = ollama_response["prompt_eval_count"]
    if usage:
        usage["total_tokens"] = usage.get("prompt_tokens", 0) + usage.get("completion_tokens", 0)

    # Assemble OpenAI-style response
    openai_completion_response = {
        "object": "list",
        "data": [data],
        "model": model
    }

    if usage:
        openai_completion_response["usage"] = usage

    return openai_completion_response



def ollama_chat_stream_to_openai_chat_completions_chunks(ollama_stream_lines):
    """
    Converts an iterable of Ollama stream JSON lines to OpenAI SSE streaming chunks.

    Args:
        ollama_stream_lines (iterable[str]): Streamed JSON lines from Ollama.

    Yields:
        str: OpenAI-compatible `data: ...\n\n` formatted SSE chunks.
    """

    completion_id = f"chatcmpl-{uuid.uuid4().hex}"
    created = int(time.time())
    for line in ollama_stream_lines:
        line = str(line).strip()
        if not line or line.startswith("data:"):
            continue

        try:
            ollama_chunk = json.loads(line)
        except json.JSONDecodeError:
            continue

        content_piece = ollama_chunk.get("message", {}).get("content", "")
        role = ollama_chunk.get("message", {}).get("role")
        tool_calls = ollama_chunk.get("message", {}).get("tool_calls")
        model = ollama_chunk.get("model", "unknown-model")
        finish_reason = ollama_chunk.get("done_reason", None)

        delta = {}
        delta["content"] = content_piece
        if role:
            delta["role"] = role
        if tool_calls:
            for idx,tool in enumerate(tool_calls):
                tool["id"] = f"call_{uuid.uuid4().hex}"
                tool["type"] = "function"
                tool["index"] = idx
                tool["function"]["arguments"] = str(tool["function"]["arguments"]).replace("'", '"')
            delta["tool_calls"] = tool_calls
        
        chunk = {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [{
                "index": 0,
                "delta": delta,
                "finish_reason": finish_reason
            }]
        }

        yield f"data: {json.dumps(chunk)}\n\n"

        if ollama_chunk.get("done") is True:
            # Final chunk — stop streaming
            eval_count = ollama_chunk.get("eval_count", 0)
            prompt_tokens = ollama_chunk.get("prompt_eval_count", 0)
            eval_duration_ns = ollama_chunk.get("eval_duration", 0)
            prompt_eval_duration_ns = ollama_chunk.get("prompt_eval_duration", 0)
            total_duration_ns = ollama_chunk.get("total_duration", 0)
            load_duration_ns = ollama_chunk.get("load_duration", 0)
            eval_duration = eval_duration_ns / 1_000_000_000 if eval_duration_ns > 0 else 0
            prompt_eval_duration = prompt_eval_duration_ns / 1_000_000_000 if prompt_eval_duration_ns > 0 else 0
            total_duration = total_duration_ns / 1_000_000_000 if total_duration_ns > 0 else 0
            load_duration = load_duration_ns / 1_000_000_000 if load_duration_ns > 0 else 0
            tokens_per_second = round(eval_count / eval_duration, 2) if eval_duration > 0 else 0
            usage = {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": eval_count,
                "total_tokens": prompt_tokens + eval_count,
                "tokens_per_second": tokens_per_second,
                "prompt_eval_duration": round(prompt_eval_duration, 4),
                "eval_duration": round(eval_duration, 4),
                "total_duration": round(total_duration, 4),
                "load_duration": round(load_duration, 4),
            }
            final_chunk = {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [],
                "usage": usage
            }
            yield f"data: {json.dumps(final_chunk)}\n\n"
            yield "data: [DONE]\n\n"
            break


def ollama_generate_stream_to_openai_completions_chunks(ollama_stream_lines):
    """
    Converts an iterable of Ollama stream JSON lines to OpenAI SSE streaming chunks.

    Args:
        ollama_stream_lines (iterable[str]): Streamed JSON lines from Ollama.

    Yields:
        str: OpenAI-compatible `data: ...\n\n` formatted SSE chunks.
    """

    completion_id = f"chatcmpl-{uuid.uuid4().hex}"
    created = int(time.time())
    for line in ollama_stream_lines:
        line = str(line).strip()
        if not line or line.startswith("data:"):
            continue

        try:
            ollama_chunk = json.loads(line)
        except json.JSONDecodeError:
            continue

        content_piece = ollama_chunk.get("response", "")
        model = ollama_chunk.get("model", "unknown-model")
        finish_reason = ollama_chunk.get("done_reason", None)

        chunk = {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [{
                "index": 0,
                "text": content_piece,
                "finish_reason": finish_reason
            }]
        }

        yield f"data: {json.dumps(chunk)}\n\n"

        if ollama_chunk.get("done") is True:
            # Final chunk — stop streaming
            final_chunk = {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": []
            }
            yield f"data: {json.dumps(final_chunk)}\n\n"
            yield "data: [DONE]\n\n"
            break


def handle_ollama_response(response, stream=False, is_chat=True):
    """
    Handles an Ollama response and converts it into either:
    - a single OpenAI-compatible JSON object (non-streaming), or
    - an iterable of SSE chunks (streaming).

    Args:
        response: `requests.Response` object from Ollama.
        stream (bool): Whether streaming was requested.

    Returns:
        dict | generator[str]: OpenAI-compatible response (full or streaming).
    """
    if stream:
        # Generator that yields OpenAI-style SSE chunks
        def stream_chunks():
            # Read the Flask Response iterable (bytes or str)
            for raw in response.response:
                # Ensure bytes are decoded
                if isinstance(raw, bytes):
                    raw = raw.decode("utf-8")
                raw = raw.strip()
                if raw:
                    # CHeck if cht or generate response
                    if is_chat:
                        yield from ollama_chat_stream_to_openai_chat_completions_chunks([raw])
                    else:
                        yield from ollama_generate_stream_to_openai_completions_chunks([raw])
                        
        return stream_chunks()
    else:
        # Full JSON response
        ollama_response = json.loads(response.get_data().decode("utf-8"))

        # CHeck if cht or generate response
        if is_chat:
            return jsonify(ollama_chat_to_openai_v1_chat_completion(ollama_response))
        else:
            return jsonify(ollama_generate_to_openai_v1_completion(ollama_response))



def handle_ollama_embedding_response(response):
    """
    Handles an Ollama response and converts it into a single OpenAI-compatible JSON object 

    Args:
        response: `requests.Response` object from Ollama.

    Returns:
        dict | generator[str]: OpenAI-compatible embedding esponse.
    """
    # Full JSON response
    ollama_response = json.loads(response.get_data().decode("utf-8"))

    # CHeck if cht or generate response
    return jsonify(ollama_embedding_to_openai_v1_embeddingns(ollama_response))



def openai_to_ollama_response(
    openai_response: Dict[str, Any],
    model: str = None,
) -> Dict[str, Any]:
    """
    Convert OpenAI Chat Completions response
    into Ollama /api/chat response format.

    Supports:
    - normal responses
    - tool calls
    - usage stats
    - multimodal text extraction
    """

    created = openai_response.get("created", int(time.time()))

    # ============================================================
    # BASIC FIELDS
    # ============================================================

    ollama_response = {
        "model": model or openai_response.get("model", ""),
        "created_at": time.strftime(
            "%Y-%m-%dT%H:%M:%S.000000Z",
            time.gmtime(created)
        ),
        "done": True,
    }

    # ============================================================
    # CHOICES
    # ============================================================

    choices = openai_response.get("choices", [])

    if not choices:
        ollama_response["message"] = {
            "role": "assistant",
            "content": ""
        }
        return ollama_response

    choice = choices[0]

    message = choice.get("message", {})

    role = message.get("role", "assistant")

    content = message.get("content")

    # ============================================================
    # MULTIMODAL CONTENT EXTRACTION
    # ============================================================

    #
    # OpenAI may return:
    #
    # "content": [
    #   {"type":"text","text":"hello"}
    # ]
    #

    if isinstance(content, list):

        extracted_text = []

        for item in content:

            if item.get("type") == "text":
                extracted_text.append(item.get("text", ""))

        content = "\n".join(extracted_text)

    if content is None:
        content = ""

    ollama_message = {
        "role": role,
        "content": content
    }

    # ============================================================
    # TOOL CALLS
    # ============================================================

    #
    # OpenAI:
    #
    # tool_calls: [
    #   {
    #       "id": "...",
    #       "type": "function",
    #       "function": {
    #           "name": "...",
    #           "arguments": "..."
    #       }
    #   }
    # ]
    #
    # Ollama:
    #
    # tool_calls: [
    #   {
    #       "function": {
    #           "name": "...",
    #           "arguments": {...}
    #       }
    #   }
    # ]
    #

    if message.get("tool_calls"):

        tool_calls = []

        for tc in message["tool_calls"]:

            function_data = tc.get("function", {})

            arguments = function_data.get("arguments", {})

            # OpenAI usually returns JSON string
            if isinstance(arguments, str):

                try:
                    import json
                    arguments = json.loads(arguments)
                except Exception:
                    pass

            tool_calls.append({
                "function": {
                    "name": function_data.get("name"),
                    "arguments": arguments
                }
            })

        ollama_message["tool_calls"] = tool_calls

    ollama_response["message"] = ollama_message

    # ============================================================
    # DONE REASON
    # ============================================================

    finish_reason = choice.get("finish_reason")

    if finish_reason:
        ollama_response["done_reason"] = finish_reason

    # ============================================================
    # USAGE TOKENS
    # ============================================================

    usage = openai_response.get("usage", {})

    if usage:

        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)

        ollama_response["prompt_eval_count"] = prompt_tokens
        ollama_response["eval_count"] = completion_tokens

    return ollama_response



def openai_stream_chunk_to_ollama(chunk: dict) -> dict:
    """
    Convert a single OpenAI streaming chunk
    into Ollama streaming format.
    """

    created = chunk.get("created", int(time.time()))

    choice = chunk.get("choices", [{}])[0]

    delta = choice.get("delta", {})

    finish_reason = choice.get("finish_reason")

    # ============================================================
    # CONTENT
    # ============================================================

    content = delta.get("content", "")

    # ============================================================
    # ROLE
    # ============================================================

    role = delta.get("role", "assistant")

    ollama_chunk = {
        "model": chunk.get("model", ""),
        "created_at": time.strftime(
            "%Y-%m-%dT%H:%M:%S.000000Z",
            time.gmtime(created)
        ),
        "message": {
            "role": role,
            "content": content
        },
        "done": finish_reason is not None
    }

    # ============================================================
    # TOOL CALLS
    # ============================================================

    if delta.get("tool_calls"):

        tool_calls = []

        for tc in delta["tool_calls"]:

            function_data = tc.get("function", {})

            arguments = function_data.get("arguments", "")

            # partial streamed arguments may not be valid JSON yet
            try:
                parsed_arguments = json.loads(arguments)
            except Exception:
                parsed_arguments = arguments

            tool_calls.append({
                "function": {
                    "name": function_data.get("name"),
                    "arguments": parsed_arguments
                }
            })

        ollama_chunk["message"]["tool_calls"] = tool_calls

    # ============================================================
    # FINISH REASON
    # ============================================================

    if finish_reason:
        ollama_chunk["done_reason"] = finish_reason

    return ollama_chunk


def strtobool (val):
    """Convert a string representation of truth to true (1) or false (0).
    True values are 'y', 'yes', 't', 'true', 'on', and '1'; false values
    are 'n', 'no', 'f', 'false', 'off', and '0'.  Raises ValueError if
    'val' is anything else.
    """
    val = val.lower()
    if val in ('y', 'yes', 't', 'true', 'on', '1'):
        return True
    elif val in ('n', 'no', 'f', 'false', 'off', '0'):
        return False
    else:
        raise ValueError("invalid truth value %r" % (val,))
    
################################## Tool Calls #####################################
def RawJSONDecoder(index):
    class _RawJSONDecoder(json.JSONDecoder):
        end = None
 
        def decode(self, s, *_):
            data, self.__class__.end = self.raw_decode(s, index)
            return data
    return _RawJSONDecoder
 
def extract_json_tools_from_text(s, index=0):
    while (index := s.find('{', index)) != -1:
        try:
            yield json.loads(s, cls=(decoder := RawJSONDecoder(index)))
            index = decoder.end
        except json.JSONDecodeError:
            index += 1


def get_tool_calls_generic(response):
    """ Return a list of formatted function calls by the LLM in the response.
        It a generic function to search any JSON response from any LLM with the required format:
        {"name": <function_name>, "parameters": <dictionary_of_argument_name_value>} 
        or
        {"name": <function_name>, "arguments": <dictionary_of_argument_name_value>} 
        For example:

        { "name": "get_current_weather", "arguments": { "location": "Paris, France", "format": "celsius" }

        Qwen models use <tool_call></tool_call> tags in chat template but for example Llama3.2 doesn't. That's why this generic implementation.


        Final response of a request must something like this: (https://github.com/ollama/ollama/blob/main/docs/api.md#chat-request-with-tools)

        {
            "model": "llama3.2",
            "created_at": "2024-07-22T20:33:28.123648Z",
            "message": {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                {
                    "function": {
                    "name": "get_current_weather",
                    "arguments": {
                        "format": "celsius",
                        "location": "Paris, FR"
                    }
                    }
                }
                ]
            },
            "done_reason": "stop",
            "done": true,
            "total_duration": 885095291,
            "load_duration": 3753500,
            "prompt_eval_count": 122,
            "prompt_eval_duration": 328493000,
            "eval_count": 33,
            "eval_duration": 552222000
        }


}
    """

    logger.debug(f"Searching tools with generic method: get_tool_calls_generic")

    # Get all the json objects
    json_tool_list = list(extract_json_tools_from_text(response))

    # Set the required keys in json object to identify tool calls
    required_keys_for_tools_option1 = set(["name", "arguments"]) # Other like Qwen
    required_keys_for_tools_option2 = set(["name", "parameters"]) # Llama default chat template
     
    tool_calls = []
    tool_calls += [{ "function": tool } for tool in json_tool_list if required_keys_for_tools_option1.issubset(tool.keys()) or required_keys_for_tools_option2.issubset(tool.keys())]

    # Rename the key "parameters" for "arguments" for standard
    tool_calls_renamed = []
    for tool in tool_calls:
      if "parameters" in tool["function"]:
          tool["function"]["arguments"] = tool["function"].pop("parameters")
      tool_calls_renamed.append(tool)
    return tool_calls_renamed


_TOOL_CALL_BLOCK_RE = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)
_XML_BLOCK_START_RE = re.compile(r"\s*<function=")
_XML_TOOL_CALL_RE = re.compile(r"<tool_call>\s*<function=")
_XML_FUNCTION_OPEN = "<function="
_XML_FUNCTION_CLOSE = "</function>"
_XML_PARAMETER_OPEN = "<parameter="
_XML_PARAMETER_CLOSE = "</parameter>"
_XML_NAME_RE = re.compile(r"([^>\n]*)>")
_XML_WHITESPACE_RE = re.compile(r"\s*")
_INTEGER_RE = re.compile(r"-?[0-9]+")
_JSON_NUMBER_RE = re.compile(r"-?(?:0|[1-9][0-9]*)(\.[0-9]+)?([eE][+-]?[0-9]+)?")
_NOT_CONVERTED = object()


def _strict_json_loads(text):
    """json.loads that rejects NaN / Infinity (they are not JSON and would make the response invalid JSON)."""
    def reject_constant(name):
        raise ValueError("non-finite number")

    def finite_float(literal):
        value = float(literal)
        if not math.isfinite(value):
            raise ValueError("non-finite number")
        return value

    return json.loads(text, parse_constant=reject_constant, parse_float=finite_float)


def _to_int(text):
    try:
        return int(text)
    except ValueError:   # more digits than int() accepts from a string (sys.set_int_max_str_digits)
        return _NOT_CONVERTED


def _convert_xml_value(text, type_name):
    """Convert the stripped text of a parameter to `type_name`, or return _NOT_CONVERTED."""
    if type_name == "integer":
        return _to_int(text) if _INTEGER_RE.fullmatch(text) else _NOT_CONVERTED
    if type_name == "number":
        match = _JSON_NUMBER_RE.fullmatch(text)
        if not match:
            return _NOT_CONVERTED
        if match.group(1) is None and match.group(2) is None:
            return _to_int(text)
        value = float(text)
        return value if math.isfinite(value) else _NOT_CONVERTED
    if type_name == "boolean":
        lowered = text.lower()
        if lowered in ("true", "false"):
            return lowered == "true"
        return _NOT_CONVERTED
    if type_name == "null":
        return None if text == "null" else _NOT_CONVERTED
    if type_name in ("object", "array"):
        try:
            value = _strict_json_loads(text)
        except (ValueError, RecursionError):
            return _NOT_CONVERTED
        expected = dict if type_name == "object" else list
        return value if isinstance(value, expected) else _NOT_CONVERTED
    return _NOT_CONVERTED


def _schema_types(prop_schema):
    """The type names a property schema allows, in the order they are tried: a single
    `type`, a list of types, or the `type` of each `anyOf` / `oneOf` entry (one level).
    "null" is tried first. A `$ref` or an unknown type counts as "string"."""
    if not isinstance(prop_schema, dict) or "$ref" in prop_schema:
        return ["string"]
    declared = prop_schema.get("type")
    if isinstance(declared, str):
        names = [declared]
    elif isinstance(declared, list):
        names = list(declared)
    else:
        names = []
        for key in ("anyOf", "oneOf"):
            options = prop_schema.get(key)
            if isinstance(options, list):
                for option in options:
                    if isinstance(option, dict) and "$ref" not in option and isinstance(option.get("type"), str):
                        names.append(option["type"])
                    else:
                        names.append("string")
                break
    known = ("string", "integer", "number", "boolean", "null", "object", "array")
    names = [name if name in known else "string" for name in names] or ["string"]
    if "null" in names:
        names = ["null"] + [name for name in names if name != "null"]
    return names


def _strip_one_newline(raw):
    """`raw` minus exactly one leading and one trailing line break ("\\r\\n" or "\\n")."""
    for newline in ("\r\n", "\n"):
        if raw.startswith(newline):
            raw = raw[len(newline):]
            break
    for newline in ("\r\n", "\n"):
        if raw.endswith(newline):
            raw = raw[:-len(newline)]
            break
    return raw


def coerce_xml_value(raw, prop_schema):
    """Turn the raw text of an XML tool-call parameter into the type its JSON schema
    asks for. Without a usable schema, or when the text does not convert, the value
    is a string: the raw text minus exactly one leading and one trailing line break
    ("\\n" or "\\r\\n"). Types that are not strings are converted from the fully stripped text."""
    as_string = _strip_one_newline(raw)
    stripped = raw.strip()
    for type_name in _schema_types(prop_schema):
        if type_name == "string":
            return as_string
        value = _convert_xml_value(stripped, type_name)
        if value is not _NOT_CONVERTED:
            return value
    return as_string


def _tool_parameter_schemas(tools, function_name):
    """The `properties` of the tool called `function_name` in the request's `tools`, or None."""
    if not isinstance(tools, list):
        return None
    for tool in tools:
        function = tool.get("function") if isinstance(tool, dict) else None
        if isinstance(function, dict) and function.get("name") == function_name:
            parameters = function.get("parameters")
            properties = parameters.get("properties") if isinstance(parameters, dict) else None
            return properties if isinstance(properties, dict) else None
    return None


def _read_xml_name(inner, pos):
    """Read `NAME>` at pos; return (stripped name, position after '>') or None for a bad name."""
    match = _XML_NAME_RE.match(inner, pos)
    if not match:
        return None
    name = match.group(1).strip()
    return (name, match.end()) if name else None


def _parse_xml_tool_call_strict(inner, tools):
    """Parse the text inside <tool_call>...</tool_call> in the XML form; return
    (call, None) or (None, reason). The grammar is strict: whitespace, one
    <function=NAME>, zero or more <parameter=PNAME>VALUE</parameter> separated only
    by whitespace, </function>, whitespace."""
    pos = _XML_WHITESPACE_RE.match(inner).end()
    if not inner.startswith(_XML_FUNCTION_OPEN, pos):
        return None, "no <function= at the start"
    function = _read_xml_name(inner, pos + len(_XML_FUNCTION_OPEN))
    if function is None:
        return None, "empty or invalid function name"
    function_name, pos = function
    schemas = _tool_parameter_schemas(tools, function_name) or {}
    arguments = {}
    while True:
        pos = _XML_WHITESPACE_RE.match(inner, pos).end()
        if inner.startswith(_XML_FUNCTION_CLOSE, pos):
            pos += len(_XML_FUNCTION_CLOSE)
            break
        if not inner.startswith(_XML_PARAMETER_OPEN, pos):
            return None, "text between elements or missing </function>"
        parameter = _read_xml_name(inner, pos + len(_XML_PARAMETER_OPEN))
        if parameter is None:
            return None, "empty or invalid parameter name"
        parameter_name, pos = parameter
        end = inner.find(_XML_PARAMETER_CLOSE, pos)
        if end == -1:
            return None, "missing </parameter>"
        arguments[parameter_name] = coerce_xml_value(inner[pos:end], schemas.get(parameter_name))
        pos = end + len(_XML_PARAMETER_CLOSE)
    if inner[pos:].strip():
        return None, "text after </function>"
    return {"function": {"name": function_name, "arguments": arguments}}, None


def parse_xml_tool_call(inner, tools=None):
    """Parse one Qwen3-Coder / Qwen3.5 XML tool call (the text inside <tool_call>...</tool_call>):

        <function=get_weather>
        <parameter=city>
        Paris
        </parameter>
        </function>

    Returns {"function": {"name": ..., "arguments": {...}}}, or None when the text does
    not follow the grammar exactly. Never raises: a block that cannot be parsed stays text.
    Argument types come from the matching tool's JSON schema in `tools` (see coerce_xml_value)."""
    try:
        call, reason = _parse_xml_tool_call_strict(inner, tools)
    except Exception as e:
        logger.warning(f"XML tool call parser failed ({type(e).__name__}); the block stays text")
        return None
    if call is None:
        logger.info(f"XML tool call block not parsed: {reason}; the block stays text")
    return call


def extract_tool_calls(response, tools=None):
    """Find the tool calls in a model answer.

    Returns (tool_calls, content, used_xml):
      - tool_calls: [{"function": {"name": ..., "arguments": {...}}}, ...] in text order
      - content: the answer with the <tool_call> blocks that gave a call removed, trimmed
      - used_xml: True when at least one call came from the XML form

    A <tool_call> block whose text starts with <function= is parsed by the XML parser only;
    every other block by the JSON code (get_tool_calls_generic). When no block gives a call,
    the whole answer is searched for JSON calls, except when it holds an XML block: a
    parameter value of a broken XML block can look like a JSON call."""
    logger.debug("Searching tool calls in <tool_call> blocks: extract_tool_calls")
    tool_calls = []
    used_xml = False
    kept_text = []
    last_end = 0
    for block in _TOOL_CALL_BLOCK_RE.finditer(response):
        inner = block.group(1)
        if _XML_BLOCK_START_RE.match(inner):
            call = parse_xml_tool_call(inner, tools)
            block_calls = [call] if call else []
            used_xml = used_xml or bool(call)
        else:
            # JSON form; words around the JSON are allowed, e.g.
            # <tool_call> Output : {"name": <function_name>, "arguments": <dictionary_of_argument_name_value>} </tool_call>
            block_calls = get_tool_calls_generic(inner)
        if block_calls:
            tool_calls += block_calls
            kept_text.append(response[last_end:block.start()])
            last_end = block.end()
    kept_text.append(response[last_end:])

    if not tool_calls and not _XML_TOOL_CALL_RE.search(response):
        # No standard format tool call found. Search for more generic way
        tool_calls = get_tool_calls_generic(response)

    return tool_calls, "".join(kept_text).strip(), used_xml


def get_tool_calls(response, tools=None):
    """ Get all the tool calls indicated by the LLM in the response """
    tool_calls, _, _ = extract_tool_calls(response, tools)
    return tool_calls


def get_base64_image_from_pil(image: Image.Image, output_format) -> str:
    """Convert a PIL Image to a base64-encoded string.
    Args:
        image (PIL.Image.Image): The PIL Image to convert.
        output_format (str): The desired output format, either "png" or "jpg".
    Returns:
        str: The base64-encoded string representation of the image.
    """
    # Save image to a bytes buffer
    buffered = io.BytesIO()

    # Save in PNG format 
    image.save(buffered, format=output_format)

    # Get byte data
    img_bytes = buffered.getvalue()

    # Encode to base64
    return base64.b64encode(img_bytes).decode("utf-8")


def get_url_image_from_pil(image: Image.Image, model_name, output_dir, output_format) -> str:
    """Convert a PIL Image to a base64-encoded string.
    Args:
        image (PIL.Image.Image): The PIL Image to convert.
        output_dir (str): The directory to save the image file.
        output_format (str): The desired output format, either "png" or "jpg".
    Returns:
        str: The url string representation of the image.
    """

    # Create output dir if not exists
    os.makedirs(output_dir, exist_ok=True)

    # Save image to dir file
    file_name= f"out_image_{int(time.time())}.{output_format.lower()}"
    out_path = f"{output_dir}/{file_name}"

    # Save the image to a file
    image.save(out_path) 

    # Get port from config
    port = rkllama.config.get("server", "port", "8080")

    # Encode to base64
    return f"http://localhost:{port}/files/{model_name}/images/{file_name}"



class OpenAIToOllamaStreamConverter:

    def __init__(self):
        self.model = None
        self.created = None

        # Tool call accumulation
        self.tool_calls = {}

        # Avoid duplicate done event
        self.sent_done = False

    def _timestamp(self):
        return time.strftime(
            "%Y-%m-%dT%H:%M:%S.000000Z",
            time.gmtime(self.created or int(time.time()))
        )

    def _base_chunk(self):
        return {
            "model": self.model or "",
            "created_at": self._timestamp(),
        }

    def process_line(self, line: str):
        """
        Input:
            data: {...}

        Output:
            list[dict]

        Usage:

            for line in response.iter_lines(decode_unicode=True):
                for chunk in converter.process_line(line):
                    yield json.dumps(chunk) + "\\n"
        """

        if not line:
            return []

        line = line.strip()

        if not line.startswith("data:"):
            return []

        payload = line[5:].strip()

        #
        # End of SSE stream
        #

        if payload == "[DONE]":

            if self.sent_done:
                return []

            self.sent_done = True

            return [{
                **self._base_chunk(),
                "message": {
                    "role": "assistant",
                    "content": ""
                },
                "done": True
            }]

        try:
            chunk = json.loads(payload)
        except Exception:
            return []

        self.model = chunk.get("model", self.model)
        self.created = chunk.get("created", self.created)

        choices = chunk.get("choices")

        if not choices:
            return []

        choice = choices[0]

        delta = choice.get("delta", {})
        finish_reason = choice.get("finish_reason")

        results = []

        #
        # ------------------------------------------------------
        # CONTENT STREAMING
        # ------------------------------------------------------
        #

        content = delta.get("content")

        if content not in (None, ""):

            results.append({
                **self._base_chunk(),
                "message": {
                    "role": "assistant",
                    "content": content
                },
                "done": False
            })

        #
        # ------------------------------------------------------
        # REASONING STREAMING
        # ------------------------------------------------------
        #

        reasoning = (
            delta.get("reasoning_content")
            or delta.get("reasoning")
            or delta.get("thinking")
        )

        if reasoning not in (None, ""):

            results.append({
                **self._base_chunk(),
                "message": {
                    "role": "assistant",
                    "thinking": reasoning
                },
                "done": False
            })

        #
        # ------------------------------------------------------
        # TOOL CALL ACCUMULATION
        # ------------------------------------------------------
        #

        if "tool_calls" in delta:

            for tc in delta["tool_calls"]:

                idx = tc.get("index", 0)

                if idx not in self.tool_calls:
                    self.tool_calls[idx] = {
                        "name": "",
                        "arguments": ""
                    }

                function_data = tc.get("function", {})

                if "name" in function_data:
                    self.tool_calls[idx]["name"] += (
                        function_data["name"] or ""
                    )

                if "arguments" in function_data:
                    self.tool_calls[idx]["arguments"] += (
                        function_data["arguments"] or ""
                    )

        #
        # ------------------------------------------------------
        # TOOL CALL FINISHED
        # ------------------------------------------------------
        #

        if finish_reason == "tool_calls":

            ollama_tool_calls = []

            for idx in sorted(self.tool_calls.keys()):

                tool = self.tool_calls[idx]

                arguments = tool["arguments"]

                try:
                    arguments = json.loads(arguments)
                except Exception:
                    pass

                ollama_tool_calls.append({
                    "function": {
                        "name": tool["name"],
                        "arguments": arguments
                    }
                })

            results.append({
                **self._base_chunk(),
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": ollama_tool_calls
                },
                "done": False
            })

            self.tool_calls.clear()

            return results

        #
        # ------------------------------------------------------
        # NORMAL COMPLETION
        # ------------------------------------------------------
        #

        if (
            finish_reason is not None
            and finish_reason != "tool_calls"
        ):

            results.append({
                **self._base_chunk(),
                "message": {
                    "role": "assistant",
                    "content": ""
                },
                "done": True,
                "done_reason": finish_reason
            })

            self.sent_done = True

        return results

def ollama_to_openai_chat(req):
    req = copy.deepcopy(req)

    openai_req = {
        "model": req["model"]
    }

    if "stream" in req:
        openai_req["stream"] = req["stream"]

    # --------------------------------------------------
    # TOOLS
    # --------------------------------------------------

    if req.get("tools"):
        openai_req["tools"] = req["tools"]

    # --------------------------------------------------
    # OPTIONS
    # --------------------------------------------------

    options = req.get("options", {})

    mappings = {
        "temperature": "temperature",
        "top_p": "top_p",
        "seed": "seed",
        "frequency_penalty": "frequency_penalty",
        "presence_penalty": "presence_penalty",
        "stop": "stop",
    }

    for ollama_key, openai_key in mappings.items():
        if ollama_key in options:
            openai_req[openai_key] = options[ollama_key]

    if "num_predict" in options:
        openai_req["max_tokens"] = options["num_predict"]

    if req.get("format") == "json":
        openai_req["response_format"] = {
            "type": "json_object"
        }

    # --------------------------------------------------
    # MESSAGES
    # --------------------------------------------------

    messages = []

    if req.get("system"):
        messages.append({
            "role": "system",
            "content": req["system"]
        })

    pending_tool_call_ids = []

    for msg in req.get("messages", []):

        role = msg.get("role", "user")

        # ==================================================
        # TOOL MESSAGE
        # ==================================================

        if role == "tool":

            #
            # Only valid if we previously saw
            # structured tool_calls.
            #

            if pending_tool_call_ids:

                tool_call_id = pending_tool_call_ids.pop(0)

                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "content": msg.get("content", "")
                })

            else:

                #
                # Cannot reconstruct OpenAI tool_call_id.
                # Degrade to plain text.
                #

                messages.append({
                    "role": "user",
                    "content":
                        "Tool result:\n\n" +
                        msg.get("content", "")
                })

            continue

        converted = {
            "role": role
        }

        # ==================================================
        # STRUCTURED TOOL CALLS
        # ==================================================

        if msg.get("tool_calls"):

            openai_tool_calls = []

            for idx, tc in enumerate(msg["tool_calls"]):

                fn = tc.get("function", {})

                tool_call_id = f"call_{idx}"

                pending_tool_call_ids.append(tool_call_id)

                openai_tool_calls.append({
                    "id": tool_call_id,
                    "type": "function",
                    "function": {
                        "name": fn.get("name"),
                        "arguments": json.dumps(
                            fn.get("arguments", {})
                        )
                    }
                })

            converted["tool_calls"] = openai_tool_calls

            converted["content"] = msg.get("content")

            messages.append(converted)
            continue

        # ==================================================
        # MULTIMODAL
        # ==================================================

        if msg.get("images"):

            content = []

            text = msg.get("content")

            if text:
                content.append({
                    "type": "text",
                    "text": text
                })

            for image in msg["images"]:

                if image.startswith("data:"):
                    url = image
                else:
                    url = (
                        "data:image/jpeg;base64,"
                        + image
                    )

                content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": url
                    }
                })

            converted["content"] = content

        else:

            converted["content"] = msg.get(
                "content",
                ""
            )

        messages.append(converted)

    openai_req["messages"] = messages

    return openai_req
