import pytest
import json
import re


ADVERSARIAL_PAYLOADS = [
    # Prototype pollution via __proto__
    '{"__proto__": {"polluted": true}}',
    # Constructor injection
    '{"constructor": {"prototype": {"polluted": true}}}',
    # Type field injection (protobufjs specific)
    '{"type": "constructor"}',
    '{"type": "__proto__"}',
    '{"type": "Object"}',
    # Nested type injection
    '{"fields": {"x": {"type": "constructor"}}}',
    '{"fields": {"x": {"type": "__proto__"}}}',
    # Arbitrary code execution via type fields
    '{"nested": {"Evil": {"fields": {"x": {"type": "constructor.prototype"}}}}}',
    # JSON with injected protobuf definition type fields
    '{"options": {"(evil)": "constructor"}}',
    # Null byte injection
    '{"type": "string\x00constructor"}',
    # Unicode escape injection
    '{"type": "\\u005f\\u005fproto\\u005f\\u005f"}',
    # Deep nesting attack
    '{"a": {"b": {"c": {"d": {"type": "constructor"}}}}}',
    # Array with malicious type
    '{"fields": [{"type": "constructor"}, {"type": "__proto__"}]}',
    # Mixed case bypass attempts
    '{"type": "Constructor"}',
    '{"type": "__PROTO__"}',
    '{"type": "CONSTRUCTOR"}',
    # Whitespace bypass
    '{"type": " constructor "}',
    '{"type": "\tconstructor\t"}',
    # Dot notation injection
    '{"type": "Object.prototype"}',
    '{"type": "Function.prototype"}',
    # Eval-like strings
    '{"type": "eval"}',
    '{"type": "Function"}',
    # Empty and boundary values
    '{"type": ""}',
    '{"type": null}',
    '{"type": 0}',
    '{"type": []}',
    # Deeply nested prototype pollution
    '{"a": {"__proto__": {"polluted": "yes"}}}',
    # toString/valueOf override attempts
    '{"toString": "constructor"}',
    '{"valueOf": "__proto__"}',
]


def is_safe_protobuf_type(type_value: str) -> bool:
    """
    Check if a protobuf type field value is safe.
    Safe types are standard protobuf scalar types or valid message names
    that don't reference dangerous JavaScript properties.
    """
    DANGEROUS_IDENTIFIERS = {
        '__proto__', 'constructor', 'prototype', 'eval',
        'function', 'object', 'Function', 'Object',
        '__defineGetter__', '__defineSetter__', '__lookupGetter__',
        '__lookupSetter__', 'hasOwnProperty', 'isPrototypeOf',
        'propertyIsEnumerable', 'toString', 'valueOf',
    }

    VALID_PROTOBUF_SCALAR_TYPES = {
        'double', 'float', 'int32', 'int64', 'uint32', 'uint64',
        'sint32', 'sint64', 'fixed32', 'fixed64', 'sfixed32', 'sfixed64',
        'bool', 'string', 'bytes',
    }

    if not isinstance(type_value, str):
        return True  # Non-string types are handled separately

    stripped = type_value.strip()

    # Check against dangerous identifiers (case-insensitive)
    if stripped.lower() in {d.lower() for d in DANGEROUS_IDENTIFIERS}:
        return False

    # Check for prototype chain access patterns
    if re.search(r'(prototype|__proto__|constructor)', stripped, re.IGNORECASE):
        return False

    # Check for dot notation that could access dangerous properties
    if '.' in stripped:
        parts = stripped.split('.')
        for part in parts:
            if part.lower() in {d.lower() for d in DANGEROUS_IDENTIFIERS}:
                return False

    return True


def extract_type_fields(obj, path=""):
    """Recursively extract all 'type' field values from a parsed JSON object."""
    type_fields = []
    if isinstance(obj, dict):
        for key, value in obj.items():
            current_path = f"{path}.{key}" if path else key
            if key == 'type' and isinstance(value, str):
                type_fields.append((current_path, value))
            elif isinstance(value, (dict, list)):
                type_fields.extend(extract_type_fields(value, current_path))
    elif isinstance(obj, list):
        for i, item in enumerate(obj):
            type_fields.extend(extract_type_fields(item, f"{path}[{i}]"))
    return type_fields


def check_prototype_pollution(obj, path=""):
    """Check if parsed JSON object contains prototype pollution keys."""
    DANGEROUS_KEYS = {'__proto__', 'constructor', 'prototype'}
    violations = []
    if isinstance(obj, dict):
        for key, value in obj.items():
            current_path = f"{path}.{key}" if path else key
            if key in DANGEROUS_KEYS:
                violations.append((current_path, key))
            if isinstance(value, (dict, list)):
                violations.extend(check_prototype_pollution(value, current_path))
    elif isinstance(obj, list):
        for i, item in enumerate(obj):
            violations.extend(check_prototype_pollution(item, f"{path}[{i}]"))
    return violations


@pytest.mark.parametrize("payload", ADVERSARIAL_PAYLOADS)
def test_protobuf_type_field_injection_safety(payload):
    """
    Invariant: Protobuf definition type fields must never reference dangerous
    JavaScript properties (constructor, __proto__, prototype, eval, etc.)
    that could lead to arbitrary code execution or prototype pollution.
    Any parsed protobuf definition must have safe type field values and
    must not contain prototype pollution keys.
    """
    # Attempt to parse the payload as JSON (simulating protobuf definition parsing)
    try:
        parsed = json.loads(payload)
    except (json.JSONDecodeError, ValueError):
        # If payload is not valid JSON, it cannot be a valid protobuf definition
        # This is safe - invalid input is rejected
        return

    # INVARIANT 1: No prototype pollution keys should be present at any level
    pollution_violations = check_prototype_pollution(parsed)
    assert len(pollution_violations) == 0, (
        f"Prototype pollution keys found in payload: {pollution_violations}. "
        f"Payload: {payload}"
    )

    # INVARIANT 2: All 'type' fields must be safe (no dangerous identifiers)
    type_fields = extract_type_fields(parsed)
    for field_path, type_value in type_fields:
        assert is_safe_protobuf_type(type_value), (
            f"Dangerous type field value '{type_value}' found at path '{field_path}'. "
            f"This could lead to arbitrary code execution via injected protobuf "
            f"definition type fields. Payload: {payload}"
        )

    # INVARIANT 3: Type fields must not contain null bytes
    for field_path, type_value in type_fields:
        assert '\x00' not in type_value, (
            f"Null byte found in type field at path '{field_path}': {repr(type_value)}. "
            f"Payload: {payload}"
        )

    # INVARIANT 4: Type fields must not contain prototype chain access patterns
    PROTOTYPE_CHAIN_PATTERN = re.compile(
        r'(prototype|__proto__|constructor)',
        re.IGNORECASE
    )
    for field_path, type_value in type_fields:
        assert not PROTOTYPE_CHAIN_PATTERN.search(type_value), (
            f"Prototype chain access pattern found in type field at path "
            f"'{field_path}': '{type_value}'. "
            f"This violates the security boundary. Payload: {payload}"
        )