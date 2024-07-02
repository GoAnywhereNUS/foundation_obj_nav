"""
String utilities for formatting queries and responses to/from foundation models
"""

def generic_string_format(s: str):
    return (
        s.lower().replace(' ', '').replace('.', '')
    )