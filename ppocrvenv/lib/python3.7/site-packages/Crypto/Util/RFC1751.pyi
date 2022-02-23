from typing import Dict, List

binary: Dict[int, str]
wordlist: List[str]

def key_to_english(key: bytes) -> str: ...
def english_to_key(s: str) -> bytes: ...
