import re


def tokenize(text: str) -> list[str]:
    """
    Tokenizer for scientific text.
    Used for both BM25 index building and query tokenization.
    Must be identical in both — single source of truth.

    - Lowercases
    - Splits hyphens: self-attention → self attention
    - Splits on non-alphanumeric characters
    - Removes single char tokens except 'a'
    """
    text = text.lower()
    text = text.replace("-", " ")
    tokens = re.split(r'[^a-z0-9]+', text)
    tokens = [t for t in tokens if len(t) > 1 or t == 'a']
    return tokens