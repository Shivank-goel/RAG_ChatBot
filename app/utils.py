# app/utils.py
def identity_chunk(text: str):
    return [text] if text else []
