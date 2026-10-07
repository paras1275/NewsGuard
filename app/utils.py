def validate_news_request(data: dict) -> tuple[bool, str]:
    """
    Validates API payload for news prediction.
    Returns (is_valid: bool, error_message: str).
    """
    if not isinstance(data, dict):
        return False, "Invalid JSON payload format. Expected JSON object."

    if "text" not in data:
        return False, "Missing required field 'text' in JSON body."

    text = data.get("text")
    if text is None or not isinstance(text, str):
        return False, "Field 'text' must be a valid non-null string."

    stripped = text.strip()
    if not stripped:
        return False, "Article text cannot be empty."

    word_count = len(stripped.split())
    if word_count < 5:
        return False, f"Article text is too short ({word_count} words). Minimum 5 words required for credible NLP analysis."

    return True, ""
