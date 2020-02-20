def truncate_str(v: str, max_length: int = 80) -> str:
    """Trim the given text if too long."""
    if len(v) <= max_length:
        return v
    head = max_length * 2 // 3
    tail = max_length - head - 3
    return v[:head] + "..." + v[-tail:]
