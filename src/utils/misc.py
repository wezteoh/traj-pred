def linear_schedule(p_start: float, p_end: float, step: int, total_steps: int) -> float:
    # bound output to be between 0 and 1
    return max(0, min(1, p_start + (p_end - p_start) * step / total_steps))
