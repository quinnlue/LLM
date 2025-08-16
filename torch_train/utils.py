import time


def calculate_steps_per_sec(current_step: int, start_time: float) -> float:
    elapsed_time = time.perf_counter() - start_time
    return current_step / elapsed_time if elapsed_time > 0 else 0.0


