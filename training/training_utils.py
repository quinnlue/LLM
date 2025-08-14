import collections
import time
from typing import Tuple, Deque

class RunningLossTracker:
    """Efficiently tracks running loss averages using sliding windows"""
    
    def __init__(self, window_sizes: Tuple[int, int, int] = (100, 1000, 10000)):
        self.window_100, self.window_1k, self.window_10k = window_sizes
        
        # Deques for storing values
        self.loss_window_100 = collections.deque(maxlen=self.window_100)
        self.loss_window_1k = collections.deque(maxlen=self.window_1k)
        self.loss_window_10k = collections.deque(maxlen=self.window_10k)
        
        # Running sums for efficient calculation
        self.sum_100 = 0.0
        self.sum_1k = 0.0
        self.sum_10k = 0.0
        
        # Counters for current window sizes
        self.count_100 = 0
        self.count_1k = 0
        self.count_10k = 0
    
    def update(self, loss_value: float):
        """Update running losses efficiently by adding new value and removing oldest if needed"""
        # Update 100-window
        if len(self.loss_window_100) == self.window_100:
            # Remove oldest value from sum
            self.sum_100 -= self.loss_window_100[0]
        else:
            self.count_100 += 1
        
        self.loss_window_100.append(loss_value)
        self.sum_100 += loss_value
        
        # Update 1K-window
        if len(self.loss_window_1k) == self.window_1k:
            # Remove oldest value from sum
            self.sum_1k -= self.loss_window_1k[0]
        else:
            self.count_1k += 1
        
        self.loss_window_1k.append(loss_value)
        self.sum_1k += loss_value
        
        # Update 10K-window
        if len(self.loss_window_10k) == self.window_10k:
            # Remove oldest value from sum
            self.sum_10k -= self.loss_window_10k[0]
        else:
            self.count_10k += 1
        
        self.loss_window_10k.append(loss_value)
        self.sum_10k += loss_value
    
    def get_running_losses(self) -> Tuple[float, float, float]:
        """Get current running loss averages"""
        running_100_loss = self.sum_100 / self.count_100 if self.count_100 > 0 else float('inf')
        running_1k_loss = self.sum_1k / self.count_1k if self.count_1k > 0 else float('inf')
        running_10k_loss = self.sum_10k / self.count_10k if self.count_10k > 0 else float('inf')
        
        return running_100_loss, running_1k_loss, running_10k_loss

def format_fraction(current: int, total: int) -> str:
    """Format fraction as percentage with 3 decimal places"""
    if total == 0:
        return "0.000%"
    percentage = (current / total) * 100
    return f"{percentage:.3f}%"

def calculate_steps_per_sec(current_step: int, start_time: float) -> float:
    """Calculate steps per second"""
    elapsed_time = time.perf_counter() - start_time
    return current_step / elapsed_time if elapsed_time > 0 else 0.0

class ProgressBarManager:
    """Manages progress bar updates and formatting"""
    
    def __init__(self, total_steps: int, start_time: float):
        self.total_steps = total_steps
        self.start_time = start_time
        self.current_step = 0
        self.loss_tracker = RunningLossTracker()
    
    def update_progress(self, loss_value: float, optimizer_lr: float, pbar) -> None:
        """Update progress bar with current metrics"""
        self.current_step += 1
        self.loss_tracker.update(loss_value)
        
        # Get metrics
        running_100_loss, running_1k_loss, running_10k_loss = self.loss_tracker.get_running_losses()
        steps_per_sec = calculate_steps_per_sec(self.current_step, self.start_time)
        fraction = format_fraction(self.current_step, self.total_steps)
        
        # Update progress bar
        pbar.set_postfix(
            loss=f"{running_100_loss:.4f}",
            ema_1k=f"{running_1k_loss:.4f}",
            ema_10k=f"{running_10k_loss:.4f}",
            lr=f"{optimizer_lr:.6f}",
            speed=f"{steps_per_sec:.2f} it/s",
            frac=fraction
        )
        pbar.update(1)
