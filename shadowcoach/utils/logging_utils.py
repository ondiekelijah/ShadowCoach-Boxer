"""
Logging utilities for the ShadowCoach system
"""
import logging
import time
import functools
from typing import Dict, List, Callable, Any

# Dictionary to store function execution times
function_timings: Dict[str, List[float]] = {}

def setup_logging(level: int = logging.INFO) -> logging.Logger:
    """
    Configure and return a logger for the ShadowCoach system
    
    Args:
        level: The logging level to use
        
    Returns:
        A configured logger instance
    """
    # Configure logging
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )
    return logging.getLogger('ShadowCoach')

# Create a default logger
logger = setup_logging()

def timed(func: Callable) -> Callable:
    """
    Decorator to measure and log function execution time
    
    Args:
        func: The function to time
        
    Returns:
        Wrapped function that logs execution time
    """
    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        func_name = func.__name__
        start_time = time.time()
        logger.debug(f"Starting {func_name}")
        
        result = func(*args, **kwargs)
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        # Store timing information
        if func_name not in function_timings:
            function_timings[func_name] = []
        function_timings[func_name].append(execution_time)
        
        logger.debug(f"Completed {func_name} in {execution_time:.2f} seconds")
        return result
    return wrapper

def print_timing_summary(total_time: float) -> None:
    """
    Print a summary of function execution times
    
    Args:
        total_time: The total execution time
    """
    logger.info("\n=== Performance Summary ===")
    logger.info(f"Total execution time: {total_time:.2f} seconds")
    
    # Print timing for each function
    logger.info("\nDetailed Function Timing:")
    for func_name, times in function_timings.items():
        # Calculate statistics
        count = len(times)
        total = sum(times)
        avg = total / count if count > 0 else 0
        max_time = max(times) if times else 0
        min_time = min(times) if times else 0
        
        # Print detailed timing information
        logger.info(f"  {func_name}:")
        logger.info(f"    Calls: {count}")
        logger.info(f"    Total time: {total:.2f}s")
        logger.info(f"    Average time: {avg:.2f}s")
        logger.info(f"    Min/Max: {min_time:.2f}s / {max_time:.2f}s")
        logger.info(f"    Percentage of total: {(total/total_time)*100:.1f}%")
