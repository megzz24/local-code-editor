def add(a, b):
    """Add two numbers together."""
    return a + b

def subtract(a, b):
    """Subtract b from a."""
    return a - b

def multiply(a, b):
    """Multiply two numbers."""
    return a * b

def divide(a, b):
    """Divide a by b. Raises ValueError if b is zero."""
    return a / b

def factorial(n):
    """Return the factorial of n recursively."""
    if n <= 1:
        return 1
    return n * factorial(n - 1)