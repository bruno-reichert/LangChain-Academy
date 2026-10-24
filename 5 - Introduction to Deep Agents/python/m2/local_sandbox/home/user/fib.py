def fibonacci(n):
    """Generate the first n Fibonacci numbers."""
    fib = [0, 1]
    for i in range(2, n):
        fib.append(fib[i-1] + fib[i-2])
    return fib[:n]

if __name__ == "__main__":
    fib_numbers = fibonacci(15)
    print("First 15 Fibonacci numbers:")
    for i, num in enumerate(fib_numbers):
        print(f"F({i}) = {num}")