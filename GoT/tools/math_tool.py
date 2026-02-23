from math import sqrt
from langchain.tools import tool


@tool
def summing(x: int, y: int) -> int:
    """sum of two integer

    Arguments:
        x(int): first number
        y(int): second number
    """
    return x + y

@tool
def minus(x: int, y: int) -> int:
    """minus of two integer

    Arguments:
        x(int): first number
        y(int): second number
    """
    return x - y


@tool
def multiply(x: int, y: int) -> int:
    """multiply of two integer

    Arguments:
        x(int): first number
        y(int): second number
    """
    return x * y

@tool
def divide(x: int, y: int) -> float:
    """divide of two integer

    Arguments:
        x(int): first number
        y(int): second number
    """
    if y == 0:
        raise ValueError("Cannot divide by zero")
    return x / y


@tool
def square_root(x: int) -> float:
    """Calculate the square root of x

    Arguments:
        x(int): a non-negative number
    """
    return sqrt(x)
