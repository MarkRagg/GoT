from math import sqrt
from langchain.tools import tool


@tool
def summing(x: float, y: float) -> float:
    """sum of two float numbers

    Arguments:
        x(float): first number
        y(float): second number
    """
    return x + y

@tool
def minus(x: float, y: float) -> float:
    """minus of two float numbers

    Arguments:
        x(float): first number
        y(float): second number
    """
    return x - y


@tool
def multiply(x: float, y: float) -> float:
    """multiply of two float numbers

    Arguments:
        x(float): first number
        y(float): second number
    """
    return x * y

@tool
def divide(x: float, y: float) -> float:
    """divide of two float numbers

    Arguments:
        x(float): first number
        y(float): second number
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
