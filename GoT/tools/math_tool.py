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
def sum_three(x: int, y: int, z: int):
    """sum of three integer
    
    Arguments:
        x(int): first number
        y(int): second number
        z(int): third number
    """
    return x + y + z


@tool
def sum_four(x, y, z, a):
    """sum of four integer
    
    Arguments:
        x(int): first number
        y(int): second number
        z(int): third number
        a(int): fourth number
    """
    return x + y + z + a


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
def square_root(x: int) -> float:
    """Calculate the square root of x
    
    Arguments:
        x(int): a non-negative number
    """
    return sqrt(x)
