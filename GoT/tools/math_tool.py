from math import sqrt
from langchain.tools import tool


@tool
def summing(x: int, y: int) -> int:
    """sum of two integer"""
    return x + y


@tool
def sum_three(x: int, y: int, z: int):
    """sum of three integer"""
    return x + y + z


@tool
def sum_four(x, y, z, a):
    """sum of four integer"""
    return x + y + z + a


@tool
def minus(x: int, y: int) -> int:
    """minus of two integer"""
    return x - y


@tool
def square_root(x):
    """Calculate the square root of x"""
    return sqrt(x)
