from math import sqrt
import time
from langchain.tools import tool

@tool
def summing(x: int, y: int) -> int: 
    """sum of two integer"""
    print(f"[TOOL CALLED] summing({x}, {y})")
    return x + y

@tool
def minus(x: int, y: int) -> int:
    """minus of two integer"""
    return x - y

@tool
def square_root(x):
    """Calculate the square root of x"""
    return sqrt(x) 

@tool
def daytime():
    """
    Return the daytime
    """
    return time.time()