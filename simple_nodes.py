import math

from grandpa import Node, Component, Workflow


@Node("sum_values")
def sum_values(arr):
    total = 0
    for item in arr:
        total += item
    return total


@Node("subtract_values")
def subtract_values(a, b):
    return a - b


@Node("multiply_values")
def multiply_values(a, b):
    return a * b


@Node("divide_values")
def divide_values(a, b):
    return a / b


@Node("round_down")
def round_down(a):
    return math.floor(a)


@Component("remainder")
def remainder(a, b):
    divide_value = divide_values(a, b)
    rounded_down = round_down(divide_value)
    multiplied = multiply_values(rounded_down, b)
    return subtract_values(a=a, b=multiplied)


@Workflow("remainder_workflow")
def remainder_workflow():
    remainder_value = remainder(23, 10)
    return remainder_value

