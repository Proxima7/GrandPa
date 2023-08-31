from grandpa.decorators import Component, Node, Pipeline, Generator
from grandpa.template_parser import TemplateParser
from random import Random

@Generator("generator")
@Node("random_number")
def gen_random_from_node():
    return Random().randint(0, 100)

@Node("add")
def add(a, b):
    return a + b

@Node("subtract")
def subtract(a, b):
    return a - b


@Node("multiply")
def multiply(a, b):
    return a * b


@Node("divide")
def divide(a, b):
    return a / b


@Node("round")
def round_(a):
    return round(a)


@Component("even_number")
def even_number(a):
    divide_by_two = divide(a, 2)
    round_divide_by_two = round_(divide_by_two)
    multiply_by_two = multiply(round_divide_by_two, 2)
    return multiply_by_two


@Node("test")
class TestNode:
    def __init__(self, value_a, value_b):
        self.value_a = value_a
        self.value_b = value_b

    def __call__(self, value_c):
        return self.value_a + self.value_b + value_c


@Pipeline("test_pipeline")
def test_pipeline():
    a = add(1, 2)
    b = even_number(a)
    c = TestNode(b, 3)
    d = c(4)

    return d
@Pipeline("gen_pipe")
def test_generators():
    g = gen_random_from_node()
    print(f"rnd from node {g}")
    return g


if __name__ == "__main__":
    pipeline = test_pipeline()
    parser = TemplateParser()
    pipeline_result = parser(pipeline)

    gen_pipe = test_generators()
    gen_pipe_result = parser(gen_pipe)

    print()
