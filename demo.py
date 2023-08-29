from grandpa.decorators import Node


@Node("add")
def add():
    return 1 + 2


@Node("subtract")
def subtract(a):
    return a - 2


@Node("test")
class TestNode:
    def __init__(self, value_a, value_b):
        self.value_a = value_a
        self.value_b = value_b

    def __call__(self, value_c):
        return self.value_a + self.value_b + value_c


def test_func():
    a = add()
    b = subtract(a)
    c = TestNode(a, 15)
    d = c(b)
    print()


if __name__ == "__main__":
    test_func()
