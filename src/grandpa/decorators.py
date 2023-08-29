import inspect

from grandpa.template_classes import NodeTemplate, FuncTemplate


class Node:
    def __init__(self, name: str, *args, **kwargs):
        self.name = name
        self.args = args
        self.kwargs = kwargs

    def __call__(self, f):
        if inspect.isclass(f):
            return NodeTemplate(f, self.name)
        elif inspect.isfunction(f):
            return FuncTemplate(f, self.name)
        else:
            raise RuntimeError("Node decorator can only be used on classes or functions.")
