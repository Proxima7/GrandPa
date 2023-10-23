import inspect
from multiprocessing import current_process

from grandpa.template_classes import (ComponentTemplate, FuncTemplate,
                                      NodeTemplate, PipelineTemplate)


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
            raise RuntimeError(
                "Node decorator can only be used on classes or functions."
            )


class Component:
    def __init__(self, name: str, *args, **kwargs):
        self.name = name
        self.args = args
        self.kwargs = kwargs

    def __call__(self, f):
        if inspect.isfunction(f):
            return ComponentTemplate(f, self.name)
        else:
            raise RuntimeError("Component decorator can only be used on functions.")


class Workflow:
    def __init__(self, name: str, *args, **kwargs):
        self.name = name
        self.args = args
        self.kwargs = kwargs

    def __call__(self, f):
        if inspect.isfunction(f):
            return PipelineTemplate(f, self.name)
        else:
            raise RuntimeError("Pipeline decorator can only be used on functions.")
