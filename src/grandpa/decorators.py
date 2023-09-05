import inspect

from grandpa.template_classes import (ComponentTemplate, FuncTemplate,
                                      NodeTemplate, PipelineTemplate)
from multiprocessing import current_process


class Node:
    def __init__(self, name: str, *args, **kwargs):
        self.name = name
        self.args = args
        self.kwargs = kwargs

    def __call__(self, f):
        if hasattr(current_process(), "is_grandpa_process") and current_process().is_grandpa_process:
            return f
        elif inspect.isclass(f):
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
        if hasattr(current_process(), "is_grandpa_process") and current_process().is_grandpa_process:
            return f
        elif inspect.isfunction(f):
            return ComponentTemplate(f, self.name)
        else:
            raise RuntimeError("Component decorator can only be used on functions.")


class Pipeline:
    def __init__(self, name: str, *args, **kwargs):
        self.name = name
        self.args = args
        self.kwargs = kwargs

    def __call__(self, f):
        if hasattr(current_process(), "is_grandpa_process") and current_process().is_grandpa_process:
            return f
        elif inspect.isfunction(f):
            return PipelineTemplate(f, self.name)
        else:
            raise RuntimeError("Pipeline decorator can only be used on functions.")
