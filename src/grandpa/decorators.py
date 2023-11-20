import inspect
from typing import Union

from grandpa.template_classes import (ComponentTemplate, FuncTemplate,
                                      NodeTemplate, PipelineTemplate)


class Node:
    """
    Decorator for node class
    """
    def __init__(self, name: str, *args, **kwargs):
        self.name = name
        self.args = args
        self.kwargs = kwargs

    def __call__(self, f: callable) -> Union[NodeTemplate, FuncTemplate]:
        """
        Node decorator call method.
        Args:
            f: Function or class to apply decorator to

        Returns:
            NodeTemplate if f is a class, FuncTemplate if f is a function, else raises RuntimeError
        """
        if inspect.isclass(f):
            return NodeTemplate(f, self.name)
        elif inspect.isfunction(f):
            return FuncTemplate(f, self.name)
        else:
            raise RuntimeError(
                "Node decorator can only be used on classes or functions."
            )


class Component:
    """
    Decorator for component class
    """
    def __init__(self, name: str, *args, **kwargs):
        self.name = name
        self.args = args
        self.kwargs = kwargs

    def __call__(self, f: callable) -> ComponentTemplate:
        """
        Component decorator call method.
        Args:
            f: Function to apply decorator to

        Returns:
            ComponentTemplate if f is a function, else raises RuntimeError
        """
        if inspect.isfunction(f):
            return ComponentTemplate(f, self.name)
        else:
            raise RuntimeError("Component decorator can only be used on functions.")


class Workflow:
    """
    Decorator for workflow function
    """
    def __init__(self, name: str, *args, **kwargs):
        self.name = name
        self.args = args
        self.kwargs = kwargs

    def __call__(self, f: callable) -> PipelineTemplate:
        """
        Workflow decorator call method.
        Args:
            f: Function to apply decorator to

        Returns:
            PipelineTemplate if f is a function, else raises RuntimeError
        """
        if inspect.isfunction(f):
            return PipelineTemplate(f, self.name)
        else:
            raise RuntimeError("Pipeline decorator can only be used on functions.")
