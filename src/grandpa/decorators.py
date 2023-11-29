import inspect
from typing import Union

from grandpa.template_classes import (ComponentTemplate, FuncTemplate,
                                      NodeTemplate, PipelineTemplate)


class Node:
    """
    Decorator for node class
    """
    def __init__(self, name: str, pass_task_executor: bool = False, *args, **kwargs):
        self.name = name
        self.pass_task_executor = pass_task_executor
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
        if hasattr(self, "__grandpa_task_node__"):
            grandpa_task_node = True
        else:
            grandpa_task_node = False
        if inspect.isclass(f):
            f.__decorated_by_grandpa_node__ = True
            f.__grandpa_node_address__ = self.name
            return NodeTemplate(f, self.name, self.pass_task_executor, grandpa_task_node=grandpa_task_node)
        elif inspect.isfunction(f):
            return FuncTemplate(f, self.name, self.pass_task_executor, grandpa_task_node=grandpa_task_node)
        else:
            raise RuntimeError(
                "Node decorator can only be used on classes or functions."
            )


class TaskNode:
    """
    Decorator for task node class. Will execute the underlying Node multiple times based on a list of arguments.
    """
    def __init__(self, name: str, *args, **kwargs):
        self.name = name
        self.args = args
        self.kwargs = kwargs

    def __call__(self, f: callable) -> NodeTemplate:
        """
        TaskNode decorator call method.
        Args:
            f: Function or class to apply decorator to

        Returns:
            NodeTemplate if f is a class, else raises RuntimeError
        """
        underlying_node = Node(self.name, False, *self.args, **self.kwargs)
        underlying_node.__grandpa_task_node__ = True
        return underlying_node(f)


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
