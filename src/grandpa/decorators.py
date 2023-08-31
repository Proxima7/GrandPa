import inspect

from grandpa.template_classes import (ComponentTemplate, FuncTemplate,
                                      NodeTemplate, PipelineTemplate, GeneratorTemplate)


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


class Pipeline:
    def __init__(self, name: str, *args, **kwargs):
        self.name = name
        self.args = args
        self.kwargs = kwargs

    def __call__(self, f):
        if inspect.isfunction(f):
            return PipelineTemplate(f, self.name)
        else:
            raise RuntimeError("Pipeline decorator can only be used on functions.")


class Generator:
    """
    Generate data until the cache_size is full. Specify where the data is kept: "RAM", "DISC"
    """
    def __init__(self, name: str, *args, **kwargs):
        self.name = name
        self.args = args
        self.kwargs = kwargs

    def __call__(self, f):
        # already wrapped supported node types
        if type(f) in [NodeTemplate, FuncTemplate, ComponentTemplate]:
            return GeneratorTemplate(f, self.name, *self.args, **self.kwargs)

        # try to wrap it
        # check types of f and wrap it into a node if not yet happened.
        if inspect.isfunction(f):
            wrapped_node = FuncTemplate(f, self.name)
        elif inspect.isclass(f):
            wrapped_node = NodeTemplate(f, self.name)
        else:
            raise RuntimeError("Generator decorator can only be used on nodes, functions, classes.")

        return GeneratorTemplate(wrapped_node, self.name, *self.args, **self.kwargs)

