def register_params(template, args, kwargs):
    for arg in args:
        if any(
            [
                isinstance(arg, InitialisedNodeTemplate),
                isinstance(arg, ResultWrapper),
                isinstance(arg, FuncTemplate),
            ]
        ):
            template.required_arg_nodes.append(arg)
        elif isinstance(arg, NodeTemplate):
            raise RuntimeError(
                "NodeTemplate must be initialised before being passed as an argument"
            )
        else:
            template.call_args.append(arg)
    for key, value in kwargs.items():
        if any(
            [
                isinstance(value, InitialisedNodeTemplate),
                isinstance(value, ResultWrapper),
                isinstance(value, FuncTemplate),
            ]
        ):
            template.required_kwarg_nodes[key] = value
        elif isinstance(value, NodeTemplate):
            raise RuntimeError(
                "NodeTemplate must be initialised before being passed as an argument"
            )
        else:
            template.call_kwargs[key] = value


class ResultWrapper:
    def __init__(self, origin):
        self.origin = origin


class FuncTemplate:
    def __init__(self, function: callable, name: str):
        self.function = function
        self.name = name
        self.required_arg_nodes = []
        self.call_args = []
        self.required_kwarg_nodes = {}
        self.call_kwargs = {}

    def __call__(self, *args, **kwargs):
        register_params(self, args, kwargs)
        return ResultWrapper(self)


class InitialisedNodeTemplate:
    def __init__(self, node_template):
        self.node_template = node_template
        self.required_arg_nodes = []
        self.call_args = []
        self.required_kwarg_nodes = {}
        self.call_kwargs = {}

    def __call__(self, *args, **kwargs):
        register_params(self, args, kwargs)
        return ResultWrapper(self)


class NodeTemplate:
    def __init__(self, cls: object, name: str):
        self.cls = cls
        self.name = name
        self.required_arg_nodes = []
        self.call_args = []
        self.required_kwarg_nodes = {}
        self.call_kwargs = {}

    def __call__(self, *args, **kwargs):
        register_params(self, args, kwargs)
        return InitialisedNodeTemplate(self)


class ComponentTemplate:
    def __init__(self, component_func: callable, name: str):
        self.component_func = component_func
        self.name = name

    def __call__(self, *args, **kwargs):
        return self.component_func(*args, **kwargs)


class PipelineTemplate:
    def __init__(self, pipeline_func: callable, name: str):
        self.pipeline_func = pipeline_func
        self.name = name

    def __call__(self, *args, **kwargs):
        return self.pipeline_func(*args, **kwargs)

class GeneratorTemplate():
    def __init__(self, node: callable, name: str, cache_size=10, cache_access_type="QUEUE"):
        self.node = node
        self.name = name
        self.cache_size = cache_size
        self.cache_access_type = cache_access_type
        self.queue = []

    def __call__(self, *args, **kwargs):
        return self.node(*args, **kwargs)