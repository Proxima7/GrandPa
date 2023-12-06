import inspect
import random
import uuid

from grandpa.sync import get_sync_params


def register_params(template: callable, args: tuple, kwargs: dict):
    """
    Registers args and kwargs for execution to a template.
    Args:
        template: Template to register the args and kwargs to.
        args: Args to register.
        kwargs: Kwargs to register.

    Returns:
        None
    """
    set_args(args, template)
    set_kwargs(kwargs, template)


def set_kwargs(kwargs: dict, template: callable):
    """
    Sets kwargs to a template. If the kwarg is a NodeTemplate, ResultWrapper or FuncTemplate, it is registered as a
    required node. Otherwise, it is registered as a call kwarg.
    Args:
        kwargs: Kwargs to register.
        template: Template to register the kwargs to.

    Returns:
        None
    """
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


def set_args(args: tuple, template: callable):
    """
    Sets args to a template. If the arg is a NodeTemplate, ResultWrapper or FuncTemplate, it is registered as a
    required node. Otherwise, it is registered as a call arg.
    Args:
        args: Args to register.
        template: Template to register the args to.

    Returns:
        None
    """
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


class ResultWrapper:
    """
    Wrapper for a FuncTemplate or InitialisedNodeTemplate. This is used to indicate that the expected value is the
    result of a InitialisedNodeTemplate or FuncTemplate, rather than the template itself.
    """
    def __init__(self, origin):
        self.origin = origin


class FuncTemplate:
    """
    Template to wrap a function.
    """
    def __init__(self, function: callable, name: str, pass_task_executor: bool = False,
                 grandpa_task_node: bool = False):
        self.function = function
        self.name = name
        self.required_arg_nodes = []
        self.call_args = []
        self.required_kwarg_nodes = {}
        self.call_kwargs = {}
        self.pass_task_executor = pass_task_executor
        self.grandpa_task_node = grandpa_task_node

    def __call__(self, *args, **kwargs) -> ResultWrapper:
        """
        Indicates the function should be called with the given args and kwargs. Creates a new FuncTemplate with the
        given args and kwargs registered, since the original FuncTemplate should not be modified (this creates issues
        when multiple calls are made to the same function).
        Args:
            *args: Args to call the function with.
            **kwargs: Kwargs to call the function with.

        Returns:
            ResultWrapper indicating the result of the function call.
        """
        loc_id = uuid.uuid4()
        loc_func_temp = FuncTemplate(self.function, self.name + str(loc_id), self.pass_task_executor, self.grandpa_task_node)
        register_params(loc_func_temp, args, kwargs)
        return ResultWrapper(loc_func_temp)


class InitialisedNodeTemplate:
    """
    Template to wrap a class object. This is used to indicate that the class should be initialised with the given args
    and kwargs.
    """
    def __init__(self, node_template, grandpa_task_node: bool = False):
        self.node_template = node_template
        self.name = self.node_template.name + "_init"
        self.required_arg_nodes = []
        self.call_args = []
        self.required_kwarg_nodes = {}
        self.call_kwargs = {}
        self.pass_task_executor = False
        self.grandpa_task_node = grandpa_task_node

    def __call__(self, *args, **kwargs) -> ResultWrapper:
        """
        Indicates the class object should be called with the given args and kwargs. Creates a new
        InitialisedNodeTemplate with the given args and kwargs registered, since the original InitialisedNodeTemplate
        should not be modified (this creates issues when multiple calls are made to the same class object).
        Args:
            *args: Args to call the class object with.
            **kwargs: Kwargs to call the class object with.

        Returns:
            ResultWrapper indicating the result of the class object call.
        """
        loc_id = uuid.uuid4()
        loc_node_temp = NodeTemplate(self.node_template.cls, self.node_template.name + str(loc_id))
        loc_node_temp.required_arg_nodes = self.node_template.required_arg_nodes
        loc_node_temp.call_args = self.node_template.call_args
        loc_node_temp.required_kwarg_nodes = self.node_template.required_kwarg_nodes
        loc_node_temp.call_kwargs = self.node_template.call_kwargs
        loc_init_node_temp = InitialisedNodeTemplate(loc_node_temp, self.grandpa_task_node)
        register_params(loc_init_node_temp, args, kwargs)
        return ResultWrapper(loc_init_node_temp)


class NodeTemplate:
    """
    Template to wrap a class. This is used to indicate that the class is a Node and can be instantiated.
    """
    def __init__(self, cls, name: str, pass_task_executor: bool = False, grandpa_task_node: bool = False):
        self.cls = cls
        self.name = name
        self.required_arg_nodes = []
        self.call_args = []
        self.required_kwarg_nodes = {}
        self.call_kwargs = {}
        self.pass_task_executor = pass_task_executor
        self.grandpa_task_node = grandpa_task_node

    def __call__(self, *args, **kwargs) -> InitialisedNodeTemplate:
        """
        Instantiates the class with the given args and kwargs. Creates a new NodeTemplate with the given
        args and kwargs registered, since the original NodeTemplate should not be modified (this creates issues when
        instantiating multiple objects of the same class).
        Args:
            *args: Args to instantiate the class with.
            **kwargs: Kwargs to instantiate the class with.

        Returns:
            InitialisedNodeTemplate indicating the instantiated class.
        """
        loc_id = uuid.uuid4()
        loc_node_temp = NodeTemplate(self.cls, self.name + str(loc_id), self.pass_task_executor, self.grandpa_task_node)
        register_params(loc_node_temp, args, kwargs)
        loc_node_temp.sync_params = get_sync_params(self.cls.__init__)
        return InitialisedNodeTemplate(loc_node_temp, grandpa_task_node=self.grandpa_task_node)


class ComponentTemplate:
    """
    Template to wrap a component function, which creates a DAG of multiple nodes.
    """
    def __init__(self, component_func: callable, name: str):
        self.component_func = component_func
        self.name = name

    def __call__(self, *args, **kwargs):
        """
        Calls the component function with the given args and kwargs.
        Args:
            *args: Args to call the component function with.
            **kwargs: Kwargs to call the component function with.

        Returns:
            Last Node of the DAG created by the component function as a template.
        """
        return self.component_func(*args, **kwargs)


class PipelineTemplate:
    """
    Template to wrap a pipeline function, which creates a DAG of multiple nodes which can be executed py template
    parser.
    """
    def __init__(self, pipeline_func: callable, name: str):
        self.pipeline_func = pipeline_func
        self.name = name

    def __call__(self, *args, **kwargs):
        """
        Calls the pipeline function with the given args and kwargs.
        Args:
            *args: Args to call the pipeline function with (not implemented).
            **kwargs: Kwargs to call the pipeline function with (not implemented).

        Returns:
            Last Node of the DAG created by the pipeline function as a template.
        """
        return self.pipeline_func(*args, **kwargs)
