class Node:
    def __init__(
        self,
        name,
        executable_func: callable,
        call_args,
        call_kwargs,
        required_arg_nodes,
        required_kwarg_nodes,
    ):
        self.name = name
        self.executable_func = executable_func
        self.call_args = call_args
        self.call_kwargs = call_kwargs
        self.required_arg_nodes = required_arg_nodes
        self.required_kwarg_nodes = required_kwarg_nodes

    def __call__(self):
        args = []
        kwargs = {}
        for node in self.required_arg_nodes:
            args.append(node())
        for key, node in self.required_kwarg_nodes.items():
            kwargs[key] = node()
        args.extend(self.call_args)
        kwargs.update(self.call_kwargs)
        return self.executable_func(*args, **kwargs)
