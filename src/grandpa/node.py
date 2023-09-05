from grandpa.routing import Router, Switch


class Node:
    def __init__(
        self,
        name: str,
        router: Router,
        executable_func: callable,
        call_args: list,
        call_kwargs: dict,
        required_arg_nodes: list,
        required_kwarg_nodes: dict,
    ):
        self.executable_func = executable_func
        self.call_args = call_args
        self.call_kwargs = call_kwargs
        self.required_arg_nodes = required_arg_nodes
        self.required_kwarg_nodes = required_kwarg_nodes
        self.address = name + str(id(self))
        self.switch = Switch(self.address, router, self)

    def __call__(self, call_method: str = None):
        args = []
        kwargs = {}
        for node in self.required_arg_nodes:
            if node[1]:
                args.append(self.switch(*node))
            else:
                node = self.switch(node[0], True)
                args.append(self.switch.add_task(node))
        for key, node in self.required_kwarg_nodes.items():
            if node[1]:
                kwargs[key] = self.switch(*node)
            else:
                node = self.switch(node[0], True)
                kwargs[key] = self.switch.add_task(node)
        for i in range(len(args)):
            if type(args[i]) == int:
                args[i] = self.switch.get_task_result(args[i])
        for key, value in kwargs.items():
            if type(value) == int:
                kwargs[key] = self.switch.get_task_result(value)
        args.extend(self.call_args)
        kwargs.update(self.call_kwargs)
        return self.executable_func(*args, **kwargs)
