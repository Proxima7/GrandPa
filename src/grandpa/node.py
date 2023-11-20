from grandpa.routing import Router, Switch


class Node:
    """
    Node is a wrapper for a function or a class. It will gather all required arguments and keyword arguments
    from other nodes, and then run the function or class it wraps.
    """
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
        self.address = name
        self.switch = Switch(self.address, router, self)

    def __call__(self, call_method: str = None):
        """
        Main execution method for Node. Will first load all required arguments and keyword arguments asynchronously,
        then run the function or class it wraps.
        Args:
            call_method: (Not implemented) Specify a different function to run (only for wrapped classes)

        Returns:
            The result of the function or class it wraps.
        """
        args = self.__load_args()
        kwargs = self.__load_kwargs()
        self.__finish_tasks_for_args(args)
        self.__finish_tasks_for_kwargs(kwargs)
        args.extend(self.call_args)
        kwargs.update(self.call_kwargs)
        return self.executable_func(*args, **kwargs)

    def __finish_tasks_for_kwargs(self, kwargs: dict):
        """
        Waits for all active tasks in kwargs to finish, and replaces them with their results.
        Args:
            kwargs: Dictionary of keyword arguments to check.

        Returns:
            None
        """
        for key, value in kwargs.items():
            if type(value) == int:
                kwargs[key] = self.switch.get_task_result(value)

    def __finish_tasks_for_args(self, args: list):
        """
        Waits for all active tasks in args to finish, and replaces them with their results.
        Args:
            args: List of arguments to check.

        Returns:
            None
        """
        for i in range(len(args)):
            if type(args[i]) == int:
                args[i] = self.switch.get_task_result(args[i])

    def __load_kwargs(self) -> dict:
        """
        Loads all keyword arguments. Node[1] is a bool which specifies if the argument is a Node object
        (-> pull the node object from the switch) or the result of a Node (-> create a Task to execute the Node).
        Returns:
            Dictionary of keyword arguments with required results replaced by their task ids.
        """
        kwargs = {}
        for key, node in self.required_kwarg_nodes.items():
            if node[1]:
                kwargs[key] = self.switch(*node)
            else:
                kwargs[key] = self.switch.add_task(node[0])
        return kwargs

    def __load_args(self) -> list:
        """
        Loads all arguments. Node[1] is a bool which specifies if the argument is a Node object
        (-> pull the node object from the switch) or the result of a Node (-> create a Task to execute the Node).
        Returns:
            List of arguments with required results replaced by their task ids.
        """
        args = []
        for node in self.required_arg_nodes:
            if node[1]:
                args.append(self.switch(*node))
            else:
                args.append(self.switch.add_task(node[0]))
        return args
