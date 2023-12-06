from .routing import Router, Switch
from .task import TaskID


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

    def __call__(self, *args, call_method: str = None, **kwargs):
        """
        Main execution method for Node. Will first load all required arguments and keyword arguments asynchronously,
        then run the function or class it wraps.
        Args:
            call_method: (Not implemented) Specify a different function to run (only for wrapped classes)

        Returns:
            The result of the function or class it wraps.
        """
        if call_method is None:
            args = self._load_args(args)
            kwargs = self._load_kwargs(kwargs)
            self._finish_tasks_for_args(args)
            self._finish_tasks_for_kwargs(kwargs)
            args.extend(self.call_args)
            kwargs.update(self.call_kwargs)
            return self.executable_func(*args, **kwargs)
        else:
            call_func = getattr(self.executable_func, call_method)
            return call_func(*args, **kwargs)

    def __getattribute__(self, item):
        """
        Used to get attributes from the executable_func. Especially important if the executable_func is a class.
        Args:
            item: Attribute to get.

        Returns:
            The attribute from the executable_func.
        """
        # Using object.__getattribute__ to avoid infinite recursion
        if item != 'executable_func' and item != '__call__' \
                and hasattr(object.__getattribute__(self, 'executable_func'), item):
            return getattr(object.__getattribute__(self, 'executable_func'), item)
        else:
            return object.__getattribute__(self, item)

    def __getattr__(self, item):
        """
        Used to get attributes from the executable_func. Especially important if the executable_func is a class.
        Args:
            item: Attribute to get.

        Returns:
            The attribute from the executable_func.
        """
        return getattr(self.executable_func, item)

    def __len__(self):
        """
        Returns:
            The length of the executable_func.
        """
        return len(self.executable_func)

    def __getitem__(self, item):
        """
        Returns:
            The item from the executable_func.
        """
        return self.executable_func[item]

    def __next__(self):
        """
        Returns:
            The next item from the executable_func.
        """
        return next(self.executable_func)

    def __iter__(self):
        """
        Returns:
            An iterator for the executable_func.
        """
        return iter(self.executable_func)

    def _finish_tasks_for_kwargs(self, kwargs: dict):
        """
        Waits for all active tasks in kwargs to finish, and replaces them with their results.
        Args:
            kwargs: Dictionary of keyword arguments to check.

        Returns:
            None
        """
        for key, value in kwargs.items():
            if isinstance(value, TaskID):
                kwargs[key] = self.switch.get_task_result(value)

    def _finish_tasks_for_args(self, args: list):
        """
        Waits for all active tasks in args to finish, and replaces them with their results.
        Args:
            args: List of arguments to check.

        Returns:
            None
        """
        for i in range(len(args)):
            if isinstance(args[i], TaskID):
                args[i] = self.switch.get_task_result(args[i])

    def _load_kwargs(self, kwargs: dict = None) -> dict:
        """
        Loads all keyword arguments. Node[1] is a bool which specifies if the argument is a Node object
        (-> pull the node object from the switch) or the result of a Node (-> create a Task to execute the Node).
        Args:
            kwargs: Dictionary of keyword arguments already passed with call.

        Returns:
            Dictionary of keyword arguments with required results replaced by their task ids.
        """
        if kwargs is None:
            kwargs = {}
        for key, node in self.required_kwarg_nodes.items():
            if node[1]:
                kwargs[key] = self.switch(*node)
            else:
                kwargs[key] = self.switch.add_task(node[0])
        return kwargs

    def _load_args(self, args: tuple = None) -> list:
        """
        Loads all arguments. Node[1] is a bool which specifies if the argument is a Node object
        (-> pull the node object from the switch) or the result of a Node (-> create a Task to execute the Node).
        Args:
            args: Tuple of arguments already passed with call.

        Returns:
            List of arguments with required results replaced by their task ids.
        """
        if args is None:
            args = []
        else:
            args = list(args)
        for node in self.required_arg_nodes:
            if node[1]:
                args.append(self.switch(*node))
            else:
                args.append(self.switch.add_task(node[0]))
        return args


class TaskNode(Node):
    """
    TaskNode will execute a Node multiple times. The count of execution is determined dynamically be the amount of
    entries per argument/keyword argument. The count must be the same for all arguments/keyword arguments. This is only
    the case for the __call__ method, all other methods will behave like a normal Node.
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
        target_address: str,
    ):
        super().__init__(name, router, executable_func, call_args, call_kwargs, required_arg_nodes,
                         required_kwarg_nodes)
        self.target_address = target_address

    @staticmethod
    def get_param_subset_count(args: list, kwargs: dict):
        """
        Returns the number of elements in the first parameter of the TaskNode. This determines how often target address
        will be executed.
        Args:
            args: Arguments passed to TaskNode
            kwargs: Keyword arguments passed to TaskNode

        Returns:
            Number of elements in first parameter of TaskNode
        """
        if len(args) > 0:
            return len(args[0])
        elif len(kwargs) > 0:
            return len(kwargs[list(kwargs.keys())[0]])
        else:
            raise RuntimeError("No parameters found for TaskNode. TaskNode requires at least one parameter.")

    @staticmethod
    def get_param_subset(index, args, kwargs):
        """
        Returns a subset of the arguments and keyword arguments of the TaskNode. This subset is determined by the index
        of the first parameter of the TaskNode.
        Args:
            index: Index of the subset to return
            args: Arguments passed to TaskNode
            kwargs: Keyword arguments passed to TaskNode

        Returns:
            Subset of arguments and keyword arguments
        """
        try:
            subset_args = []
            for arg in args:
                subset_args.append(arg[index])
            subset_kwargs = {}
            for key, value in kwargs.items():
                subset_kwargs[key] = value[index]
            return subset_args, subset_kwargs
        except IndexError:
            raise RuntimeError("Index out of range for TaskNode. Please ensure that all parameters have the same "
                               "length.")

    def __call__(self, *args, call_method: str = None, **kwargs):
        """
        Main execution method for Node. Will first load all required arguments and keyword arguments asynchronously,
        then run the function or class it wraps.
        Args:
            call_method: (Not implemented) Specify a different function to run (only for wrapped classes)

        Returns:
            The result of the function or class it wraps.
        """
        if call_method is None:
            args = self._load_args(args)
            kwargs = self._load_kwargs(kwargs)
            self._finish_tasks_for_args(args)
            self._finish_tasks_for_kwargs(kwargs)
            args.extend(self.call_args)
            kwargs.update(self.call_kwargs)
            tasks = []
            for i in range(self.get_param_subset_count(args, kwargs)):
                subset_args, subset_kwargs = self.get_param_subset(i, args, kwargs)
                tasks.append(self.switch.add_task(self.target_address, *subset_args, **subset_kwargs))
            return [self.switch.get_task_result(task) for task in tasks]
        else:
            call_func = getattr(self.executable_func, call_method)
            return call_func(*args, **kwargs)
