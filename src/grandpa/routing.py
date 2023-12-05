from .task import TaskID


class Switch:
    """
    Used to connect nodes to each other.
    """

    def __init__(self, address: str, router, node):
        self.address = address
        self.router = router
        self.node = node
        self.router.register(self, address)

    def __call__(
        self, address: str, return_node: bool = False, call_method: str = None, call_args: tuple = None,
            call_kwargs: dict = None
    ):
        """
        Connects to another node. Returns the result of the node if return_node is False, else returns the node itself.
        You can also specify a method to call on the node.
        Args:
            address: Address of the node to connect to.
            return_node: Whether to return the node or its result.
            call_method: Method to call on the node.

        Returns:
            The result of the node if return_node is False, else the node itself.
        """
        if address == self.address:
            if return_node:
                return self.node
            else:
                if call_args is None:
                    call_args = ()
                if call_kwargs is None:
                    call_kwargs = {}
                return self.node(call_method=call_method, *call_args, **call_kwargs)
        else:
            node_switch = self.router.get_switch(address)
            return node_switch(address, return_node, call_method, call_args, call_kwargs)

    def add_task(self, target: callable, *args, **kwargs) -> TaskID:
        """
        Adds a task to the multiprocessing manager.
        Args:
            target: Function to run.
            *args: Arguments to pass to the function.
            **kwargs: Keyword arguments to pass to the function.

        Returns:
            The task ID.
        """
        return self.router.add_task(target, *args, **kwargs)

    def get_task_result(self, task_id: TaskID):
        """
        Gets the result of a task.
        Args:
            task_id: ID of the task.

        Returns:
            The result of the task.
        """
        return self.router.get_task_result(task_id)


class Router(Switch):
    """
    Connects to different systems (not implemented). Also handles multiprocessing on the local system.
    """
    def __init__(self, multiprocessing_manager):
        self.address_table = {}
        self.multiprocessing_manager = multiprocessing_manager
        super().__init__(None, self, None)

    def get_switch(self, address: str) -> Switch:
        """
        Gets the switch for a given address.
        Args:
            address: Address of the switch.

        Returns:
            The switch.
        """
        if address not in self.address_table:
            raise RoutingError(address)
        return self.address_table[address]

    def register(self, switch: Switch, address: str):
        """
        Registers a switch to an address.
        Args:
            switch: Switch to register.
            address: Address to register the switch to.

        Returns:
            None
        """
        self.address_table[address] = switch

    def add_task(self, target: callable, *args, **kwargs) -> TaskID:
        """
        Adds a task to the multiprocessing manager.
        Args:
            target: Function to run.
            *args: Arguments to pass to the function.
            **kwargs: Keyword arguments to pass to the function.

        Returns:
            The task ID.
        """
        return self.multiprocessing_manager.add_task(target, *args, **kwargs)

    def get_task_result(self, task_id: TaskID):
        """
        Gets the result of a task.
        Args:
            task_id: ID of the task.

        Returns:
            The result of the task.
        """
        return self.multiprocessing_manager.get_task_result(task_id)


class RoutingError(Exception):
    """
    Raised when a route does not exist.
    """
    def __init__(self, route):
        self.message = f'Route {route} does not exist - it might not have been created.'
        super().__init__(self.message)
