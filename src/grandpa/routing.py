from grandpa.multiprocessing_manager import MultiprocessingManager


class Switch:
    """
    Used to connect nodes to each other.
    """

    def __init__(self, address, router, node):
        self.address = address
        self.router = router
        self.node = node
        self.router.register(self, address)

    def __call__(
        self, address: str, return_node: bool = False, call_method: str = None
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
                return self.node(call_method=call_method)
        else:
            node_switch = self.router.get_switch(address)
            return node_switch(address, return_node, call_method)

    def add_task(self, target, *args, **kwargs):
        return self.router.add_task(target, *args, **kwargs)

    def get_task_result(self, task_id):
        return self.router.get_task_result(task_id)


class Router(Switch):
    def __init__(self, multiprocessing_manager: MultiprocessingManager):
        self.address_table = {}
        self.multiprocessing_manager = multiprocessing_manager
        super().__init__(None, self, None)

    def get_switch(self, address):
        if address not in self.address_table:
            raise RoutingError(address)
        return self.address_table[address]

    def register(self, switch, address):
        self.address_table[address] = switch

    def add_task(self, target, *args, **kwargs):
        return self.multiprocessing_manager.add_task(target, *args, **kwargs)

    def get_task_result(self, task_id):
        return self.multiprocessing_manager.get_task_result(task_id)


class RoutingError(Exception):
    def __init__(self, route):
        self.message = f'Route {route} does not exist - it might not have been created.'
        super().__init__(self.message)
