import re
from typing import Union

from psutil import virtual_memory

from grandpa.multiprocessing_manager import (MultiprocessingManager, Task,
                                             WorkerQueue)


class Switch:
    """
    Used to connect nodes to each other.
    """

    def __init__(self, address, node, main_router=None):
        self.address = address
        self.factory = node
        self.default_gateway = (
            main_router.get_parent(address) if address and main_router else None
        )
        self.address_table = {}
        self.main_router = main_router
        if address:
            if self.default_gateway:
                self.default_gateway.register(self.factory, self.address)
            else:
                if main_router:
                    main_router.register(self.factory, self.address)

    def execute_task(self, target, *args, **kwargs) -> Task:
        """
        Adds a task to execute by the MultiProcessingManager.

        Args:
            target: Callable target.
            *args: Call args.
            **kwargs: Call kwargs.

        Returns:
            Task Object, that can be used to get the result.
        """
        return self.main_router.execute_task(target, *args, **kwargs)

    def add_worker_queue(
        self,
        target,
        target_size: int = int(virtual_memory().total / 1000000000),
        split_results: bool = False,
        *args,
        **kwargs,
    ) -> WorkerQueue:
        """
        Adds a worker queue to the MultiProcessingManager.

        Args:
            target: Callable target of the queue.
            target_size: Maximum size of the queue.
            split_results: If True, results will be split (e.g., ['A', 'B', 'C'] would add 3 entries to the queue.
            *args: Args to use when calling target.
            **kwargs: Kwargs to use when calling target.

        Returns:
            Worker Queue object.
        """
        return self.main_router.add_worker_queue(
            target,
            target_size=target_size,
            split_results=split_results,
            *args,
            **kwargs,
        )

    def register(self, target, name=None):
        if name:
            self.address_table[f'{self.address}/{name}'] = target
        else:
            self.address_table[f'{self.address}/{target.name}'] = target

    def delete_object_at_path(self, path):
        target = self.get_instruction(path)
        del target
        if path in self.address_table:
            del self.address_table[path]

    def register_foreign(self, path, target):
        self.address_table[path] = target

    def route(self, path, return_value, call_trace: list = None):
        """
        Public call for the route function.

        Args:
            path: Path to route to.
            return_value: Whether to return the value of the object (True) or the object stored at this location.
            call_trace: The switch / router this function is called at will share its address table with the
            routers / switches in this list.

        Returns:
            The object or the object value stored at the location given by the path.
        """
        if call_trace:
            for switch in call_trace:
                self.__share_address_table(switch)
        return self.__route(path, return_value)

    def __share_address_table(self, switch):
        for address in self.address_table:
            if address not in switch.address_table:
                switch.register_foreign(address, self.__route(address, False))

    def __route(self, path: str, return_value):
        """
        Routes the request. Returns the requested value if the switch can directly access it,
        or forwards the request to the next switch in the direction of the target.

        Args:
            path: Path of the requested value.
            return_value: Whether to return the value of the object (True) or the object stored at this location.

        Returns:
            The object or the object value stored at the location given by the path.
        """
        if path.startswith('//'):
            path = path[1:]
        if path in self.address_table:
            return self.__get_routing_target(path, return_value)
        elif self.__get_closest_switch(path):
            return self._forward_to_switch(
                path, return_value, self.__get_closest_switch(path)
            )
        else:
            if self.default_gateway:
                return self.default_gateway.route(path, return_value)
            else:
                raise RoutingError(route=path)

    def _forward_to_switch(self, path, return_value, switch):
        if type(switch) is Switch:
            return switch.route(path, return_value)
        elif type(switch) is dict:
            return self.__resolve_dict_path(path)

    def __resolve_dict_path(self, path):
        path = path.replace(self.address, '')[1:]
        resolve_dict = self.get_value(self.address + '/' + path.split('/')[0])
        in_dict_path = path.replace(path.split('/')[0], '')[1:]
        for key in in_dict_path.split('/'):
            resolve_dict = resolve_dict[key]
        return resolve_dict

    def __get_closest_switch(self, path: str):
        matching_switches = [
            self.address_table[s_path]
            for s_path in self.address_table
            if re.match(s_path + '/.*', path)
        ]
        return max(matching_switches) if matching_switches else None

    def __get_routing_target(self, path, return_value):
        """
        Returns the value or object at the path location.

        Args:
            path: Path where the object is stored.
            return_value: Whether to return the object or the object value.

        Returns:
            The object or the object value stored at the location given by the path.
        """
        if return_value and callable(self.address_table[path]):
            return self.address_table[path]()
        else:
            return self.address_table[path]

    def get_instruction(self, path):
        return self.__route(path, return_value=False)

    def get_value(self, path):
        return self.__route(path, return_value=True)

    def get_all_targets(self):
        return list(self.address_table.values())

    def get_filtered_targets(
        self, target_type_filter: Union[type, list] = None
    ) -> list:
        if not target_type_filter:
            return self.get_all_targets()

        if type(target_type_filter) == type:
            target_type_filter = [target_type_filter]
            return list(
                p for p in self.address_table.values() if type(p) in target_type_filter
            )

    def get_target_dependencies(self, target_type_filter: Union[type, list] = None):
        targets = self.get_filtered_targets(target_type_filter)
        return [p for p in targets if "/" in p.value]


class Router(Switch):
    def __init__(self, default_gateway=None, routing_table=None):
        super().__init__('', None, None)
        self.default_gateway = default_gateway
        self.routing_table = routing_table
        self.multiprocessing_manager = MultiprocessingManager()

    def execute_task(self, target, *args, **kwargs) -> Task:
        return Task(self.multiprocessing_manager, target, *args, **kwargs)

    def add_worker_queue(
        self,
        target,
        target_size: int = int(virtual_memory().total / 1000000000),
        split_results: bool = False,
        *args,
        **kwargs,
    ) -> WorkerQueue:
        return WorkerQueue(
            self.multiprocessing_manager,
            target,
            target_size=target_size,
            split_results=split_results,
            *args,
            **kwargs,
        )

    def get_parent(self, address: str):
        if '/' not in address:
            return self
        else:
            address = (
                address.rsplit('/', 1)[0]
                if not address.endswith('/')
                else address.rsplit('/', 2)[0]
            )
            try:
                return self.get_instruction(address).switch if address != '' else self
            except RoutingError:
                return self.get_parent(address)

    @staticmethod
    def __convert_to_relative_path(path):
        assert (
            path[0] == '/'
        ), f'Error: {path} is not a global path. Did you want to search for /{path}?'
        path = path.replace('/', '', 1)
        return path


class RoutingError(Exception):
    def __init__(self, route):
        self.message = f'Route {route} does not exist - it might not have been created.'
        super().__init__(self.message)
