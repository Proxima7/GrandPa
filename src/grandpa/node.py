import multiprocessing
import threading
import time

from grandpa.routing import Switch
from grandpa.utils.standard import print_warning, filter_kwargs
import random


class NodeCache:
    def __init__(self, cache_size=multiprocessing.cpu_count() * 6):
        self.cached_values = []
        self.cache_ids = []
        self.cache_size = cache_size

    def save(self, id, value):
        try:
            if self.contains(id):
                self.delete(id)
            if len(self.cache_ids) >= self.cache_size:
                del self.cached_values[0]
                del self.cache_ids[0]
            self.cached_values.append(value)
            self.cache_ids.append(id)
        except Exception as e:
            raise RuntimeError("Error while saving to cache: " + str(e))

    def contains(self, id):
        try:
            return id in self.cache_ids
        except Exception as e:
            raise RuntimeError("Error while checking cache: " + str(e))

    def delete(self, id):
        try:
            if id in self.cache_ids:
                del self.cached_values[self.cache_ids.index(id)]
                del self.cache_ids[self.cache_ids.index(id)]
        except Exception as e:
            raise RuntimeError("Error while deleting from cache: " + str(e))

    def __call__(self, id):
        try:
            if id in self.cache_ids:
                return self.cached_values[self.cache_ids.index(id)]
        except Exception as e:
            raise RuntimeError("Error while reading from cache: " + str(e))


class InProcess:
    pass


class Node:
    def __init__(self, address, main_router):
        self.nodes = {}
        self.address = address
        self.params = {}
        self.settings = {}
        self.switch = Switch(address, self, main_router)
        # print(f'Created new node {self.address}')
        self.cache = NodeCache()
        self.t_lock = threading.Lock()

    def add_node(self, name, node):
        self.__check_if_node_already_exists(name)
        self.nodes[name] = '//' + node.address

    def add_param(self, name, default='__required__'):
        self.params[name] = default

    def add_setting(self, name, value):
        self.settings[name] = value
        return value

    def _validate_params(self, params):
        for req_param, default in self.params.items():
            if req_param not in params:
                if default != '__required__':
                    params[req_param] = default
                else:
                    raise ValueError(f'The required param {req_param} for {self.address} was not set in graph '
                                     f'definition and no default was provided.')
        return params

    def has_param(self, name):
        return name in self.params

    def __check_if_node_already_exists(self, name):
        if name in self.nodes:
            print_warning(f'Parameter {name} for node {self.address} has already been set - '
                          f'previous value will be overridden.')

    def __call_sub_nodes_threaded(self, call_params, nodes, call_id):
        tasks = {}
        for name, node in nodes.items():
            tasks[name] = self.switch.execute_task(node, **{"call_id": call_id, **call_params})
        result = {name: t.get_result() for name, t in tasks.items()}
        return result

    def run(self, *args, **kwargs):
        raise NotImplementedError('run() must be implemented in sub classes of Node.')

    def run_with_call_id(self, *args, call_id=None, **call_params):
        run_nodes = {name: self.switch.get_instruction(node) for name, node in self.nodes.items() if
                     name not in call_params}
        params = self.__call_sub_nodes_threaded(call_params=call_params, nodes=run_nodes, call_id=call_id)
        filtered_call_params = filter_kwargs(self.run, call_params)
        params = {**params, **filtered_call_params}
        params = self._validate_params(params)
        return self.run(*args, **params)

    def await_return_value(self, call_id):
        while True:
            self.t_lock.acquire()
            ret_value = self.cache(call_id)
            self.t_lock.release()
            if type(ret_value) is not InProcess:
                return ret_value
            else:
                time.sleep(0.01)

    def __call__(self, call_id=None, **call_params):
        # print(f"Running node {self.address} with call_id {call_id}")
        self.t_lock.acquire()
        try:
            if call_id is None:
                call_id = random.getrandbits(128)
            if self.cache.contains(call_id):
                ret_value = self.cache(call_id)
                self.t_lock.release()
                if type(ret_value) is InProcess:
                    return self.await_return_value(call_id)
            else:
                self.cache.save(call_id, InProcess())
                self.t_lock.release()
                ret_value = self.run_with_call_id(call_id=call_id, **call_params)
                self.t_lock.acquire()
                self.cache.save(call_id, ret_value)
                self.t_lock.release()
            # print(f"Fetched result for node {self.address} with call_id {call_id}")
            return ret_value
        except Exception as e:
            print(f"Error in node {self.address} with call_id {call_id}")
            self.t_lock.acquire()
            self.cache.delete(call_id)
            self.t_lock.release()
            raise e
