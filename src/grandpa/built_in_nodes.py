import json
import os
from concurrent.futures.thread import ThreadPoolExecutor
import threading
from typing import Union

from grandpa.node import Node
from grandpa.utils.standard import do_import, filter_kwargs


class SettingNode(Node):
    """
    This class can be used to create a node that validates a user setting. Requires a SettingReaderNode as
    param 'settings'.
    """
    def __init__(self, setting, default=None, setting_type=None, val_func=None, **kwargs):
        self.setting = setting
        self.default = default
        self.setting_type = setting_type
        self.val_func = val_func
        super().__init__(**kwargs)

    def __check_setting_value(self, setting_value):
        """

        Args:
            setting_value: Value of the setting.

        Returns:
            Settings value if it passes all checks, else the default value given at init.
        """
        if not self.__check_type(setting_value):
            return self.default
        elif not self.__get_val_func_affirmation(setting_value):
            return self.default
        else:
            return setting_value

    def __check_type(self, setting_value):
        """

        Args:
            setting_value: Value of the setting.

        Returns:
            True if the setting matches the expected type, else False.
        """
        if self.setting_type is None:
            return True
        elif type(setting_value) == self.setting_type:
            return True
        else:
            return False

    def __get_val_func_affirmation(self, setting_value):
        """

        Args:
            setting_value: Value of the setting.

        Returns:
            True if the setting passes the val func (or there is no val func), else False.
        """
        if self.val_func is not None:
            return self.val_func(setting_value)
        else:
            return True

    def run(self, **params):
        """
        Executes the node. Shouldn't be called directly. Instead, use __call__.

        Args:
            **params: Not implemented.

        Returns:
            Settings value if it passes all checks, else the default value given at init.
        """
        setting_value = self.setting
        return self.__check_setting_value(setting_value)


class SettingReaderNode(Node):
    """
    This class will read the Settings.
    """
    def __init__(self, setting_path: Union[dict, str], **kwargs):
        self.__init_settings(setting_path)
        super().__init__(**kwargs)
        self.add_param('setting')

    def __init_settings(self, setting_path: Union[dict, str]):
        """
        Initialises the settings.

        Args:
            setting_path: Path to the settings file or a dict containing the settings.

        Returns:
            None
        """
        if type(setting_path) is str:
            self.__read_setting_from_file(setting_path)
        elif type(setting_path) is dict:
            self.framework_settings = setting_path
        else:
            raise NotImplementedError(f'Type {type(setting_path)} is not implemented, it must be either dict or '
                                      f'string.')

    def __read_setting_from_file(self, setting_path: str):
        """
        Reads the settings from a file.

        Args:
            setting_path: Path to the settings file.

        Returns:
            None
        """
        assert os.path.exists(setting_path), f'File {setting_path} does not exist.'
        with open(setting_path, 'r') as file:
            self.framework_settings = json.load(file)

    def __parse_dict_reference(self, setting: str):
        """

        Args:
            setting: Framework internal settings path. E.g. 'network/backbone/input_shape'

        Returns:
            The value of the setting at the given path.
        """
        target_setting = self.framework_settings
        if setting == '' or setting == "*":
            return target_setting
        for part in setting.split('/'):
            target_setting = target_setting[part]
        return target_setting

    def run(self, **params):
        """
        Executes the node. Shouldn't be called directly. Instead, use __call__.

        Args:
            **params: Must include a parameter setting that specifies the framework internal settings path, e.g.
            'network/backbone/input_shape'.

        Returns:
            Settings value for the given settings path.
        """
        assert hasattr(self, 'settings'), 'Something must have gone horribly wrong, since your SettingReaderNode is ' \
                                          'lacking the "settings" attribute that should always be created at init. ' \
                                          'As this is most likely a source code problem, please check the code and ' \
                                          'contact the developers.'
        return self.__parse_dict_reference(params['setting'])


class QueueWrapper(Node):
    """
    Wraps a node as queue. Will constantly generate data until a upper limit (based on pc specs) is reached using the
    framework multiprocessing. Calling this node will return an entry from the queue.
    """
    def __init__(self, target_node, **kwargs):
        super().__init__(**kwargs)
        self.worker_queue = self.switch.add_worker_queue(target=target_node)

    def run(self, **params):
        return self.worker_queue.get()


class FuncExecNode(Node):
    """
    This node can be understood as a function wrapper. The function must be passed as func at init, all parameters
    required for the function need to be passed at call, either directly (call with params) or via nodes
    (self.add_node(name, node)).
    """
    def __init__(self, func, **kwargs):
        if not callable(func):
            func = do_import(func)
        self.func = func
        super().__init__(**kwargs)

    def run(self, **params):
        filtered_kwargs = filter_kwargs(self.func, params)
        return self.func(**filtered_kwargs)


class FuncCacheNode(Node):
    """
    This Node wraps a function the same way as FuncExecNode, except it only calls the function once and caches the
    result.
    """
    def __init__(self, func, **kwargs):
        self.func = func
        self.value = None
        self.lock = threading.Lock()
        super().__init__(**kwargs)

    def run(self, **params):
        self.lock.acquire()
        if not self.value:
            self.value = self.func(**params)
        self.lock.release()
        return self.value


class ThreadedMultiNode(Node):
    """
    This node can be used to call another node multiple times with different parameters in parallel (threading).
    Deprecated.
    """
    def __init__(self, node, **kwargs):
        self.node = node
        super().__init__(**kwargs)
        self.add_param('settings')

    def run(self, settings):
        with ThreadPoolExecutor() as tpe:
            threads = {}
            for name, setting_set in settings.items():
                threads[name] = tpe.submit(self.node, **setting_set)
            return {name: t.result() for name, t in threads.items()}


class ListWrapperNode(Node):
    """
    This node can be used to wrap a list as a node.
    """
    def __init__(self, sub_nodes: list, **kwargs):
        self.sub_nodes = sub_nodes
        super().__init__(**kwargs)

    def run(self):
        return [node() for node in self.sub_nodes]


class DictWrapperNode(Node):
    """
    This node can be used to wrap a dict as a node.
    """

    def __init__(self, sub_nodes: dict, **kwargs):
        self.sub_nodes = sub_nodes
        super().__init__(**kwargs)

    def run(self):
        return {key: node() for key, node in self.sub_nodes.items()}


class ValueWrapperNode(Node):
    """
    This node can be used in case you want to store a fixed value in a node. It will return the stored value every
    time it is called.
    """
    def __init__(self, value, **kwargs):
        self.value = value
        super().__init__(**kwargs)

    def run(self):
        return self.value


class ResultWrapperNode(Node):
    """
    This node is used in order to wrap the result of multiple nodes into one dict. All nodes must be added via
    add_node(name, node) to the ResultWrapperNode.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def run(self, **params):
        return {name: value for name, value in params.items()}


class GrandMLWrapperNode(Node):
    def __init__(self, *args, node_location: str, address, main_router, **kwargs):
        super().__init__(address, main_router)
        self.execute_node = do_import(node_location)(*args, **kwargs)

    def run(self, *args, **kwargs):
        return self.execute_node(*args, **kwargs)
