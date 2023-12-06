import inspect


class Queue:
    """
    Requests a queue which can be used to synchronise data between processes.
    """
    pass


class List:
    """
    Requests a list which can be used to synchronise data between processes.
    """
    pass


class Dict:
    """
    Requests a dict which can be used to synchronise data between processes.
    """
    pass


class Value:
    """
    Requests a value which can be used to synchronise data between processes.
    """
    def __init__(self, dtype: type, value: any):
        self.dtype = dtype
        self.value = value


class Lock:
    """
    Requests a lock which can be used to synchronise states between processes.
    """
    pass


def get_sync_params(func: callable):
    sig = inspect.signature(func)
    parameters = sig.parameters
    sync_params = {}
    for name, param in parameters.items():
        if isinstance(param.default, Queue):
            sync_params[name] = param.default
        elif isinstance(param.default, List):
            sync_params[name] = param.default
        elif isinstance(param.default, Dict):
            sync_params[name] = param.default
        elif isinstance(param.default, Value):
            sync_params[name] = param.default
        elif isinstance(param.default, Lock):
            sync_params[name] = param.default
    return sync_params
