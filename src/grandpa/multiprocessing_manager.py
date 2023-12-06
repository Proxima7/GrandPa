import multiprocessing
import os
import queue
import random
import threading
import time
import uuid
from typing import Union, Tuple, Any, Dict
from .node import Node
from .task import Task, TaskID


def is_grandpa_node_method(method: callable):
    """
    Checks if a method is part of a class decorated with a grandpa node decorator. Important for multiprocessing.
    Args:
        method: Method to check.

    Returns:
        bool: True if the method is part of a class decorated with a grandpa node decorator, else False.
    """
    if hasattr(method, '__self__') and hasattr(method.__self__, '__decorated_by_grandpa_node__'):
        return getattr(method.__self__, '__decorated_by_grandpa_node__', False)
    return False


def is_grandpa_node_class(cls):
    """
    Checks if a method is part of a class decorated with a grandpa node decorator. Important for multiprocessing.
    Args:
        cls: Class to check.

    Returns:
        bool: True if the method is part of a class decorated with a grandpa node decorator, else False.
    """
    return isinstance(cls, Node)


class MultiprocessingManager:
    """
    Manages the multiprocessing and threading of the framework.
    """

    def __init__(self, manager: multiprocessing.Manager, process_count: int = multiprocessing.cpu_count() // 4,
                 threads_per_process: int = multiprocessing.cpu_count() // 4):
        self.process_count = process_count
        self.threads_per_process = threads_per_process
        self.task_queue_in = multiprocessing.Queue()
        self.finished_tasks = manager.dict()
        self.finished_thread_tasks = {}
        self.router = None
        self.thread_queue_in = None

    def start_processes(self, start_func: callable, *args, **kwargs):
        """
        Starts the processes for the framework.
        Args:
            start_func: Function which the processes call to initialise (usually function which build the graph for the
            framework).
            *args: Additional args to call the start_func with.
            **kwargs: Additional kwargs to call the start_func with.

        Returns:
            None, but starts the processes. The processes will process the task_queue_in.
        """
        for _ in range(self.process_count):
            process = multiprocessing.Process(
                target=self.run_process, args=(start_func, *args), kwargs=kwargs
            )
            process.start()

    def run_process(self, start_func: callable, *args, **kwargs):
        """
        Run function for the processes.
        Args:
            start_func: Function which the processes call to initialise (usually function which build the graph for the
            framework).
            *args: Additional args to call the start_func with.
            **kwargs: Additional kwargs to call the start_func with.

        Returns:
            None, processes the task_queue_in until the process is terminated.
        """
        self.start_threads()
        start_func(*args, **kwargs)
        while True:
            self.process_in_queue()

    def start_threads(self):
        """
        Starts the threads for the framework. Both the main process and the processes started by the framework will
        start threads.
        Returns:
            None, but starts the threads. The threads will process the thread_queue_in.
        """
        self.thread_queue_in = queue.Queue()
        for _ in range(self.threads_per_process):
            thread = threading.Thread(target=self.run_thread)
            thread.start()

    def run_thread(self):
        """
        Run function for the threads.
        Returns:
            None, processes the thread_queue_in until the thread is terminated.
        """
        while True:
            self.process_thread_queue()

    def process_in_queue(self):
        """
        Processes one entry from the task_queue_in.
        Returns:
            None, sets the result of the task in the finished_tasks dict.
        """
        task = self.task_queue_in.get()
        if self.thread_queue_in.qsize() < self.threads_per_process:
            self.thread_queue_in.put(task)
        else:
            if type(task.target) == str:
                if "call_method" in task.kwargs:
                    call_method = task.kwargs["call_method"]
                    del task.kwargs["call_method"]
                else:
                    call_method = None
                result = self.router(task.target, call_method=call_method, call_args=task.args, call_kwargs=task.kwargs)
            else:
                result = task.target(*task.args, **task.kwargs)
            if task.origin == os.getpid():
                self.finished_thread_tasks[task.task_id] = result
            else:
                self.finished_tasks[task.task_id] = result

    def process_thread_queue(self):
        """
        Processes one entry from the thread_queue_in.
        Returns:
            None, sets the result of the task in the finished_thread_tasks dict.
        """
        task = self.thread_queue_in.get()
        if type(task.target) == str:
            if "call_method" in task.kwargs:
                call_method = task.kwargs["call_method"]
                del task.kwargs["call_method"]
            else:
                call_method = None
            result = self.router(task.target, call_method=call_method, call_args=task.args, call_kwargs=task.kwargs)
        else:
            result = task.target(*task.args, **task.kwargs)
        if task.origin == os.getpid():
            self.finished_thread_tasks[task.task_id] = result
        else:
            self.finished_tasks[task.task_id] = result

    @staticmethod
    def convert_to_router_instruction(target: Union[callable, str], *args, **kwargs) -> \
            Tuple[Union[callable, str], Tuple[Any], Dict[str, Any]]:
        """
        Checks if the target can be mapped as a router instruction. If so, it will be converted to a router
        instruction, else it will be returned unchanged.
        Args:
            target: Function which the task will execute.
            *args: Additional args to call the target with.
            **kwargs: Additional kwargs to call the target with.

        Returns:
            target: Target as a router instruction.
        """
        if type(target) != str:
            if is_grandpa_node_method(target):
                grandpa_node_address = target.__self__.__grandpa_node_address__
                method_name = target.__name__
                kwargs["call_method"] = method_name
                return grandpa_node_address, args, kwargs
            elif is_grandpa_node_class(target):
                grandpa_node_address = target.address
                return grandpa_node_address, args, kwargs
        return target, args, kwargs

    def add_task(self, target: Union[callable, str], *args, **kwargs) -> TaskID:
        """
        Adds a task to the task_queue_in or thread_queue_in. The queue with the least entries will be chosen.
        Args:
            target: Function which the task will execute.
            *args: Additional args to call the target with.
            **kwargs: Additional kwargs to call the target with.

        Returns:
            task_id: ID of the task, which can be used to get the result of the task.
        """
        task_id = TaskID(str(uuid.uuid4()))
        target, args, kwargs = self.convert_to_router_instruction(target, *args, **kwargs)
        task = Task(target, task_id, *args, **kwargs)
        if self.thread_queue_in.qsize() < self.threads_per_process * 3 or self.process_count == 0:
            self.thread_queue_in.put(task)
        else:
            self.task_queue_in.put(task)
        return task_id

    def get_task_result(self, task_id: TaskID):
        """
        Gets the result of a task. Will wait until the task is finished.
        Args:
            task_id: ID of the task.

        Returns:
            result: Result of the task.
        """
        while True:
            if task_id in self.finished_tasks:
                result = self.finished_tasks[task_id]
                del self.finished_tasks[task_id]
                return result
            elif task_id in self.finished_thread_tasks:
                result = self.finished_thread_tasks[task_id]
                del self.finished_thread_tasks[task_id]
                return result
            elif not self.task_queue_in.empty():
                self.process_in_queue()
            elif not self.thread_queue_in.empty():
                self.process_thread_queue()
            else:
                time.sleep(0.01)
