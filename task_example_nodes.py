import multiprocessing

import grandpa
from grandpa import Node, Workflow, TaskNode
import os
import threading
import time

from grandpa.multiprocessing_manager import MultiprocessingManager


def print_process_and_thread_info():
    process_id = os.getpid()  # Get the current process ID
    thread_id = threading.get_ident()  # Get the current thread ID
    print(f"Current Process ID: {process_id}, Current Thread ID: {thread_id}")


def task_target(a: int, b: int) -> int:
    print_process_and_thread_info()
    time.sleep(0.1)
    return a + b


class TaskTarget:
    def task_target(self, a: int, b: int) -> int:
        print_process_and_thread_info()
        time.sleep(0.1)
        return a + b


@TaskNode("task_node")
class TaskNode:
    def __init__(self, sample_value: multiprocessing.Value = grandpa.sync.Value(float, 1.0)):
        self.sample_value = sample_value

    def __call__(self, a: int, b: int) -> int:
        print_process_and_thread_info()
        time.sleep(0.1)
        return a + b


@Node("task_creator_class", pass_task_executor=True)
class TaskCreatorClass:
    def __init__(self, task_executor: MultiprocessingManager):
        self.task_executor = task_executor

    def task_target(self, a: int, b: int) -> int:
        print_process_and_thread_info()
        time.sleep(0.1)
        return a + b

    def __call__(self, task_node):
        tasks = [self.task_executor.add_task(task_node, 1, 2) for _ in range(100)]
        results = [self.task_executor.get_task_result(task) for task in tasks]
        return results


@Node("task_creator", pass_task_executor=True)
def task_creator(a: int, b: int, task_executor: MultiprocessingManager):
    t = TaskTarget()
    tasks = [task_executor.add_task(t.task_target, a, b) for _ in range(100)]
    results = [task_executor.get_task_result(task) for task in tasks]
    return results


@Workflow("task_workflow")
def task_workflow():
    tn = TaskNode()
    return tn([1 for _ in range(100)], [2 for _ in range(100)])
