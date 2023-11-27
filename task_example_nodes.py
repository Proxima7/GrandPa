from grandpa import Node, Workflow
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


@Node("task_creator", pass_task_executor=True)
def task_creator(a: int, b: int, task_executor: MultiprocessingManager):
    tasks = [task_executor.add_task(task_target, a, b) for _ in range(100)]
    results = [task_executor.get_task_result(task) for task in tasks]
    return results


@Workflow("task_workflow")
def task_workflow():
    return task_creator(1, 2)
