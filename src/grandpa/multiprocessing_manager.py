import multiprocessing

import dill
import random
import time
from multiprocessing import Queue, Process


class MultiprocessingManager:
    """
    Manages the multiprocessing of the framework.
    """

    def __init__(self):
        self.task_queue_in = Queue()
        self.task_queue_out = Queue()
        self.finished_tasks = {}

    def start_processes(self):
        pass

    def add_task(self, target, *args, **kwargs):
        task_id = random.randint(0, 1000000000)
        task = Task(target, task_id, *args, **kwargs)
        task.generate_result()
        self.finished_tasks[task_id] = task.get_result()
        return task_id

    def get_task_result(self, task_id):
        while True:
            if task_id in self.finished_tasks:
                return self.finished_tasks[task_id]
            elif not self.task_queue_out.empty():
                self.read_out_queue()
            else:
                time.sleep(0.1)

    def read_out_queue(self):
        task = self.task_queue_out.get()
        self.finished_tasks[task.task_id] = task.get_result()

    def process_in_queue(self):
        while True:
            task = self.task_queue_in.get()
            task.generate_result()
            self.task_queue_out.put(task)


class Task:
    """
    Task Class for Multiprocessing. Will execute once.
    """

    def __init__(
        self, target, task_id, *args, **kwargs
    ):
        self.target = target
        self.task_id = task_id
        self.args = args
        self.kwargs = kwargs
        self.result = None

    def generate_result(self):
        self.result = self.target(*self.args, **self.kwargs)

    def get_result(self):
        return self.result
