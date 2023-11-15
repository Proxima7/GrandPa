import multiprocessing
import queue
import random
import threading
import time


class MultiprocessingManager:
    """
    Manages the multiprocessing of the framework.
    """

    def __init__(self):
        self.task_queue_in = multiprocessing.Queue()
        manager = multiprocessing.Manager()
        self.finished_tasks = manager.dict()
        self.finished_thread_tasks = {}
        self.router = None
        self.thread_queue_in = None

    def start_processes(self, start_func: callable, *args, **kwargs):
        for _ in range(int(multiprocessing.cpu_count() / 4)):
            process = multiprocessing.Process(
                target=self.run_process, args=(start_func, *args), kwargs=kwargs
            )
            process.start()

    def run_process(self, start_func: callable, *args, **kwargs):
        self.start_threads()
        start_func(*args, **kwargs)
        while True:
            self.process_in_queue()

    def start_threads(self):
        self.thread_queue_in = queue.Queue()
        for _ in range(int(multiprocessing.cpu_count() / 4)):
            thread = threading.Thread(target=self.run_thread)
            thread.start()

    def run_thread(self):
        while True:
            self.process_thread_queue()

    def process_in_queue(self):
        task = self.task_queue_in.get()
        result = self.router(task.target)
        self.finished_tasks[task.task_id] = result

    def process_thread_queue(self):
        task = self.thread_queue_in.get()
        result = self.router(task.target)
        self.finished_thread_tasks[task.task_id] = result

    def add_task(self, target, *args, **kwargs):
        task_id = random.randint(0, 1000000000)
        task = Task(target, task_id, *args, **kwargs)
        process_queue_size = self.task_queue_in.qsize()
        thread_queue_size = self.thread_queue_in.qsize()
        if process_queue_size > thread_queue_size:
            self.thread_queue_in.put(task)
        else:
            self.task_queue_in.put(task)
        return task_id

    def get_task_result(self, task_id):
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


class Task:
    """
    Task Class for Multiprocessing. Will execute once.
    """

    def __init__(self, target, task_id, *args, **kwargs):
        self.target = target
        self.task_id = task_id
        self.args = args
        self.kwargs = kwargs
        self.result = None

    def generate_result(self):
        self.result = self.target(*self.args, **self.kwargs)

    def get_result(self):
        return self.result
