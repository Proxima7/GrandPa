import multiprocessing
import queue
import random
import threading
import time


class MultiprocessingManager:
    """
    Manages the multiprocessing and threading of the framework.
    """

    def __init__(self):
        self.task_queue_in = multiprocessing.Queue()
        manager = multiprocessing.Manager()
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
        for _ in range(int(multiprocessing.cpu_count() / 4)):
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
        for _ in range(int(multiprocessing.cpu_count() / 4)):
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
        result = self.router(task.target)
        self.finished_tasks[task.task_id] = result

    def process_thread_queue(self):
        """
        Processes one entry from the thread_queue_in.
        Returns:
            None, sets the result of the task in the finished_thread_tasks dict.
        """
        task = self.thread_queue_in.get()
        result = self.router(task.target)
        self.finished_thread_tasks[task.task_id] = result

    def add_task(self, target: callable, *args, **kwargs) -> int:
        """
        Adds a task to the task_queue_in or thread_queue_in. The queue with the least entries will be chosen.
        Args:
            target: Function which the task will execute.
            *args: Additional args to call the target with.
            **kwargs: Additional kwargs to call the target with.

        Returns:
            task_id: ID of the task, which can be used to get the result of the task.
        """
        task_id = random.randint(0, 1000000000)
        task = Task(target, task_id, *args, **kwargs)
        process_queue_size = self.task_queue_in.qsize()
        thread_queue_size = self.thread_queue_in.qsize()
        if process_queue_size > thread_queue_size:
            self.thread_queue_in.put(task)
        else:
            self.task_queue_in.put(task)
        return task_id

    def get_task_result(self, task_id: int):
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


class Task:
    """
    Task Class for Multiprocessing. Will execute once.
    """

    def __init__(self, target: callable, task_id: int, *args, **kwargs):
        self.target = target
        self.task_id = task_id
        self.args = args
        self.kwargs = kwargs
        self.result = None

    def generate_result(self):
        """
        Executes the target function with the args and kwargs.
        Returns:
            None, but sets the result of the task.
        """
        self.result = self.target(*self.args, **self.kwargs)

    def get_result(self):
        """
        Gets the result of the task.
        Returns:
            result: Result of the task.
        """
        return self.result
