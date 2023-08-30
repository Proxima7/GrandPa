import multiprocessing
import os
import threading
import time
from queue import Queue
from threading import Thread

from psutil import virtual_memory

from grandpa.utils.standard import check_debug, print_warning


class MultiprocessingManager:
    """
    Manages the multiprocessing of the framework.
    """

    def __init__(self):
        self.__task_queue = Queue()
        self.__worker_queue_list = []
        self.__init_workers()

    def add_worker_queue(self, worker_queue):
        """
        Registers a worker Queue.

        Args:
            worker_queue: Worker Queue Object to register.

        Returns:
            None
        """
        self.__worker_queue_list.append(worker_queue)

    def add_task(self, task):
        """
        Adds a task to be processed.

        Args:
            task: Task Object.

        Returns:
            None
        """
        self.__task_queue.put(task)

    def __init_workers(self):
        debug = (
            os.environ["project_x_debug"] if "project_x_debug" in os.environ else False
        )
        if debug == "true":
            worker_count = 1
        else:
            worker_count = multiprocessing.cpu_count() - 2
        for _ in range(worker_count):
            t = Thread(target=self.__work)
            t.daemon = True
            t.start()

    def __work(self):
        while True:
            try:
                if self.__task_queue.qsize() > 0:
                    task = self.__task_queue.get()
                    if task.lock.locked():
                        continue
                    else:
                        task.get_result()
                elif len(self.__worker_queue_list) > 0:
                    task_queue = self.__get_task_queue()
                    if task_queue is None:
                        time.sleep(0.01)
                    else:
                        task_queue.generate()
                else:
                    time.sleep(0.01)
            except Exception as e:
                if check_debug():
                    raise e
                else:
                    print_warning(str(e))

    def __get_task_queue(self):
        target_queue = None
        target_queue_size = float('inf')
        for t_queue in self.__worker_queue_list:
            if (
                t_queue.qsize() / t_queue.target_size < target_queue_size
                and t_queue.qsize() < t_queue.target_size
            ):
                target_queue_size = t_queue.qsize() / t_queue.target_size
                target_queue = t_queue
        return target_queue


class Task:
    """
    Task Class for Multiprocessing. Will execute once.
    """

    def __init__(
        self, multiprocessing_manager: MultiprocessingManager, target, *args, **kwargs
    ):
        self.target = target
        self.args = args
        self.kwargs = kwargs
        self.lock = threading.Lock()
        self.result = NoData()
        multiprocessing_manager.add_task(self)

    def get_result(self):
        self.lock.acquire()
        if type(self.result) == NoData:
            self.result = self.target(*self.args, **self.kwargs)
        self.lock.release()
        return self.result


class WorkerQueue(Queue):
    """
    Class for registering a worker queue. Runs constantly until target_size is reached.
    """

    def __init__(
        self,
        multiprocessing_manager: MultiprocessingManager,
        target,
        target_size: int = int(virtual_memory().total / 1000000000),
        split_results: bool = False,
        batches_per_run: int = 1,
        *args,
        **kwargs
    ):
        super().__init__()
        self.target = target
        self.args = args
        self.kwargs = kwargs
        self.target_size = target_size
        self.split_results = split_results
        self.batches_per_run = batches_per_run
        self.__pending_batches = 0
        self.static_workers = []
        self.__init_static_workers(1)
        multiprocessing_manager.add_worker_queue(self)

    def __init_static_workers(self, worker_num: int):
        debug = os.getenv("project_x_debug", False)
        if not debug == "true":
            for _ in range(worker_num):
                t = Thread(target=self.__worker_thread)
                t.daemon = True
                t.start()
                self.static_workers.append(t)

    def __worker_thread(self):
        if self.qsize() < self.target_size:
            self.generate()
        else:
            time.sleep(0.1)

    def get(self, block=True, timeout=None):
        debug = os.getenv("project_x_debug", False)
        if self.qsize() > 0:
            return super().get(block, timeout)
        else:
            try:
                self.generate()
            except Exception as e:
                if debug:
                    raise e
                else:
                    print(e)
            return self.get(block, timeout)

    def generate(self):
        self.__pending_batches += self.batches_per_run
        try:
            result = self.target(*self.args, **self.kwargs)
            if self.split_results:
                for r in result:
                    self.put(r)
            else:
                self.put(result)
        except Exception as e:
            if check_debug():
                raise e
            else:
                print_warning(str(e))
                return self.generate()
        self.__pending_batches -= self.batches_per_run

    def qsize_old(self) -> int:
        return super().qsize() + self.__pending_batches


class NoData:
    """
    Helper class. Symbolizes that a Task has not yet been completed.
    """

    pass
