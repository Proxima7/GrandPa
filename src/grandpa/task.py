import os


class TaskID:
    """
    Task ID class for multiprocessing.
    """

    def __init__(self, task_id: int):
        self.task_id = task_id

    def __str__(self):
        return str(self.task_id)

    def __repr__(self):
        return str(self.task_id)

    def __eq__(self, other):
        return self.task_id == other.task_id

    def __hash__(self):
        return hash(self.task_id)


class Task:
    """
    Task Class for Multiprocessing. Will execute once.
    """

    def __init__(self, target: callable, task_id: TaskID, *args, **kwargs):
        self.target = target
        self.task_id = task_id
        self.args = args
        self.kwargs = kwargs
        self.result = None
        self.origin = os.getpid()

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
