from grandpa import Component, Node, GeneratorNode, Workflow, TemplateParser


@GeneratorNode("task_target", max_queue_size=500)
class TaskTarget:
    def __init__(self, a, b):
        self.a = a
        self.b = b

    def __call__(self):
        image = self.a + self.b
        return image


@Node("data_manager")
class DataManager:
    def __init__(self, data_node, batch_size: int = 25):
        self.data_node = data_node
        self.batch_size = batch_size

    def __call__(self, *args, **kwargs):
        return [self.data_node() for _ in range(self.batch_size)]


@Workflow("data_workflow")
def generator_workflow():
    gn = TaskTarget(a=2, b=4)
    dm = DataManager(gn, batch_size=50)
    return dm()




