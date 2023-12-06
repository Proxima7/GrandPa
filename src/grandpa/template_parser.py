import multiprocessing

import dill

from grandpa.multiprocessing_manager import MultiprocessingManager
from grandpa.node import Node, TaskNode
from grandpa.routing import Router
from grandpa.template_classes import (FuncTemplate, InitialisedNodeTemplate,
                                      NodeTemplate, ResultWrapper)
import grandpa.sync as sync


class TemplateParser:
    """
    TemplateParser class. Used to parse a workflow template into an executable pipeline.
    """
    def __init__(self, process_count: int = multiprocessing.cpu_count() // 4,
                 threads_per_process: int = multiprocessing.cpu_count() // 4):
        self.initialised_nodes = {}
        self.process_count = process_count
        self.threads_per_process = threads_per_process
        self.multiprocessing_manager = None
        self.router = None

    def init_multiprocessing_manager(self, manager: multiprocessing.Manager):
        """
        Initialises the grandpa multiprocessing manager. Needs to be done after init in order to use only on manager.
        Args:
            manager: Multiprocessing manager instance.

        Returns:
            None, but initialises the grandpa multiprocessing manager.
        """
        self.multiprocessing_manager = MultiprocessingManager(manager=manager, process_count=self.process_count,
                                                              threads_per_process=self.threads_per_process)
        self.router = Router(self.multiprocessing_manager)
        self.multiprocessing_manager.router = self.router

    def init_node(self, node, sync_objects):
        """
        Recursive function to initialise node templates. Will check the prerequisites of the node template and
        initialise them if they have not already been initialised. Will then initialise the node template itself.
        Args:
            node: Node template to initialise (Can be a NodeTemplate, InitialisedNodeTemplate, ResultWrapper or
            FuncTemplate)
            sync_objects: Sync objects to use for IPC.

        Returns:
            init_result: Result of the initialisation (e.g. the result for ResultWrapper, or the initialised node for
            NodeTemplate or FuncTemplate)
            type_of_result: Type of the result (e.g. "Result" for ResultWrapper, "Node" for NodeTemplate or
            FuncTemplate)
        """
        if isinstance(node, ResultWrapper):
            return self.init_node(node.origin, sync_objects)[0], "Result"
        else:
            arg_nodes = self.__args_get_required_nodes(node, sync_objects)
            kwarg_nodes = self.__kwargs_get_required_nodes(node, sync_objects)
            if isinstance(node, InitialisedNodeTemplate):
                return self.__init_node_cls_call(arg_nodes, kwarg_nodes, node, sync_objects)
            elif isinstance(node, NodeTemplate):
                return self.__init_node_cls_init(arg_nodes, kwarg_nodes, node, sync_objects)
            elif isinstance(node, FuncTemplate):
                return self.__init_node_function(arg_nodes, kwarg_nodes, node, sync_objects)
            else:
                raise RuntimeError(f"Unknown node type: {type(node)}")

    def __init_node_function(self, arg_nodes, kwarg_nodes, node, sync_objects):
        """
        Initialises a FuncTemplate. Will check the prerequisites of the node template and initialise them if they have
        not already been initialised. Will then initialise the node template itself.
        Args:
            arg_nodes: Required arg nodes for the node template.
            kwarg_nodes: Required kwarg nodes for the node template.
            node: Node template to initialise.
            sync_objects: Sync objects to use for IPC.

        Returns:
            init_result: Result of the initialisation (in this case initialised node which wraps the function)
            type_of_result: Type of the result (in this case "Node")
        """
        (
            call_args,
            call_kwargs,
            required_arg_nodes,
            required_kwarg_nodes,
        ) = self.get_node_parameters(arg_nodes, kwarg_nodes, node, sync_objects)
        if node.grandpa_task_node:
            t_node = Node(
                node.name + "_target",
                self.router,
                node.function,
                [],
                {},
                [],
                {},
            )
            f_node = TaskNode(
                node.name,
                self.router,
                node.function,
                call_args,
                call_kwargs,
                required_arg_nodes,
                required_kwarg_nodes,
                t_node.address,
            )
        else:
            f_node = Node(
                node.name,
                self.router,
                node.function,
                call_args,
                call_kwargs,
                required_arg_nodes,
                required_kwarg_nodes,
            )
        self.initialised_nodes[node] = f_node.address
        return f_node.address, "Node"

    def __init_node_cls_init(self, arg_nodes, kwarg_nodes, node, sync_objects):
        """
        Initialises a NodeTemplate. Will check the prerequisites of the node template and initialise them if they have
        not already been initialised. Will then initialise the node template itself.
        Args:
            arg_nodes: Required arg nodes for the node template.
            kwarg_nodes: Required kwarg nodes for the node template.
            node: Node template to initialise.
            sync_objects: Sync objects to use for IPC.

        Returns:
            init_result: Result of the initialisation (in this case instance of the class wrapped by the Node)
            type_of_result: Type of the result (in this case "initialised_cls")
        """
        (
            call_args,
            call_kwargs,
            required_arg_nodes,
            required_kwarg_nodes,
        ) = self.get_node_parameters(arg_nodes, kwarg_nodes, node, sync_objects)
        for req_arg_node in required_arg_nodes:
            call_args.append(self.router(*req_arg_node))
        for key, req_kwarg_node in required_kwarg_nodes.items():
            call_kwargs[key] = self.router(*req_kwarg_node)
        init_cls = node.cls(*call_args, **call_kwargs)
        return init_cls, "initialised_cls"

    def __init_node_cls_call(self, arg_nodes, kwarg_nodes, node, sync_objects):
        """
        Initialises a InitialisedNodeTemplate. Will check the prerequisites of the node template and initialise them if
        they have not already been initialised. Will then initialise the node template itself.
        Args:
            arg_nodes: Required arg nodes for the node template.
            kwarg_nodes: Required kwarg nodes for the node template.
            node: Node template to initialise.
            sync_objects: Sync objects to use for IPC.

        Returns:
            init_result: Result of the initialisation (in this case Node wrapping an instance of a class)
            type_of_result: Type of the result (in this case "Node")
        """
        (
            call_args,
            call_kwargs,
            required_arg_nodes,
            required_kwarg_nodes,
        ) = self.get_node_parameters(arg_nodes, kwarg_nodes, node, sync_objects)
        init_cls, _ = self.init_node(node.node_template, sync_objects)
        if node.grandpa_task_node:
            t_node = Node(
                node.node_template.name + "_target",
                self.router,
                init_cls,
                [],
                {},
                [],
                {},
            )
            f_node = TaskNode(
                node.node_template.name,
                self.router,
                init_cls,
                call_args,
                call_kwargs,
                required_arg_nodes,
                required_kwarg_nodes,
                t_node.address,
            )
        else:
            f_node = Node(
                node.node_template.name,
                self.router,
                init_cls,
                call_args,
                call_kwargs,
                required_arg_nodes,
                required_kwarg_nodes,
            )
        self.initialised_nodes[node] = f_node.address
        return f_node.address, "Node"

    def __kwargs_get_required_nodes(self, node, sync_objects):
        """
        Gets the required kwarg nodes for a node template and makes sure they are initialised.
        Args:
            node: Node template to get the required kwarg nodes for.
            sync_objects: Sync objects to use for IPC.

        Returns:
            kwarg_nodes: Required kwarg nodes for the node template.
        """
        kwarg_nodes = {}
        for key, req_node in node.required_kwarg_nodes.items():
            if req_node not in self.initialised_nodes:
                kwarg_nodes[key] = self.init_node(req_node, sync_objects)
            else:
                kwarg_nodes[key] = self.initialised_nodes[req_node]
        return kwarg_nodes

    def __args_get_required_nodes(self, node, sync_objects):
        """
        Gets the required arg nodes for a node template and makes sure they are initialised.
        Args:
            node: Node template to get the required arg nodes for.
            sync_objects: Sync objects to use for IPC.

        Returns:
            arg_nodes: Required arg nodes for the node template.
        """
        arg_nodes = []
        for req_node in node.required_arg_nodes:
            if req_node not in self.initialised_nodes:
                arg_nodes.append(self.init_node(req_node, sync_objects))
            else:
                arg_nodes.append(self.initialised_nodes[req_node])
        return arg_nodes

    def get_node_parameters(self, arg_nodes, kwarg_nodes, node, sync_objects):
        """
        Gets the parameters for a node template.
        Args:
            arg_nodes: Required arg nodes for the node template.
            kwarg_nodes: Required kwarg nodes for the node template.
            node: Node template to initialise.
            sync_objects: Sync objects to use for IPC.

        Returns:
            call_args: Arguments to pass to the node template.
            call_kwargs: Keyword arguments to pass to the node template.
            required_arg_nodes: Required arg nodes for the node template.
            required_kwarg_nodes: Required kwarg nodes for the node template.
        """
        call_args, required_arg_nodes = self.__get_node_args(arg_nodes, node)
        call_kwargs, required_kwarg_nodes = self.__get_node_kwargs(kwarg_nodes, node)
        if node.name in sync_objects:
            for key, value in sync_objects[node.name].items():
                call_kwargs[key] = value
        return call_args, call_kwargs, required_arg_nodes, required_kwarg_nodes

    def __get_node_kwargs(self, kwarg_nodes, node):
        """
        Gets the required kwarg nodes for a node template in the desired format.
        Args:
            kwarg_nodes: Required kwarg nodes for the node template.
            node: Node template to initialise.

        Returns:
            call_kwargs: Keyword arguments to pass to the node template.
            required_kwarg_nodes: Required kwarg nodes for the node template.
        """
        call_kwargs = node.call_kwargs
        if node.pass_task_executor:
            call_kwargs["task_executor"] = self.multiprocessing_manager
        required_kwarg_nodes = {}
        for key, kwarg_node in kwarg_nodes.items():
            if kwarg_node[1] == "Result":
                required_kwarg_nodes[key] = (kwarg_node[0], False)
            else:
                required_kwarg_nodes[key] = (kwarg_node[0], True)
        return call_kwargs, required_kwarg_nodes

    @staticmethod
    def __get_node_args(arg_nodes, node):
        """
        Gets the required arg nodes for a node template in the desired format.
        Args:
            arg_nodes: Required arg nodes for the node template.
            node: Node template to initialise.

        Returns:
            call_args: Arguments to pass to the node template.
            required_arg_nodes: Required arg nodes for the node template.
        """
        call_args = node.call_args
        required_arg_nodes = []
        for arg_node in arg_nodes:
            if arg_node[1] == "Result":
                required_arg_nodes.append((arg_node[0], False))
            else:
                required_arg_nodes.append((arg_node[0], True))
        return call_args, required_arg_nodes

    def create_process_graph(self, final_node, sync_objects):
        """
        Creates the executable graph for a different process.
        Args:
            final_node: Final node in the graph.
            sync_objects: Sync objects for IPC.

        Returns:
            None
        """
        unpickled_node = dill.loads(final_node)
        self.init_node(unpickled_node, sync_objects)

    @staticmethod
    def get_manager_object(grandpa_sync_object, manager):
        if isinstance(grandpa_sync_object, sync.Queue):
            return manager.Queue()
        elif isinstance(grandpa_sync_object, sync.Value):
            return manager.Value(grandpa_sync_object.dtype, grandpa_sync_object.value)
        elif isinstance(grandpa_sync_object, sync.List):
            return manager.list()
        elif isinstance(grandpa_sync_object, sync.Dict):
            return manager.dict()
        elif isinstance(grandpa_sync_object, sync.Lock):
            return manager.Lock()
        else:
            raise TypeError(f"Unknown sync object type {type(grandpa_sync_object)}")

    def init_sync_classes(self, manager: multiprocessing.Manager, node, sync_objects: dict = None):
        if sync_objects is None:
            sync_objects = {}
        if hasattr(node, "sync_params"):
            for name, param in node.sync_params.items():
                if node.name not in sync_objects:
                    sync_objects[node.name] = {}
                sync_objects[node.name][name] = self.get_manager_object(param, manager)
        if isinstance(node, ResultWrapper):
            return self.init_sync_classes(manager, node.origin, sync_objects)
        elif isinstance(node, InitialisedNodeTemplate):
            return self.init_sync_classes(manager, node.node_template, sync_objects)
        else:
            for arg_node in node.required_arg_nodes:
                sync_objects = self.init_sync_classes(manager, arg_node, sync_objects)
            for kwarg_node in node.required_kwarg_nodes.values():
                sync_objects = self.init_sync_classes(manager, kwarg_node, sync_objects)
            return sync_objects

    def __call__(self, workflow, settings=None):
        """
        Runs a workflow. Returns either the result of the final node or the final node itself, depending on the
        workflow template. Also initialises the multiprocessing manager and its processes.
        Args:
            workflow: Workflow template to run.

        Returns:
            result: Result of the final node in the workflow.
        """
        if settings is None:
            settings = {}
        manager = multiprocessing.Manager()
        self.init_multiprocessing_manager(manager)
        final_node = workflow(**settings)
        sync_objects = self.init_sync_classes(manager, final_node)
        pickled_workflow = dill.dumps(final_node)
        self.multiprocessing_manager.start_processes(
            self.create_process_graph, pickled_workflow, sync_objects
        )
        self.multiprocessing_manager.start_threads()
        final_node, node_type = self.init_node(final_node, sync_objects)
        if node_type == "Node":
            return self.router(final_node, True)
        elif node_type == "Result":
            return self.router(final_node, False)
        else:
            raise RuntimeError(f"Unexpected node type: {node_type}")
