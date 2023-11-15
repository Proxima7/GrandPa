import dill

from grandpa.multiprocessing_manager import MultiprocessingManager
from grandpa.node import Node
from grandpa.routing import Router
from grandpa.template_classes import (FuncTemplate, InitialisedNodeTemplate,
                                      NodeTemplate, ResultWrapper)


class TemplateParser:
    def __init__(self):
        self.initialised_nodes = {}
        self.multiprocessing_manager = MultiprocessingManager()
        self.router = Router(self.multiprocessing_manager)
        self.multiprocessing_manager.router = self.router

    def init_node(self, node):
        if isinstance(node, ResultWrapper):
            return self.init_node(node.origin)[0], "Result"
        else:
            arg_nodes = self.__args_get_required_nodes(node)
            kwarg_nodes = self.__kwargs_get_required_nodes(node)
            if isinstance(node, InitialisedNodeTemplate):
                return self.__init_node_cls_call(arg_nodes, kwarg_nodes, node)
            elif isinstance(node, NodeTemplate):
                return self.__init_node_cls_init(arg_nodes, kwarg_nodes, node)
            elif isinstance(node, FuncTemplate):
                return self.__init_node_function(arg_nodes, kwarg_nodes, node)
            else:
                raise RuntimeError(f"Unknown node type: {type(node)}")

    def __init_node_function(self, arg_nodes, kwarg_nodes, node):
        (
            call_args,
            call_kwargs,
            required_arg_nodes,
            required_kwarg_nodes,
        ) = self.get_node_parameters(arg_nodes, kwarg_nodes, node)
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

    def __init_node_cls_init(self, arg_nodes, kwarg_nodes, node):
        (
            call_args,
            call_kwargs,
            required_arg_nodes,
            required_kwarg_nodes,
        ) = self.get_node_parameters(arg_nodes, kwarg_nodes, node)
        for req_arg_node in required_arg_nodes:
            call_args.append(self.router(*req_arg_node))
        for key, req_kwarg_node in required_kwarg_nodes.items():
            call_kwargs[key] = self.router(*req_kwarg_node)
        init_cls = node.cls(*call_args, **call_kwargs)
        return init_cls, "initialised_cls"

    def __init_node_cls_call(self, arg_nodes, kwarg_nodes, node):
        (
            call_args,
            call_kwargs,
            required_arg_nodes,
            required_kwarg_nodes,
        ) = self.get_node_parameters(arg_nodes, kwarg_nodes, node)
        init_cls, _ = self.init_node(node.node_template)
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

    def __kwargs_get_required_nodes(self, node):
        kwarg_nodes = {}
        for key, req_node in node.required_kwarg_nodes.items():
            if req_node not in self.initialised_nodes:
                kwarg_nodes[key] = self.init_node(req_node)
            else:
                kwarg_nodes[key] = self.initialised_nodes[req_node]
        return kwarg_nodes

    def __args_get_required_nodes(self, node):
        arg_nodes = []
        for req_node in node.required_arg_nodes:
            if req_node not in self.initialised_nodes:
                arg_nodes.append(self.init_node(req_node))
            else:
                arg_nodes.append(self.initialised_nodes[req_node])
        return arg_nodes

    def get_node_parameters(self, arg_nodes, kwarg_nodes, node):
        call_args, required_arg_nodes = self.__get_node_args(arg_nodes, node)
        call_kwargs, required_kwarg_nodes = self.__get_node_kwargs(kwarg_nodes, node)
        return call_args, call_kwargs, required_arg_nodes, required_kwarg_nodes

    @staticmethod
    def __get_node_kwargs(kwarg_nodes, node):
        call_kwargs = node.call_kwargs
        required_kwarg_nodes = {}
        for key, kwarg_node in kwarg_nodes.items():
            if kwarg_node[1] == "Result":
                required_kwarg_nodes[key] = (kwarg_node[0], False)
            else:
                required_kwarg_nodes[key] = (kwarg_node[0], True)
        return call_kwargs, required_kwarg_nodes

    @staticmethod
    def __get_node_args(arg_nodes, node):
        call_args = node.call_args
        required_arg_nodes = []
        for arg_node in arg_nodes:
            if arg_node[1] == "Result":
                required_arg_nodes.append((arg_node[0], False))
            else:
                required_arg_nodes.append((arg_node[0], True))
        return call_args, required_arg_nodes

    def create_process_graph(self, final_node):
        unpickled_node = dill.loads(final_node)
        self.init_node(unpickled_node)

    def __call__(self, workflow):
        pickled_workflow = dill.dumps(workflow)
        self.multiprocessing_manager.start_processes(
            self.create_process_graph, pickled_workflow
        )
        self.multiprocessing_manager.start_threads()
        final_node, node_type = self.init_node(workflow())
        if node_type == "Node":
            return self.router(final_node, True)
        elif node_type == "Result":
            return self.router(final_node, False)
        else:
            raise RuntimeError(f"Unexpected node type: {node_type}")
