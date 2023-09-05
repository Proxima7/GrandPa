import multiprocessing

import dill

from grandpa.node import Node
from grandpa.routing import Router
from grandpa.template_classes import (FuncTemplate, InitialisedNodeTemplate,
                                      NodeTemplate, ResultWrapper)


class TemplateParser:
    def __init__(self):
        self.initialised_nodes = {}
        self.router = Router()

    def init_node(self, node):
        if isinstance(node, ResultWrapper):
            return self.init_node(node.origin)[0], "Result"
        else:
            arg_nodes = []
            for req_node in node.required_arg_nodes:
                if req_node not in self.initialised_nodes:
                    arg_nodes.append(self.init_node(req_node))
                else:
                    arg_nodes.append(self.initialised_nodes[req_node])
            kwarg_nodes = {}
            for key, req_node in node.required_kwarg_nodes.items():
                if req_node not in self.initialised_nodes:
                    kwarg_nodes[key] = self.init_node(req_node)
                else:
                    kwarg_nodes[key] = self.initialised_nodes[req_node]
            if isinstance(node, InitialisedNodeTemplate):
                (
                    call_args,
                    call_kwargs,
                    required_arg_nodes,
                    required_kwarg_nodes,
                ) = self.get_node_args(arg_nodes, kwarg_nodes, node)
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
            elif isinstance(node, NodeTemplate):
                (
                    call_args,
                    call_kwargs,
                    required_arg_nodes,
                    required_kwarg_nodes,
                ) = self.get_node_args(arg_nodes, kwarg_nodes, node)
                for req_arg_node in required_arg_nodes:
                    call_args.append(self.router(*req_arg_node))
                for key, req_kwarg_node in required_kwarg_nodes.items():
                    call_kwargs[key] = self.router(*req_kwarg_node)
                init_cls = node.cls(*call_args, **call_kwargs)
                return init_cls, "initialised_cls"
            elif isinstance(node, FuncTemplate):
                (
                    call_args,
                    call_kwargs,
                    required_arg_nodes,
                    required_kwarg_nodes,
                ) = self.get_node_args(arg_nodes, kwarg_nodes, node)
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
            else:
                raise RuntimeError(f"Unknown node type: {type(node)}")

    def get_node_args(self, arg_nodes, kwarg_nodes, node):
        call_args = node.call_args
        call_kwargs = node.call_kwargs
        required_arg_nodes = []
        required_kwarg_nodes = {}
        for arg_node in arg_nodes:
            if arg_node[1] == "Result":
                required_arg_nodes.append((arg_node[0], False))
            else:
                required_arg_nodes.append((arg_node[0], True))
        for key, kwarg_node in kwarg_nodes.items():
            if kwarg_node[1] == "Result":
                required_kwarg_nodes[key] = (kwarg_node[0], False)
            else:
                required_kwarg_nodes[key] = (kwarg_node[0], True)
        return call_args, call_kwargs, required_arg_nodes, required_kwarg_nodes

    def create_process_graph(self, final_node):
        unpickled_node = dill.loads(final_node)
        final_node = self.init_node(unpickled_node)
        print("Graph created")

    def __call__(self, final_node):
        pickled_node = dill.dumps(final_node)
        for _ in range(4):
            process = multiprocessing.Process(target=self.create_process_graph, args=(pickled_node,))
            process.start()
        final_node, node_type = self.init_node(final_node)
        if node_type == "Node":
            return self.router(final_node, True)
        elif node_type == "Result":
            return self.router(final_node, False)
        else:
            raise RuntimeError(f"Unexpected node type: {node_type}")
