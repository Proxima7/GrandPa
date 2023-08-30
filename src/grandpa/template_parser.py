from grandpa.node import Node
from grandpa.template_classes import (FuncTemplate, InitialisedNodeTemplate,
                                      NodeTemplate, ResultWrapper)


class TemplateParser:
    def __init__(self):
        self.initialised_nodes = {}

    def init_node(self, node):
        if isinstance(node, ResultWrapper):
            return self.init_node(node.origin)[0], "Result"
        else:
            arg_nodes = []
            for node in node.required_arg_nodes:
                if node not in self.initialised_nodes:
                    arg_nodes.append(self.init_node(node))
                else:
                    arg_nodes.append(self.initialised_nodes[node])
            kwarg_nodes = {}
            for key, node in node.required_kwarg_nodes.items():
                if node not in self.initialised_nodes:
                    kwarg_nodes[key] = self.init_node(node)
                else:
                    kwarg_nodes[key] = self.initialised_nodes[node]
            if isinstance(node, InitialisedNodeTemplate):
                (
                    call_args,
                    call_kwargs,
                    required_arg_nodes,
                    required_kwarg_nodes,
                ) = self.get_node_args(arg_nodes, kwarg_nodes, node)
                init_cls, _ = self.init_node(node.node_template)
                node = Node(
                    node.node_template.name,
                    init_cls,
                    call_args,
                    call_kwargs,
                    required_arg_nodes,
                    required_kwarg_nodes,
                )
                self.initialised_nodes[node] = node
                return node, "Node"
            elif isinstance(node, NodeTemplate):
                (
                    call_args,
                    call_kwargs,
                    required_arg_nodes,
                    required_kwarg_nodes,
                ) = self.get_node_args(arg_nodes, kwarg_nodes, node)
                for req_arg_node in required_arg_nodes:
                    call_args.append(req_arg_node)
                for key, req_kwarg_node in required_kwarg_nodes.items():
                    call_kwargs[key] = req_kwarg_node
                init_cls = node.cls(*call_args, **call_kwargs)
                return init_cls, "initialised_cls"
            elif isinstance(node, FuncTemplate):
                (
                    call_args,
                    call_kwargs,
                    required_arg_nodes,
                    required_kwarg_nodes,
                ) = self.get_node_args(arg_nodes, kwarg_nodes, node)
                node = Node(
                    node.name,
                    node.function,
                    call_args,
                    call_kwargs,
                    required_arg_nodes,
                    required_kwarg_nodes,
                )
                self.initialised_nodes[node] = node
                return node, "Node"
            else:
                raise RuntimeError(f"Unknown node type: {type(node)}")

    def get_node_args(self, arg_nodes, kwarg_nodes, node):
        call_args = node.call_args
        call_kwargs = node.call_kwargs
        required_arg_nodes = []
        required_kwarg_nodes = {}
        for arg_node in arg_nodes:
            if arg_node[1] == "Result":
                required_arg_nodes.append(arg_node[0])
            else:
                call_args.append(arg_node[0])
        for key, kwarg_node in kwarg_nodes.items():
            if kwarg_node[1] == "Result":
                required_kwarg_nodes[key] = kwarg_node[0]
            else:
                call_kwargs[key] = kwarg_node[0]
        return call_args, call_kwargs, required_arg_nodes, required_kwarg_nodes

    def __call__(self, final_node):
        final_node, node_type = self.init_node(final_node)
        if node_type == "Node":
            return final_node
        elif node_type == "Result":
            return final_node()
        else:
            raise RuntimeError(f"Unexpected node type: {node_type}")
