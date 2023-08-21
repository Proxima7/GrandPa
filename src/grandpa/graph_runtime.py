from copy import deepcopy

from grandpa.template_parser import TemplateParser
from grandpa.node import Node
from grandpa.routing import Router
from grandpa.utils.standard import print_warning


class GraphRuntime(Node):
    """
    Core of the framework. Manages all graph functionalities.
    """
    def __init__(self, **kwargs):
        self.router = Router()
        super().__init__(address="runtime", main_router=self.router)
        self.graphs = {}

    def add_graph(self, graph_name: str, template: dict, template_settings: dict, import_prefix: str = ""):
        """
        Adds a new graph to the runtime.

        Args:
            graph_name: Name of the graph.
            template: Template that should be used to create the graph.
            template_settings: Settings that should be used for the graph.
            import_prefix: Prefix that should be used when doing imports.

        Returns:
            Executable graph.
        """
        if graph_name in self.graphs:
            assert "GraphName {graph_name} already defined. Did you mean merge_graph?"

        new_graph = TemplateParser(template, template_settings, self.router, import_prefix=import_prefix)
        new_graph.initialized_nodes.append(self.address)
        self.graphs[graph_name] = new_graph

        return new_graph

    def get_graph_of_node(self, node):
        """
        Args:
            node: Node to get the graph of.

        Returns:
            Executable graph the given node is part of.
        """
        for graph in self.graphs.values():
            for graph_node in graph.initialized_nodes:
                graph_node = self.router.get_instruction('//' + graph_node)
                if node is graph_node:
                    return graph
        print_warning(f'{node.address} was not found in any graph of this runtime.')

    def update_graph(self, graph_to_update: str, update_with: str):
        """

        Args:
            graph_to_update: Graph to update.
            update_with: Graph to update the graph with.

        Returns:
            A graph that represents the new graph_to_update, a merge f both graphs.
        """
        graph_1 = self.graphs[graph_to_update]
        graph_2 = self.graphs[update_with]
        new_nodes, removed_nodes, nodes_with_changes = self.__get_differences(graph_1, graph_2)
        for node in new_nodes:
            graph_1.init_node(node.name, node.orig_node_def)
        for node in removed_nodes:
            self.router.delete_object_at_path('//' + node.name)
        for node in nodes_with_changes:
            if node.update_settings or node.update_setting_nodes:
                self.__reinitialize_node(node, graph_1)
            elif node.update_params or node.update_param_nodes:
                self.__update_node_params(node, graph_1)
        return graph_1

    def __update_node_params(self, node, graph):
        target_node = self.router.get_instruction('//' + node.name)
        target_node.nodes = {}
        graph.add_nodes(target_node, node.orig_node_def['params'], node.address)

    def __reinitialize_node(self, node, graph):
        self.router.delete_object_at_path('//' + node.name)
        graph.init_node(node.name, node.orig_node_def)

    def merge_graph(self, graph_name_1: str, graph_name_2: str, result_graph_name: str):
        """
        Merge two graphs using the graph names.

        Args:
            graph_name_1: Name of graph 1
            graph_name_2: Name of graph 2
            result_graph_name: Name of the resulting graph.

        Returns:
            A new graph, the result of merging graph1 and graph2.
        """
        graph_1 = deepcopy(self.graphs[graph_name_1])
        graph_2 = deepcopy(self.graphs[graph_name_2])
        new_nodes, _, nodes_with_changes = self.__get_differences(graph_1, graph_2)
        new_graph_def = graph_1.graph_def
        new_graph_def = self.__add_new_nodes(new_graph_def, new_nodes)
        new_graph_def = self.__copy_updated_nodes(new_graph_def, nodes_with_changes)
        self.add_graph(result_graph_name + '_temp', new_graph_def, graph_1.settings_reader.framework_settings)
        return self.update_graph(graph_name_1, result_graph_name + '_temp')

    def __copy_updated_nodes(self, new_graph_def, nodes_with_changes):
        for node in nodes_with_changes:
            new_graph_def[node.name + '_1'] = node.orig_node_def
        return new_graph_def

    def __add_new_nodes(self, new_graph_def, new_nodes):
        for node in new_nodes:
            new_graph_def[node.name] = node.orig_node_def
        return new_graph_def

    def __get_differences(self, graph_1: TemplateParser, graph_2: TemplateParser):
        graph_1_nodes = self.__get_graph_nodes(graph_1, 'graph1')
        graph_2_nodes = self.__get_graph_nodes(graph_2, 'graph2')
        self.__init_vnodes(graph_1_nodes)
        self.__init_vnodes(graph_2_nodes)
        new_nodes = self.__get_additional_nodes(graph_1_nodes, graph_2_nodes)
        removed_nodes = self.__get_additional_nodes(graph_2_nodes, graph_1_nodes)
        nodes_with_changes = self.__get_nodes_with_changes(graph_1_nodes, graph_2_nodes)
        return new_nodes, removed_nodes, nodes_with_changes

    def __init_vnodes(self, nodes):
        for node in nodes.values():
            node.connect_to_nodes(nodes)

    @staticmethod
    def __get_nodes_with_changes(graph_1_nodes, graph_2_nodes):
        nodes_with_changes = []
        for key, v_node in graph_2_nodes.items():
            if key in graph_1_nodes:
                if v_node.check_differences(graph_1_nodes[key]):
                    nodes_with_changes.append(v_node)
        return nodes_with_changes

    @staticmethod
    def __get_additional_nodes(original_nodes, new_nodes):
        additional_nodes = []
        for node, v_node in new_nodes.items():
            if node not in original_nodes:
                additional_nodes.append(v_node)
        return additional_nodes

    def __get_graph_nodes(self, graph: TemplateParser, graph_name: str):
        graph_def = graph.graph_def
        graph_def = self.__del_keys(graph_def)
        return {key: VirtualNode(key, node, graph_name) for key, node in graph_def.items()}

    @staticmethod
    def __del_keys(graph_def):
        del_keys = ['settings', 'imports', 'outputs']
        for key in del_keys:
            if key in graph_def:
                del graph_def[key]
        return graph_def

    def run(self, **params):
        pass

    def run_all(self):
        """
        Executes all registered graphs.

        Returns:
            None
        """
        for graph_name, graph in self.graphs.items():
            print(f"initializing graph {graph_name}.")
            initialized_graph = graph()
            print(f"executing graph {graph_name}")
            initialized_graph()

    def run_graph(self, graph_name: str):
        """
        Runs the graph with the given name.

        Args:
            graph_name: Name of the graph to run.

        Returns:
            The result given by the graph.
        """
        return self.graphs[graph_name]()

    def init_graph(self, graph: str):
        """
        Initialises the graph.

        Args:
            graph: Name of the graph.

        Returns:
            Initialised graph.
        """
        graph = self.graphs[graph]
        return graph()


class VirtualNode:
    """
    Helper class for graph merging.
    """
    def __init__(self, name, node_definition, graph):
        self.orig_node_def = deepcopy(node_definition)
        self.name = name
        self.graph = graph
        self.setting_nodes = {}
        self.param_nodes = {}
        self.node = deepcopy(node_definition)['node']
        self.node_settings = self.__get_node_settings(deepcopy(node_definition))
        self.params = self.__get_params(deepcopy(node_definition))
        self.update_settings = False
        self.update_params = False
        self.update_setting_nodes = False
        self.update_param_nodes = False
        self.differences_checked = False

    def check_differences(self, new_graph_node):
        if not self.differences_checked:
            self.update_settings = self.__compare(self.node_settings, new_graph_node.node_settings)
            self.update_params = self.__compare(self.params, new_graph_node.params)
            self.update_setting_nodes = self.__compare_nodes(self.setting_nodes, new_graph_node.setting_nodes)
            self.update_param_nodes = self.__compare_nodes(self.param_nodes, new_graph_node.param_nodes)
            self.differences_checked = True
        result = any([self.update_settings, self.update_params, self.update_setting_nodes, self.update_param_nodes])
        return result

    @staticmethod
    def __compare(own_dict, new_graph_dict):
        if not own_dict and not new_graph_dict:
            return False
        elif not own_dict or not new_graph_dict:
            return True
        else:
            for key, value in own_dict.items():
                if key not in new_graph_dict:
                    return True
                elif value != new_graph_dict[key]:
                    return True
            return False

    @staticmethod
    def __compare_nodes(own_nodes, new_graph_nodes):
        if not own_nodes and not new_graph_nodes:
            return False
        elif not own_nodes or not new_graph_nodes:
            return True
        else:
            for key, node in own_nodes.items():
                if key not in new_graph_nodes:
                    return True
                elif node.node != new_graph_nodes[key].node:
                    return True
                elif node.check_differences(new_graph_nodes[key]):
                    return True
            for key, node in new_graph_nodes.items():
                if key not in own_nodes:
                    return True
                elif node.node != own_nodes[key].node:
                    return True
                elif node.check_differences(own_nodes[key]):
                    return True
            return False

    def connect_to_nodes(self, nodes: dict):
        self.setting_nodes = self.__connect_nodes(nodes, self.setting_nodes)
        self.param_nodes = self.__connect_nodes(nodes, self.param_nodes)

    def __connect_nodes(self, nodes: dict, node_paths: dict):
        del_keys = []
        for key, node in node_paths.items():
            for m_node in nodes.values():
                if m_node.name == node:
                    node_paths[key] = m_node
                    break
                elif node in m_node.name and '/' in m_node.name:
                    node_paths[key] = m_node
                    break
            if type(node_paths[key]) is str:
                if node_paths[key] == 'GraphRuntime':
                    del_keys.append(key)
                else:
                    raise ValueError(f'{node} does not exist in {self.graph}.')
        for key in del_keys:
            del node_paths[key]
        return node_paths

    def __get_node_settings(self, node_definition: dict):
        tags = ['//', '/', 'node://', 'node:/']
        if 'node_settings' in node_definition:
            for key, setting in deepcopy(node_definition)['node_settings'].items():
                if type(setting) is str and any([setting.startswith(tag) for tag in tags]):
                    for tag in tags:
                        if setting.startswith(tag):
                            setting = setting.replace(tag, '', 1)
                    self.setting_nodes[key] = setting
                    del node_definition['node_settings'][key]
            return node_definition['node_settings']
        else:
            return None

    def __get_params(self, node_definition: dict):
        tags = ['//', '/', 'node://', 'node:/']
        if 'params' in node_definition:
            for key, param in deepcopy(node_definition)['params'].items():
                if type(param) is str and any([param.startswith(tag) for tag in tags]):
                    for tag in tags:
                        if param.startswith(tag):
                            param = param.replace(tag, '', 1)
                    self.param_nodes[key] = param
                    del node_definition['params'][key]
            return node_definition['params']
        else:
            return None
