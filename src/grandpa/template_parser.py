import os
import random
from typing import Union

from grandpa.built_in_nodes import ValueWrapperNode, ResultWrapperNode, SettingReaderNode, \
    ListWrapperNode, DictWrapperNode, FuncExecNode
from grandpa.utils.standard import do_import, print_warning
from grandpa.routing import Router
import json


class TemplateParser:
    def __init__(self, template: dict, settings, router: Router, import_prefix: str = ""):
        self.main_router = router
        self.settings_reader = self.__init_settings_reader(settings)
        self.template = template
        self.outputs = self.__init_outputs(template)
        self.graph_def = self.__do_imports(template)
        self.initialized_nodes = []
        self.import_prefix = import_prefix

    def __call__(self):
        return self.parse()

    def parse(self):
        for node_address, node_definition in self.graph_def.items():
            if node_address not in self.initialized_nodes:
                self.init_node(node_address, node_definition)
        if self.outputs:
            return self.__get_output_node()
        else:
            return self.__auto_get_output_node()

    def __do_imports(self, graph_def):
        if "imports" in graph_def:
            assert type(graph_def["imports"]) is list, f"Imports must be given to the parser as a list of graph " \
                                                       f"definitions, not a single graph definition. If you want to " \
                                                       f"pass a single graph definition, use [graph_def]."
            for imp in graph_def["imports"]:
                graph_def_imp = self.__do_imports(imp)
                for node, definition in graph_def_imp.items():
                    if node not in graph_def:
                        graph_def[node] = definition
        graph_def = self.__del_obsolete_params(graph_def)
        return graph_def

    @staticmethod
    def __del_obsolete_params(graph):
        keys = ["settings", "outputs", "imports"]
        for key in keys:
            if key in graph:
                del graph[key]
        return graph

    def __init_outputs(self, graph_def):
        if 'outputs' in graph_def:
            outputs = graph_def['outputs']
            return outputs
        else:
            return None

    def __get_output_node(self):
        if type(self.outputs) is str:
            return self.__get_param_node(self.outputs, '')
        elif type(self.outputs) is dict:
            return self.__create_result_wrapper(self.outputs)
        else:
            raise TypeError(f'{type(self.outputs)} is not allowed as type to declare outputs. '
                            f'Use either string for a single output or dict for multiple outputs.')

    def __auto_get_output_node(self):
        output_node = None
        for node in self.graph_def:
            dependency_count = self.__calc_dependency_count(node)
            if dependency_count == 0:
                output_node = node
        if not output_node.startswith('//'):
            output_node = '//' + output_node
        return self.__get_param_node(output_node, '')

    def __calc_dependency_count(self, node):
        dependency_count = 0
        for node_name, graph_params in self.graph_def.items():
            for key in ['node_settings', 'params']:
                if key in graph_params:
                    for key_node in graph_params[key].values():
                        if type(key_node) is str and self.__get_node_name(key_node, node_name) == node:
                            dependency_count += 1
        return dependency_count

    def __create_result_wrapper(self, output_def: dict):
        result_wrapper = ResultWrapperNode(address=f'GenResultWrapper{random.getrandbits(128)}',
                                           main_router=self.main_router)
        for name, node in output_def.items():
            result_wrapper.add_node(name, self.__get_param_node(node, ''))
        return result_wrapper

    def init_node(self, node_address, node_definition, node_call_stack=None):
        if node_call_stack is None:
            node_call_stack = [node_address]
        else:
            self.__check_for_circular_definition(node_address, node_call_stack)
            node_call_stack.append(node_address)
        self.__make_node_ready_for_init(node_call_stack, node_definition, node_address)
        try:
            node = self.__import_node(node_definition['node'])
        except ModuleNotFoundError as e:
            raise GraphDefinitionError(
                f"Node {node_address} could not be imported from path {node_definition['node']}. "
                f"Please check the import path. Error raised: {e}")
        if 'node_settings' in node_definition:
            node_settings = self.__get_node_settings(node_address, node_definition['node_settings'])
        else:
            node_settings = {}
        initialized_node = node(**node_settings, address=node_address, main_router=self.main_router)
        if 'params' in node_definition:
            self.add_nodes(initialized_node, node_definition['params'], node_address)
        self.initialized_nodes.append(node_address)

    def add_nodes(self, initialized_node, params, node_address):
        for name, param in params.items():
            param_node = self.__get_param_node(param, node_address)
            initialized_node.add_node(name, param_node)
        return initialized_node

    def __get_param_node(self, param, node_address: str):
        if type(param) is str:
            node_get_dict = {
                '//': (self.main_router.get_instruction, {"path": param}),
                '/': (self.main_router.get_instruction, {"path": '//' + node_address + param}),
            }
            node_create_dict = {
                'settings:': (self.settings_reader, {"setting": param.replace('settings:', '')}),
                'node:': (self.main_router.get_instruction, {"path": param.replace('node:', '')
                if param.replace('node:', '').startswith('//') else '//' + node_address + param.replace('node:', '')}),
                'func:': (FuncExecNode,
                          {"func": param.replace('func:', ''), "address": f"GenFuncExecNode/{random.getrandbits(128)}",
                           "main_router": self.main_router})
            }
            for tag, node in node_get_dict.items():
                if param.startswith(tag):
                    return node[0](**node[1])
            for tag, node in node_create_dict.items():
                if param.startswith(tag):
                    return ValueWrapperNode(node[0](**node[1]), address=f"GenValueWrapper/{random.getrandbits(128)}",
                                            main_router=self.main_router)
        elif type(param) is list:
            return ListWrapperNode([self.__get_param_node(p, node_address) for p in param],
                                   address=f"GenListWrapper/{random.getrandbits(128)}", main_router=self.main_router)
        elif type(param) is dict:
            return DictWrapperNode({key: self.__get_param_node(p, node_address) for key, p in param.items()},
                                   address=f"GenDictWrapper/{random.getrandbits(128)}", main_router=self.main_router)
        return ValueWrapperNode(param, address=f"GenValueWrapper/{random.getrandbits(128)}",
                                main_router=self.main_router)

    def __get_node_settings(self, node_address, def_node_settings: dict):
        node_settings = {}
        for name, setting in def_node_settings.items():
            if type(setting) is str and setting.startswith('**'):
                node_settings = {**self.__resolve_setting(node_address, setting), **node_settings}
            else:
                node_settings[name] = self.__resolve_setting(node_address, setting)
        return node_settings

    def __resolve_setting(self, node_address: str, setting):
        if type(setting) is str:
            return self.__resolve_settings_string(node_address, setting)
        elif type(setting) is list:
            return [self.__resolve_setting(node_address, sub_setting) for sub_setting in setting]
        elif type(setting) is dict:
            return {key: self.__resolve_setting(node_address, value) for key, value in setting.items()}
        else:
            return setting

    def __resolve_settings_string(self, node_address, setting):
        if setting.startswith('**'):
            setting.replace('**', '')
            return_dict = True
        else:
            return_dict = False
        tag_action_dict = {
            '//': (self.main_router.get_value, {"path": setting}),
            '/': (self.main_router.get_value, {"path": '//' + node_address + setting}),
            'settings:': (self.settings_reader, {"setting": setting.replace('settings:', '')}),
            'node:': (self.main_router.get_instruction, {"path": setting.replace('node:', '')
            if setting.replace('node:', '').startswith(
                '//') else '//' + node_address + setting.replace('node:', '')}),
            'func:': (FuncExecNode, {"func": setting.replace('func:', ''),
                                     "address": f"GenFuncExecNode/{random.getrandbits(128)}",
                                     "main_router": self.main_router})
        }
        for tag, action in tag_action_dict.items():
            if setting.startswith(tag):
                return action[0](**action[1])
        if return_dict:
            return {**setting}
        else:
            return setting

    def __import_node(self, node: str):
        if '.' in node:
            return do_import(self.import_prefix + node)
        else:
            return do_import('grandpa.built_in_nodes.' + node)

    @staticmethod
    def __check_for_circular_definition(node_address, node_call_stack):
        if node_address in node_call_stack:
            # raise CircularGraphError(node_call_stack)
            pass

    def __make_node_ready_for_init(self, node_call_stack, node_definition: dict, node_address):
        for node_def_key in ['node_settings', 'params']:
            if node_def_key in node_definition:
                self.__ensure_that_entry_is_dict(node_definition[node_def_key], node_def_key)
                required_uninitialised_nodes = self.__get_required_uninitialised_notes(node_definition[node_def_key],
                                                                                       node_address)
                if required_uninitialised_nodes:
                    for req_uninit_node in required_uninitialised_nodes:
                        self.init_node(req_uninit_node, self.graph_def[req_uninit_node], node_call_stack)

    def __get_required_uninitialised_notes(self, requirements: dict, node_address):
        required_uninitialised_nodes = []
        for req in requirements.values():
            self.__append_node_name_if_required(node_address, req, required_uninitialised_nodes)
        return required_uninitialised_nodes

    def __append_node_name_if_required(self, node_address, requirement, required_uninitialised_nodes):
        if type(requirement) is str:
            node_name = self.__get_node_name(requirement, node_address)
            if self.__is_node(requirement, node_name):
                if node_name not in self.initialized_nodes:
                    required_uninitialised_nodes.append(node_name)
        elif type(requirement) is list:
            for req in requirement:
                self.__append_node_name_if_required(node_address, req, required_uninitialised_nodes)
        elif type(requirement) is dict:
            for k, req in requirement.items():
                self.__append_node_name_if_required(node_address, req, required_uninitialised_nodes)

    def __is_node(self, req, node_name):
        if 'node:' in req:
            self.__ensure_that_node_is_in_graph_def(node_name, req)
            return True
        elif '/' in req:
            return self.__check_if_name_in_graph_def(node_name)
        else:
            return False

    def __check_if_name_in_graph_def(self, node_name):
        if node_name in self.graph_def:
            return True
        else:
            return False

    def __ensure_that_node_is_in_graph_def(self, node_name, req):
        if node_name == "runtime":
            return

        assert node_name in self.graph_def, \
            f'Node {node_name} was directly referenced in graph def as {req}, but there is no node with the ' \
            f'name {node_name} defined.'

    @staticmethod
    def __get_node_name(node_ref: str, node_address: str):
        if type(node_ref) is str:
            node_ref = node_ref.replace('node:', '')
            if not node_ref.startswith('//'):
                return node_address + node_ref
            else:
                return node_ref[:2].replace('/', '') + node_ref[2:]
        else:
            return None

    @staticmethod
    def __ensure_that_entry_is_dict(entry, name: str):
        assert type(entry) is dict, f'{name} in graph definition must be defined as a dict, not a {type(entry)}.'

    def __init_settings_reader(self, settings: Union[dict, str]):
        if type(settings) is str:
            if not os.path.exists(settings):
                print_warning(f'Graph parser tried to load settings form file {settings}, '
                              f'but the file does not exist. Please check the file path.')
                return None
            return SettingReaderNode(json.load(open(settings)), address='SettingsReader', main_router=self.main_router)
        elif type(settings) is dict:
            return SettingReaderNode(settings, address='SettingsReader', main_router=self.main_router)
        elif settings is not dict:
            print_warning(f'Graph parser was called with settings of type {type(settings)} - however, '
                          f'this type is not supported, the settings must be either given as a file path or a dict.')
        else:
            print_warning(f'No settings file was handed to the graph parser - all settings will therefore be '
                          f'initialized with their default values.')

        return None


class CircularGraphError(Exception):
    def __init__(self, call_stack: list):
        self.message = f"You have successfully created a circular graph. This means the graph you defined is " \
                       f"essentially like a dog, it will attempt to chase its tail but never catch up to it. The " \
                       f"call order the parser generated from your graph definition was {' -> '.join(call_stack)}, " \
                       f"so in order to fix this problem, you will need to change the node_settings or params of one " \
                       f"of these nodes."
        super().__init__(self.message)


class CircularImportError(Exception):
    def __init__(self, call_stack: list):
        self.message = f"You have successfully created a circular import. This means the imports you defined are " \
                       f"importing each other, creating an endless loop. The import order careated was " \
                       f"{' -> '.join(call_stack)}, so in order to fix this problem, you will need to remove one of " \
                       f"these imports."
        super().__init__(self.message)


class GraphDefinitionError(Exception):
    def __init__(self, message):
        super().__init__(message)
