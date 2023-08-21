# GrandPa
![GrandPa Logo!](/grandpa_logo.png)

Build and run workflows at any scale. Graph-and-Parser is a python-based framework that allows for the intuitive creation of workflows and data projects as Directed Acyclic Graphs (DAGs). Run workflows efficient either locally in an IDE or on a large scale powered by Kubernetes (& Kubeflow).

Our promise for machine learning:
- More than 1000x faster training of models than on a local workstation.
- Fully solve training bottlenecks by distributing workloads in the cloud.
- Efficient and simple coding of machine learning pipelines.
- Train huge neural networks like GPT models efficiently in a Kubernetes cluster.
- Save development time and cost by reusing large parts of your code. Simply program in your IDE and have a one click deploy into a cluster.
  

## Features
<ul>
  <li>Intuitive DAG-based data workflow creation using Python code</li>
  <li>Efficient local execution of workflows within an IDE. </li>
  <li>Distributed execution of worflows on a larger scale using Kubernetes and Kubeflow.</li>
  <li>Write intuitively maintainable code within any IDE which runs anywhere</li>  
</ul>
We highly recommend to use GrandPa with other Grand* extensions like GrandML and GrandHub. GrandPa provides tools to access GrandHub.
<ul>
  <li>The GrandHub package manager and registry allows you the easy installation and reuse of workflows, nodes, and componentss</li>
  <li>Web-based visualization of workflows for easy editing and management of nodes and workflows.</li>
  <li>Intermediate file format for easy parsing and execution to other frameworks.</li>
</ul>

## Nodes, Components, Workflows and Projects

**Node**	A single unit of computation or operation that receives input and produces output. Nodes are the building blocks of a DAG.	In a machine learning context, a node might be a layer in a neural network that performs a specific mathematical operation.

**Component**	A collection of one or more nodes that perform a specific task or set of tasks. Components can be reused across workflows and projects.	A component for image classification might include multiple nodes to preprocess the image data, feed it into a neural network, and output the predicted class.

**Workflow**	A sequence of connected nodes or components that perform a specific task or set of tasks. Workflows define the flow of data between nodes or components.	A workflow for image classification might include a data loading node, a pre-processing component, a neural network component, and an output node to display the predicted class. Workflows can influence the behaviour of its nodes with eivnironment variables,  error handling, events and callbacks.

**Project**	A collection of workflows and components that together accomplish a larger goal or set of goals. Projects can be reused and shared across teams or organizations.	A machine learning project might include multiple workflows for different tasks (e.g. image classification, text classification, object detection) and a set of reusable components for data preprocessing, model training, and evaluation.


## Documentation

This development guide offers detailed information about how to use the framework. 
In addition, it offers information about contributing code, documentation, tests and more.

#### Getting started 

These snippets explain the basic usage of GrandPa:

Download a workflow from GrandHub. To do so, run in your cli:
`` grandpa install MNIST-classifier-mobilenetv2``

To execute the workflow locally run:
`` python -main.py ``
or open the main.py file of the workflow and run it in your IDE.

To execute a workflow in kubernetes run:
.....

To write custom nodes use:
To create a workflow using GrandPa, simply define your nodes as Python functions and use the @node decorator to register them with GrandPa. Then, use the @workflow decorator to define your workflow as a DAG, specifying the nodes and their dependencies. Finally, run the workflow using the execute method.

## Installation
To install GrandPa, simply run:

`` pip install grandpa ``

To install GrandPa only for local execution of workflows use:

`` pip install grandpa-local``

To install GrandPa core without any extensions like GrandML, GrandHub, Kubeflow run:

`` pip install grandpa-core``

Note that this version does not allow the execution of workflows in a cluster, offers no standard nodes frequently used in ML and has no GrandHub access.



### visualization
To manage and edit your workflows in the browser push them to grandhub from your terminal run:
grandpa push "grandhub_project_id"
You can then open the project on grandhub.


## Execution in Kubernetes: Setup

To execute workflows in a distributed environment it is required that you setup your kubernetes cluster first. 
1. Setup Kubeflow by following the instructions of Kubeflow. 
2. We provide terraform scripts for X.. Install simply with...


## Usage

### Define a Node
Each Node requires the super.init call to be made with the **kwargs parameter. 
This is used to pass the node's name and other parameters to the super class. 
A run method also needs to be defined. This is the method that will be called when the node is executed.
```
class Example(Node):
    def __init__(self, a, b, **kwargs):
        super().__init__(**kwargs)
        
    def run(self, *args, **kwargs):
        pass
```

### Define a Graph
A graph is a collection of nodes. The graph is responsible for executing the nodes in the correct order.
```
graph_def = {
    "__node_name__": {
        "node": "__node_import_path__",
        "node_settings": {
            "__init_param__": "__init_value__,
        }, 
        "params": {
            "__runtime_param__": "__param_value__",
        }
    }
}
```
An example looks like this:
```
graph_def = {
    "example": {
        "node": "example.Example",
        "node_settings": {
            "a": 1,
            "b": 2,
        }, 
        "params": {
            "c": 3,
        }
    }
}
```
Instead of hardcoding the values for the node settings and params, you can also reference other nodes.
```
graph_def = {
    "example": {
        "node": "example.Example",
        "node_settings": {
            "a": 1,
            "b": 2,
        }, 
        "params": {
            "c": 3,
        }
    },
    "example2": {
        "node": "example.Example",
        "node_settings": {
            "a": 1,
            "b": 2,
        }, 
        "params": {
            "c": "//example",
        }
    }
}
```
You can also pass a reference of another node using ```node://__node_name__```. This will pass the node object to the node's run or init method.
```
graph_def = {
    "example": {
        "node": "example.Example",
        "node_settings": {
            "a": 1,
            "b": "node://example2",
        }, 
        "params": {
            "c": 3,
        }
    },
    "example2": {
        "node": "example.Example",
        "node_settings": {
            "a": 1,
            "b": 2,
        }, 
        "params": {
            "c": 3,
        }
    }
}
```
Instead of hardcoding setting values in teh graph, you can also reference a value in the settings file using ```settings://__setting_name__```.
```
graph_def = {
    "example": {
        "node": "example.Example",
        "node_settings": {
            "a": 1,
            "b": "node://example2",
        }, 
        "params": {
            "c": "settings:example/c",
        }
    },
    "example2": {
        "node": "example.Example",
        "node_settings": {
            "a": 1,
            "b": 2,
        }, 
        "params": {
            "c": 3,
        }
    }
}
```
The corresponding settings file would look like this:
```
graph_settings: {
    "example": {
        "c": 3,
    }
}
```
### Execute a Graph
```
from grandpa import GraphRuntime
from graph_def import graph_def
from graph_settings import graph_settings


gr = GraphRuntime()
gr.add_graph("example", graph_def, graph_settings)
gr.init_graph("example")
result = gr.router.get_value("//example_node")
print(result)
```

#### Other useful features
##### Use tasks to automatically run jobs in parallel
```
class Example(Node):
    def __init__(self, a, b, **kwargs):
        super().__init__(**kwargs)
        
    def method_with_long_execution_time(self, *args, **kwargs):
        time.sleep(10)
        
    def run(self, *args, **kwargs):
        tasks = [self.switch.execute_task(self.method_with_long_execution_time) for i in range(10)]
        return [task.get_result() for task in tasks]  
```
##### Queue data in a worker queue to always have data available once the node is called
```
class Example(Node):
    def __init__(self, a, b, **kwargs):
        super().__init__(**kwargs)
        self.queue = self.switch.add_worker_queue(self.method_with_long_execution_time)
        
    def method_with_long_execution_time(self, *args, **kwargs):
        time.sleep(10)
        return 5
        
    def run(self, *args, **kwargs):
        return self.queue.get()
```




