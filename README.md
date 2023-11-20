# GrandPa
![GrandPa Logo!](/grandpa_logo.png)

Build and run workflows at any scale. Graph-and-Parser is a python-based framework that allows for the intuitive creation of workflows and data projects as Directed Acyclic Graphs (DAGs). Run workflows efficient either locally in an IDE or on a large scale powered by Kubernetes (& Kubeflow).

Our promise for machine learning:
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

## Installation
To install GrandPa, simply run:

`` pip install grandpa ``

## Usage

### Define a Node
To define a Node, you can add the @Node decorator to any class or function. A class needs to have a __init__ and 
__call__ method defined.
```
@Node("node_name")
class Example1:
    def __init__(self, a, b):
        pass
        
    def __call__(self, *args, **kwargs):
        pass
        
        
@Node("node_name")
def example2(a, b):
    pass
```

### Define a Component
A component is a connected collection of nodes. It is defined by writing a method to connect the nodes to one another.
```
@Component("component_name")
def example_component(a, b):
    node1 = Example1(a, b)
    node2 = Example2(a, node1())
    return node2()
```

### Define a Workflow
A workflow is a connected collection of components and nodes. It is defined by writing a method to connect the 
components and nodes to one another. A workflow currently can not have any parameters.
```
@Workflow("workflow_name")
def example_workflow():
    a = 5
    b = 10
    component1 = ExampleComponent(a, b)
    node1 = Example1(a, component1())
    return node1()
```

### Execute a Graph
```
from grandpa import TemplateParser


parser = TemplateParser()
pipeline_result = parser(example_workflow)
```


## High level technical functionality

In general, GrandPa works based on a 3-steep process. Understand the high level functionality is crucial for 
understanding the code.

### Step 1: Template creation
The python code that needs to be executed is written by the user and annotated with the grandpa decorators. These 
decorators wrap the code into Grandpa Templates, which enables us to intercept any calls to the class/function.

### Step 2: Creating the DAG
The user not gives a workflow to the GrandPa TemplateParser. The Grandpa Parser then makes a call to the underlying 
function. However, since all/most parts of the function are Nodes (or Components, which technically are identical to 
pipelines), the code will not be directly executed. Instead, the GrandPa Templates will intercept the call and store
the information about the execution dependencies, which creates a DAG.

### Step 3: Executing the DAG
The execution of the DAG happens in a decentralised manner. This means that each node is responsible to gather its own
dependencies, and it does so by utilising the multiprocessing manager for parallelization. Therefore, in order to 
execute the entire DAG, only the last nodes needs to be called, and everything else happens "automatically".

### Bonus: How multiprocessing is implemented
As you might be aware, multiprocessing in Python can be prone to errors. For this reason, multiprocessing in Grandpa
does not directly interact with classes or functions. Instead, each process replicates the entire graph. In order to
request a result from a different process, the unique name of the node is passed to the process, which is then mapped
to the corresponding node in the local process graph.

