# GrandPa
![GrandPa Logo!](/grandpa_logo.png)
Build and run workflows at any scale. Graph-and-Parser is a python-based framework that allows for the intuitive creation of workflows and data projects as Directed Acyclic Graphs (DAGs). Run worfklows efficient either locally in an IDE or on a large scale powered by Kubernetes.

## Features
<ul>
<li>Intuitive DAG-based data workflow creation using Python code</li>
<li>Efficient local execution of workflows within an IDE or on a larger scale using Kubernetes and Kubeflow</li>
<li>Web-based visualization of workflows for easy editing and management of nodes and workflows.</li>
<li>Intermediate file format for easy parsing and execution to other frameworks.</li>
<li>GrandHub package manager and registry for easy installation and reuse of workflows, nodes, and components</li>
</ul>

## Nodes, Components, Workflows and Projects

**Node**	A single unit of computation or operation that receives input and produces output. Nodes are the building blocks of a DAG.	In a machine learning context, a node might be a layer in a neural network that performs a specific mathematical operation.

**Component**	A collection of one or more nodes that perform a specific task or set of tasks. Components can be reused across workflows and projects.	A component for image classification might include multiple nodes to preprocess the image data, feed it into a neural network, and output the predicted class.

**Workflow**	A sequence of connected nodes or components that perform a specific task or set of tasks. Workflows define the flow of data between nodes or components.	A workflow for image classification might include a data loading node, a pre-processing component, a neural network component, and an output node to display the predicted class. Workflows can influence the behaviour of its nodes with eivnironment variables,  error handling, events and callbacks.

**Project**	A collection of workflows and components that together accomplish a larger goal or set of goals. Projects can be reused and shared across teams or organizations.	A machine learning project might include multiple workflows for different tasks (e.g. image classification, text classification, object detection) and a set of reusable components for data preprocessing, model training, and evaluation.


## Getting Started
To create a workflow using GrandPa, simply define your nodes as Python functions and use the @node decorator to register them with GrandPa. Then, use the @workflow decorator to define your workflow as a DAG, specifying the nodes and their dependencies. Finally, run the workflow using the execute method.

## Installation
To install GrandPa, simply run:
pip install grandpa

## visualization
To manage and edit your workflows in the browser push them to grandhub from your terminal run:
grandpa push "grandhub_project_id"
You can then open the project on grandhub.
