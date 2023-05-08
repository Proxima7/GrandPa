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


## Getting Started
To create a workflow using GrandPa, simply define your nodes as Python functions and use the @node decorator to register them with GrandPa. Then, use the @workflow decorator to define your workflow as a DAG, specifying the nodes and their dependencies. Finally, run the workflow using the execute method.

## Installation
To install GrandPa, simply run:
pip install grandpa

## visualization
To manage and edit your workflows in the browser push them to grandhub from your terminal run:
grandpa push "grandhub_project_id"
You can then open the project on grandhub.
