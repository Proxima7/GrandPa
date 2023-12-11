from grandpa import Component, Node, GeneratorNode, Workflow, TemplateParser
from generator_example_nodes import generator_workflow

def main():
    parser = TemplateParser()
    pipeline_result = parser(generator_workflow)
    print(f"Result of generator_workflow: {pipeline_result}")


if __name__ == "__main__":
    main()




