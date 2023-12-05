from grandpa import TemplateParser
from task_example_nodes import task_workflow


def main():
    parser = TemplateParser()
    pipeline_result = parser(task_workflow)
    print(f"Result of task workflow: {pipeline_result}")


if __name__ == "__main__":
    main()
