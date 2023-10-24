from grandpa import TemplateParser
from simple_nodes import remainder_workflow


def simple_demo():
    parser = TemplateParser()
    pipeline_result = parser(remainder_workflow)
    print(f"Result of remainder_workflow: {pipeline_result}")


if __name__ == "__main__":
    simple_demo()
