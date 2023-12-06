from grandpa import TemplateParser
from simple_nodes import remainder_workflow, remainder_workflow_with_settings


def simple_demo():
    parser = TemplateParser()
    pipeline_result = parser(remainder_workflow)
    print(f"Result of remainder_workflow: {pipeline_result}")

    parser = TemplateParser()
    pipeline_result_with_settings = parser(remainder_workflow_with_settings, settings={"a": 5, "b": 10})
    print(f"Result of remainder_workflow with arguments: {pipeline_result_with_settings}")

if __name__ == "__main__":
    simple_demo()
