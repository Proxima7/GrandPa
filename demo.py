from grandpa.template_parser import TemplateParser
from nodes import test_pipeline

if __name__ == "__main__":
    pipeline = test_pipeline
    parser = TemplateParser()
    pipeline_result = parser(pipeline)
    print()
