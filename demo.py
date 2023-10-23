import time

from grandpa.template_parser import TemplateParser
from nodes import run_sum_array, sum_array_workflow, process_images, process_images_workflow


def compare_sum_array():
    start = time.time()
    run_sum_array()
    end = time.time()
    print(f"Time for sum_array without framework: {end - start}")
    parser = TemplateParser()
    start = time.time()
    parser(sum_array_workflow)
    end = time.time()
    print(f"Time for sum_array with framework: {end - start}")


def compare_process_images():
    start = time.time()
    process_images()
    end = time.time()
    print(f"Time for process_images without framework: {end - start}")
    parser = TemplateParser()
    start = time.time()
    parser(process_images_workflow)
    end = time.time()
    print(f"Time for process_images with framework: {end - start}")


if __name__ == "__main__":
    compare_sum_array()
    compare_process_images()
    print()
