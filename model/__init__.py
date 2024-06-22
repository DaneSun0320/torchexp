# module/__init__.py
import os
import importlib

def auto_import_modules():
    package_dir = os.path.dirname(__file__)
    package_name = __name__
    print("importing modules from", package_dir)
    for root, dirs, files in os.walk(package_dir):
        for file in files:
            if file.endswith('.py') and file != '__init__.py':
                module_name = os.path.join(root, file).replace(package_dir + os.sep, '').replace(os.sep, '.').rstrip('.py')
                importlib.import_module(f'{package_name}.{module_name}')

def initialize_modules():
    auto_import_modules()
