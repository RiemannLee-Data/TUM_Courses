import inspect

def get_all_functions(module):
    functions = [i for i,_ in inspect.getmembers(module, inspect.isfunction)]
    return functions

def check_for_function(name, module):
    return True if name in get_all_functions(module) else False

def check_imported_libraries(module):
    libs = [i for i, _ in inspect.getmembers(module, inspect.ismodule)]
    libs_given = ['ipywidgets', 'np', 'nx', 'os', 'plt', 'random']
    flag = True
    for lib in libs:
        if lib not in libs_given:
            flag = False
    return flag