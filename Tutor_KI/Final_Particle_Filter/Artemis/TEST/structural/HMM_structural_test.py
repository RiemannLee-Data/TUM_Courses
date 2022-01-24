import unittest
from timeout_decorator import *
from structural import HMM_structural_helpers


try:
    import assignment.HMM as HMM
    importFlag = True
except:
    importFlag = False



class TestStructural(unittest.TestCase):
    TIMEOUT_CONSTANT = 180
    time_error = f"Importing the notebook took more than {TIMEOUT_CONSTANT} seconds. This is longer than expected. Please make sure that every cell is compiling and prevent complex structures."
    import_error = "There seems to be an error in the provided notebook. Please make sure that every cell is compiling without an error."
    method_error = "Function %s could not be found. Please don\'t rename the methods."
    library_error = "You can only use the given libraries, please check if you imported other libraries."

    @timeout_decorator.timeout(TIMEOUT_CONSTANT, timeout_exception=TimeoutError, exception_message=time_error)
    def test_notebook_import(self):
        if (importFlag is False):
            raise ImportError(self.import_error)
        else:
            pass

    def test_imported_librabries(self):
        self.assertIs(HMM_structural_helpers.check_imported_libraries(HMM),True,self.library_error)

    def test_check_function_names(self):
        self.assertIs(HMM_structural_helpers.check_for_function('get_prior', HMM), True,self.method_error % ('get_prior'))
        self.assertIs(HMM_structural_helpers.check_for_function('get_transition_model', HMM), True, self.method_error % ('get_transition_model'))
        self.assertIs(HMM_structural_helpers.check_for_function('get_sensor_model', HMM), True, self.method_error % ('get_sensor_model'))
        self.assertIs(HMM_structural_helpers.check_for_function('return_most_likely_trajectory', HMM), True,self.method_error % ('return_most_likely_trajectory'))


# To test it locally
if __name__ == '__main__':
    unittest.main()

