import unittest
from timeout_decorator import *
from structural import PF_structural_helpers


try:
    import assignment.Particle_Filter as PF
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
        self.assertIs(PF_structural_helpers.check_imported_libraries(PF),True,self.library_error)        

    def test_check_function_names(self):
        # self.assertIs(PF_structural_helpers.check_for_function('uniform_particles_construction', PF), True, self.method_error % ('uniform_particles_construction'))
        self.assertIs(PF_structural_helpers.check_for_function('find_valid_particles', PF), True, self.method_error % ('find_valid_particles'))
        self.assertIs(PF_structural_helpers.check_for_function('weights_update', PF), True, self.method_error % ('weights_update'))
        self.assertIs(PF_structural_helpers.check_for_function('resample_procedure', PF), True, self.method_error % ('resample_procedure'))
        self.assertIs(PF_structural_helpers.check_for_function('particle_filtering', PF), True, self.method_error % ('particle_filtering'))