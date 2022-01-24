import unittest
from timeout_decorator import *
from behavior.HMM_solutions import *
import numpy as np

try:
    import assignment.HMM as HMM
    importFlag = True
except:
    importFlag = False



class TestNotebook(unittest.TestCase):
    TIMEOUT_CONSTANT = 240
    TIME_ERROR = 'Timeout Error. %s seems to take more than 4m to execute. Please keep the computational complexity in mind.'
        
    def setUp(self):
        self.test_data = []
        self.test_data.append(load_test_graph("3_3"))
        self.test_data.append(load_test_graph("5_3"))
        self.test_data.append(load_test_graph("8_6"))
        self.test_data.append(load_test_graph("5_5"))
        self.test_data.append(load_test_graph("10_5"))
        colorlist = ["pink", "red", "orange", "yellow", "green", "blue", "purple", "grey"]
        color_matrix = np.eye(len(colorlist)) * 0.5 + 0.2

    @timeout_decorator.timeout(TIMEOUT_CONSTANT, timeout_exception=TimeoutError, exception_message=TIME_ERROR % ('get_prior()'))
    def test_get_prior_solution(self):
        for testdatai in self.test_data:
            a = HMM.get_prior(testdatai)
            b = get_prior_solution(testdatai)
            self.assertIsNone(np.testing.assert_array_equal(a, b))

    @timeout_decorator.timeout(TIMEOUT_CONSTANT, timeout_exception=TimeoutError, exception_message=TIME_ERROR % ('get_transition_model()'))
    def test_get_transition_model(self):
        for testdatai in self.test_data:
            a = HMM.get_transition_model(testdatai)
            b = get_transition_model_solution(testdatai)
            self.assertIsNone(np.testing.assert_array_equal(a, b))


    @timeout_decorator.timeout(TIMEOUT_CONSTANT, timeout_exception=TimeoutError, exception_message=TIME_ERROR % ('get_sensor_model()'))
    def test_get_sensor_model(self):
        for testdatai in self.test_data:
            a = HMM.get_sensor_model(testdatai,colorlist, color_matrix)
            b = get_sensor_model_solution(testdatai,colorlist, color_matrix)
            self.assertIsNone(np.testing.assert_array_equal(a, b))


    @timeout_decorator.timeout(TIMEOUT_CONSTANT, timeout_exception=TimeoutError, exception_message=TIME_ERROR % ('return_most_likely_trajectory_solution()'))
    def test_return_most_likely_trajectory_solution(self):
        for testdatai in self.test_data:
            hmm_1 = HMM.HiddenMarkovModel(testdatai, colorlist, color_matrix)
            hmm_2 = HiddenMarkovModel_solution(testdatai, colorlist, color_matrix)
            n = random.choice(range(1,15))
            truth_trajectory_lables, obs_trajectory1 = randomly_generate_obs_trajectory(testdatai, n)
            a = HMM.return_most_likely_trajectory(hmm_1, obs_trajectory1)
            b = return_most_likely_trajectory_solution(hmm_2, obs_trajectory1)
            self.assertIsNone(np.testing.assert_array_equal(a, b))


# To test it locally
#if __name__ == '__main__':
#    unittest.main()