# -*- coding: utf-8 -*-
"""
Created on Tue Dec  8 11:13:32 2020

@author: Riemann

Make assertions between the notebook and sample functions
"""
import unittest
import numpy as np
from timeout_decorator import *
import behavior.PF_solution as solution
from behavior.PF_solution import particle_filtering as particle_filtering_solution
import copy

try:
    import assignment.Particle_Filter as PF
    importFlag = True
except:
    importFlag = False

from assignment.Particle_Filter import particle_filtering

class TestNotebook(unittest.TestCase):
    TIMEOUT_CONSTANT = 240
    TIME_ERROR = 'Timeout Error. %s seems to take more than 4m to execute.' \
                 'There is most likely an infinite loop or very inefficient method implemented.'
    
    def setUp(self):
        self.width = 800
        self.height = 600
        self.seed=200
        # np.random.seed(self.seed)
        self.particles = solution.uniform_particles_construction(self.width, self.height, 100)

        self.centers = []
        self.radii = []
        self.trajectory = []
        self.velocity = []
        self.distance = []
        self.weights = []

        self.uncertainty_motion_model = {'no noise': 1.1, 'low noise': 7.1, 'high noise': 12.1}
        self.std_sensor = 3.0

        for i in range(10):
            self.centers.append(np.load('./data/PF_data/centers/centers_%s.npy' %i))
            self.radii.append(np.load('./data/PF_data/radii/radii_%s.npy' %i))
            self.trajectory.append(np.load('./data/PF_data/trajectory/trajectory_%s.npy' %i))
            self.velocity.append(np.load('./data/PF_data/velocity/velocity_%s.npy' %i))
            self.distance.append(np.load('./data/PF_data/distance/distance_%s.npy' %i))
            self.weights.append(np.load('./data/PF_data/weights/weights_%s.npy' %i))

    # @timeout_decorator.timeout(TIMEOUT_CONSTANT, timeout_exception=TimeoutError, exception_message=TIME_ERROR % ('uniform_particles_construction()'))
    # def test_uniform_particles_construction(self):
    #     particles = PF.uniform_particles_construction(self.width, self.height, 100)
    #     if np.any(particles[:, 0]) < 0 or np.any(particles[:, 0]) > self.width:
    #         raise ValueError("There are particles located beyond the width restriction")
    #     if np.any(particles[:, 1]) < 0 or np.any(particles[:, 1]) > self.height:
    #         raise ValueError("There are particles located beyond the height restriction")


    @timeout_decorator.timeout(TIMEOUT_CONSTANT, timeout_exception=TimeoutError, exception_message=TIME_ERROR % ('find_valid_particles()'))    
    def test_find_valid_particles(self):
        for i in range(10):

            self.assertIsNone(np.testing.assert_array_equal(PF.find_valid_particles(self.particles, self.centers[i], self.radii[i]),\
                             solution.find_valid_particles(self.particles, self.centers[i], self.radii[i]) ))
    

    @timeout_decorator.timeout(TIMEOUT_CONSTANT, timeout_exception=TimeoutError, exception_message=TIME_ERROR % ('weights_update()'))
    def test_weights_update(self):
        std_sensor = 5.0
        for i in range(10):
            for j in range(len(self.distance[i])):
                self.assertIsNone(np.testing.assert_array_equal(PF.weights_update(self.particles, self.weights[i], self.distance[i][j], self.centers[i], self.radii[i], std_sensor), \
                    solution.weights_update(self.particles, self.weights[i], self.distance[i][j], self.centers[i], self.radii[i], std_sensor) ))
    


    @timeout_decorator.timeout(TIMEOUT_CONSTANT, timeout_exception=TimeoutError, exception_message=TIME_ERROR % ('resample_procedure()'))
    def test_resample_procedure(self):
        for i in range(4):
            np.random.seed(self.seed + 1)
            u = np.random.uniform()
            dummy_particles = np.arange(start=0, stop=len(self.weights[i]), step=0.5).reshape([-1,2])
            dummy_particles2 = np.arange(start=0, stop=len(self.weights[i]), step=0.5).reshape([-1,2])
            t1, id1 = PF.resample_procedure(dummy_particles, self.weights[i], u0=u+0.0)
            t2, id2 = solution.resample_procedure(dummy_particles2, self.weights[i], u0=u+0.0)
            self.assertIsNone(np.testing.assert_array_equal(t1, t2))
            self.assertIsNone(np.testing.assert_array_equal(id1, id2))
    

    @timeout_decorator.timeout(TIMEOUT_CONSTANT, timeout_exception=TimeoutError, exception_message=TIME_ERROR % ('particle_filtering()'))
    def test_particle_filtering(self):
        for i in range(3):
            noise = self.uncertainty_motion_model['low noise']
            np.random.seed(4001)
            test_trajectory, test_estimates, test_weights = particle_filtering(copy.deepcopy(self.width), copy.deepcopy(self.height), 600, copy.deepcopy(self.centers[i]), copy.deepcopy(self.radii[i]), copy.deepcopy(self.velocity[i]), copy.deepcopy(self.distance[i]), noise, self.std_sensor)
            # test_estimates = copy.deepcopy(test_estimates[test_estimates[:,0].argsort()][:,0])
            # np.random.seed(4001)
            # sample_trajectory, sample_estimates, sample_weights = particle_filtering_solution(copy.deepcopy(self.width), copy.deepcopy(self.height), 600, copy.deepcopy(self.centers[i]), copy.deepcopy(self.radii[i]), copy.deepcopy(self.velocity[i]), copy.deepcopy(self.distance[i]), noise, self.std_sensor)
            # sample_estimates = copy.deepcopy(sample_estimates[sample_estimates[:,0].argsort()][:,0])

            # allclose(actual, desired, rtol, atol) compares the difference between actual and desired to atol + rtol * abs(desired)
            # for ii in range(sample_estimates.shape[0]):
            # self.assertTrue(np.allclose(sample_trajectory, test_trajectory, atol=0.01, rtol=1e-2) )
            #     try:
            # self.assertIsNone(np.testing.assert_allclose(sample_estimates[:ii,1], test_estimates[:ii,1], atol=1., rtol=1e-2) )
            self.assertIsNone(np.testing.assert_allclose(test_estimates[-1], self.trajectory[i][-1], atol=70., rtol=1e-2) )



