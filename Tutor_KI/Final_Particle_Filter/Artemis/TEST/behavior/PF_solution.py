"""
@author: Riemann Lee 2020.
"""

import copy
import numpy as np
import matplotlib.pyplot as plt

from scipy import stats
from ipywidgets import interactive
from matplotlib.patches import Circle


def uniform_particles_construction(width, height, N) -> np.ndarray:
    """Sample particles in 2D space with no landmarks randomly.

    Args:
        width, height: float, width (x) and height (y) of the area in which the robot moves
        N: int, number of particles
        seed: value for random seed
    Return:
        particles: numpy (np) array with particles centered around the origin with dimension particles.size = [N,2]
    """
    particles = np.random.uniform([0,0], [width, height], size=(N, 2))
    return particles


def find_valid_particles(particles, centers, radii):
    """Given randomly sampled particles, as well as centers and radii of landmarks, decide which particles are valid.

    Args:
        particles: the particles we get through random generation in 2D space
        centers: coordinate for each centers of landmarks
        radii: the radius of cicular landmarks 
    Return:
        valid_particles: the particles that locate outside the circular landmarks
    """
    valid_particles = []

    ###################################################################
    # YOUR CODE HERE
    for count, particle_coord in enumerate(particles):
        dis = np.linalg.norm(particle_coord - centers, axis=1, keepdims=True)
        if np.all(dis >= radii):
            valid_particles.append(particles[count])
            ###################################################################
    return np.asarray(valid_particles)


def propagate_particle_pos(particles, v, std):
    """Predict the position at the next time step for each particle given current angles and velocities.

    Args:
        particles: the particles we get after rejecting the ones that are not valid
        v： array with current measurements [angle, velocity]
        std: control the uncertainty of the prediction
    """
    N = len(particles)
    dt = 0.4
    # add some noise to the distance, the level is controlled by std
    delta_dist = (v[1] * dt)
    mean = (0, 0)

    # covariance for tangential and orthogonal uncertainty of the motion
    cov = [[std ** 2, 0], [0, (0.15 * std) ** 2]]
    x = np.random.multivariate_normal(mean, cov, N)

    # update the positions of all particles
    particles[:, 0] += np.cos(v[0]) * (delta_dist + x[:, 0]) - np.sin(v[0]) * x[:, 1]
    particles[:, 1] += np.sin(v[0]) * (delta_dist + x[:, 0]) + np.cos(v[0]) * x[:, 1]


def weights_update(particles, weights, dist_r_l, centers, radii, std_sensor):
    """Given the noised distances between robot and the landmarks, update the weights of particles

    Args:
        particles: coordinate of particles
        weights: weights of particles
        dist_r_l: the current distance between robot and landmarks
        centers: coordinate for each center of the landmarks
        radii: the radius of the circular landmarks
        std_sensor: standard deviation of the normal distribution that represents the sensor model

    Return:
        weights: array with updated weights for each particle

    """
    weights.fill(1.)

    # in order to avoid round-off to zero, we add eps to each weight before multiplying it
    # e.g.: weights *= weight + eps
    eps = 1.e-15

    for count, center in enumerate(centers):
        ###################################################################
        # YOUR CODE HERE
        # distance between the particles and each landmark
        dist_p_l = np.linalg.norm(particles - center, axis=1, keepdims=True) - radii[count]
        # weights_i = # to be computed
        weights_i = stats.norm.pdf(dist_p_l, dist_r_l[count], std_sensor)
        ###################################################################
        weights *= weights_i + eps

    weights += eps
    # normalize weights
    weights /= sum(weights) + eps
    return weights


def resample_procedure(x, weights, u0=np.random.uniform()):
    """Perform the resampling procedure described above

    Args:
        x: particles from which to resample, shape = [N,2]
        weights: the weights of the particles
        u0: random number which is drawn only once

    Return:
        xx: the resampled particles
        idx: indices of the particles in the resampled array xx (same order as in xx)
    """
    N = len(weights)

    u = u0 / N
    xx = np.zeros_like(x)
    idx = np.zeros(N, dtype='int')
    sumQ = copy.copy(weights[0])
    ###################################################################
    # YOUR CODE HERE
    i, j = 0, 0
    for i in range(N):
        while sumQ < u and j < N - 1:
            j += 1
            sumQ += weights[j]
        xx[i, :] = x[j, :]
        idx[i] = j
        u += 1 / N
    ###################################################################
    return xx, idx


def update_resampled_weights(weights, idx):
    weights[:] = weights[idx]
    weights /= np.sum(weights)


def particle_filtering(width, height, N, centers, radii, velocity, distance, std_prediction, std_sensor):
    """ Main method for particle filtering. Returns the estimated trajectories of N particles.

    Args:
        width, height: decide moving regions of the robot
        N: number of particles
        centers: centers of landmarks
        radii: radii of landmarks
        distance: measured distances from the robot to the landmarks
        std_prediction: uncertainty used for prediction
        std_sensor: standard deviation of the sensor model
    Return:
        list of positions of all particles, shape = [len(distance), N, 2],
        list particle weights, shape = [len(distance), N]
        final weights of particles, shape = [len(distance), 1]

    """
    # First we initialize the particles and the particle weights
    random_particles = uniform_particles_construction(width, height, N)
    particle_pos = find_valid_particles(random_particles, centers, radii)
    particle_weights = np.ones((len(particle_pos), 1))

    # Now we need to record the coordinates of moving particles
    particle_trajectories = [copy.copy(particle_pos)]
    estimate_positions = []

    for t in range(len(distance) - 1):
        ###################################################################
        # YOUR CODE HERE
        propagate_particle_pos(particle_pos, velocity[t], std_prediction)
        weights_update(particle_pos, particle_weights, distance[t + 1], centers, radii, std_sensor)
        particle_pos, random_index = resample_procedure(particle_pos, particle_weights)
        ###################################################################

        update_resampled_weights(particle_weights, random_index)
        particle_trajectories.append(copy.copy(particle_pos))
        estimate_positions.append(np.average(particle_pos, weights=particle_weights.flatten(), axis=0))

    particle_trajectory = np.asarray(particle_trajectories)
    estimate_positions = np.asarray(estimate_positions)
    return particle_trajectory, estimate_positions, particle_weights
