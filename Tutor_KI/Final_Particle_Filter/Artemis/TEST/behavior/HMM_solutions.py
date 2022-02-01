"""
@author: Qiao Qiao 2020.
"""

import numpy as np
import networkx as nx
import os
import random


# only the color in colorlist will be used
colorlist = ["pink","red","orange","yellow","green","blue","purple","grey"]

color_matrix = np.eye(len(colorlist)) * 0.5 + 0.2

def load_test_graph(filename):
    """Load the test graph from gpickle file

    :returns data: networkx graph

    """
    path = os.getcwd() + "/data/" + filename + ".gpickle"
    Graph = nx.read_gpickle(path)

    return Graph

class HiddenMarkovModel_solution:
    """ Hidden Markov Model
       
    Args:
        Graph: 2d NetworkX graph
        colorlist: a list of color name
        color_matrix: numpy array, conditional probability matrix
        transition_model: S * S numpy array (S: number of nodes)
        sensor_model: S * C numpy array (S: number of nodes, C: number of color types)
        prior: S * 1 numpy array (S: number of nodes)

    """
  
    def __init__(self, Graph,colorlist, color_matrix, transition_model= None, 
                 sensor_model= None, prior = None):
        
        self.Graph = Graph
        self.colorlist = colorlist
        self.color_matrix = color_matrix
        
        if transition_model != None:
            self.transition_model = transition_model
        else:
            self.transition_model = get_transition_model_solution(Graph)
        if sensor_model != None:
            self.sensor_model = sensor_model
        else:
            self.sensor_model = get_sensor_model_solution(Graph,colorlist,color_matrix)
        if prior != None:    
            self.prior = prior
        else:
            self.prior = get_prior_solution(Graph)


def get_prior_solution(Graph):
    
    """ given the NetworkX 2d graph, return the prior matrix.
        The robot can start from any position with same probability.
    
    Args:
        Graph: 2d NetworkX graph
        
    Return: 
        prior: S * 1 numpy array (S: number of nodes)

    """
    number_node = len(Graph.nodes())
    #######################################
    # YOUR CODE HERE
    prior = np.full((number_node,1), 1 / number_node)
    #######################################
    return prior


def get_transition_model_solution(Graph):
    """ given the NetworkX 2d graph, return the transition model matrix.
       
    Args:
        Graph: 2d NetworkX graph
        
    Return: 
        transition_model: S * S numpy array (S: number of nodes)

    """
    
    # An adjacency matrix is a square matrix used to represent a finite graph. 
    # The elements of the matrix indicate whether pairs of vertices are adjacent 
    # or not in the graph. 
    
    # adjacency_matrix is a SciPy sparse matrix. We convert it to a dense matrix.
    adjacency_matrix = nx.adjacency_matrix(Graph).todense()
    
    # transform SciPy matrix to numpy array
    adjacency_matrix = np.asarray(adjacency_matrix)
    
    #######################################
    # YOUR CODE HERE
    transition_model = adjacency_matrix / adjacency_matrix.sum(axis = 1, keepdims=1)
    #######################################
    
    return transition_model


def get_sensor_model_solution(Graph, colorlist, color_matrix):
    """ given the NetworkX 2d graph, colorlist, color_matrix, return the sensor model matrix.
       
    Args:
        Graph: 2d NetworkX graph
        colorlist: a list of color name
        color_matrix: numpy array, conditional probability matrix
        
    Return: 
        sensor_model: S * C numpy array (S: number of nodes, C: number of color types)

    """
    
    sensor_model = np.zeros((len(Graph.nodes()),len(colorlist)))
    
    for num,(node,node_color) in enumerate(Graph.nodes.data('color')):
        #######################################
        # YOUR CODE HERE
        for num_color, color in enumerate (colorlist):  
            if (node_color == color):
                sensor_model[num,:] += color_matrix[num_color,:]
        #######################################
    return sensor_model

def return_most_likely_trajectory_solution(HMM, obs_trajectory):
    """use Viterbi algorithm to get the most likely trajectory of robot 
       given observed color sequence
    
    Args:
        HMM: an object of class HiddenMarkovModel
        obs_trajectory: observed color sequence, a list of color name
    
    Return:
        most_likely_trajectory: a list of labels, e.g. [0 2 1 3]
        most_likely_trajectory and obs_trajectory should has the same length 
    """
    #######################################
    # YOUR CODE HERE
    # TO DO： implement the Viterbi algorithm to get the most likely trajectory
    
    colorlist = HMM.colorlist
    most_likely_trajectory = np.zeros(len(obs_trajectory),dtype = int)
    
    # store the probability of each state for detecting each color
    prob_states = np.zeros((HMM.transition_model.shape[0],len(obs_trajectory)))
    
    #store which previous state is connected to current state
    index_matrix = np.zeros((HMM.transition_model.shape[0],len(obs_trajectory)))
    
    # for every observation
    for i in range (len(obs_trajectory)):
        color = obs_trajectory[i]
        color_index = colorlist.index(color)
        if i == 0:  # first observation
            fi = HMM.transition_model * HMM.prior
        else:       
            fi = HMM.transition_model * prob_states[:,i-1][np.newaxis].T
            
        # maximal probability product for reaching state    
        fi_max = np.amax(fi, axis = 0)
        
        # store the index of state that has maximal probability for backtracking
        index_matrix[:,i] = np.argmax(fi, axis = 0)
        
        # dot product 
        prob_state_i = HMM.sensor_model[:,color_index] * fi_max
        
        # normalization can also be ignored
        prob_state_i = prob_state_i / prob_state_i.sum()
        
        prob_states[:,i] = prob_state_i
        
    
    # start with the most likely final state
    most_likely_trajectory[-1] = np.argmax(prob_states[:,-1])
    
    # backtracking along the path which maximized the probability of the final state
    for i in range(len(obs_trajectory)-2,-1,-1):    
        most_likely_trajectory[i] = index_matrix[most_likely_trajectory[i+1],i+1]
    #######################################
    return most_likely_trajectory


def randomly_generate_obs_trajectory(Graph,n):
    """ randomly generate a trajectory with length of n
    
    Args:
        Graph: 2d NetworkX graph
        n: the length of the trajectory
    
    Return:
        truth_trajectory_lables: a list of labels, e.g. [0, 2, 1, 3]
        obs_trajectory: a list of color name 
    """
    
    number_node = len(Graph.nodes)
    # list of node
    truth_trajectory = []
    # list of node color
    obs_trajectory =[]
    # list of node label
    truth_trajectory_lables = []
    
    labels_inv = Graph.graph['labels_inv']
    labels = Graph.graph['labels']
    
    # randomly choose the first node
    first = labels_inv[random.choice(range(0,number_node-1))]
    # add the node at the end of the list
    truth_trajectory.append(first)
    obs_trajectory.append(Graph.nodes[first]['color'] )
    
    for i in range(0,n-1):
        
        # get the last node in trajectory
        last_node = truth_trajectory[-1]
        # get the neighbors of the last node
        neighbors = [nb for nb in Graph.neighbors(last_node)]
        # randomly choose a node in the neighbors of the last node
        new_node = random.choice(neighbors)
        
        # add the node at the end of the list
        truth_trajectory.append(new_node)
        obs_trajectory.append(Graph.nodes[new_node]['color'] )
        
    # convert the list with node labels to the list of node
    for node in truth_trajectory:
        truth_trajectory_lables.append(labels[node])
        
    return truth_trajectory_lables, obs_trajectory


