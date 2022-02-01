import numpy as np
import copy


class Graph:
    def __init__(self):
        """
        Inits the graph, each graph has a list of names for the nodes and an adjacency matrix

        """
        self.m_names = []
        self.m_adj_mat = np.array([])

    def amount_of_nodes(self):
        """
        Returns the amount of currently used nodes in the graph
        :return: amount of nodes in the graph
        """
        return np.shape(self.m_adj_mat)[0]

    def add_nodes(self, list_of_nodes):
        """
        Add several nodes to the graph (see add_node())

        :param list_of_nodes: list of node names
        :return: None
        """
        for node in list_of_nodes:
            self.add_node(node)

    def add_node(self, node):
        """
        Adds a node to the graph, it is unconnected to all other nodes in the beginning

        :param node: the name of a node
        :return: None
        """
        if self.is_node(node):
            print('This node was added before!')
            return
        amount_of_nodes = self.amount_of_nodes()
        self.m_names.append(node)
        if amount_of_nodes > 0:
            temp = self.m_adj_mat
            self.m_adj_mat = np.zeros([amount_of_nodes + 1, amount_of_nodes + 1])
            self.m_adj_mat[:-1, :-1] = temp
            self.m_adj_mat[amount_of_nodes, amount_of_nodes] = 1  # self referencing of nodes
        else:
            self.m_adj_mat = np.ones([1, 1])

    def is_connected(self, node1, node2):
        """
        Returns true if two nodes are connected, the connection is directed this means
        the connection has to go from node1 to node2 to return true
        :param node1: the name of the first node (start)
        :param node2: the name of the second node (end)
        :return: if these two node are connected
        """
        if self.is_node(node1) and self.is_node(node2):
            id1, id2 = self.id(node1), self.id(node2)
            return self.m_adj_mat[id1, id2] != 0
        else:
            print('One of this nodes in unknown: ' + node1 + ', ' + node2)
        return False

    def connect(self, node1, node2):
        """
        Connects two nodes with each other, the connection is directed
        :param node1: the name of the first node (start)
        :param node2: the name of the second node (end)
        :return: None
        """
        if self.is_node(node1) and self.is_node(node2):
            id1, id2 = self.id(node1), self.id(node2)
            if self.m_adj_mat[id2, id1] == 0:
                self.m_adj_mat[id1, id2] = 1
            else:
                print('This connection was already set in the other direction!')
        else:
            print('One of this nodes in unknown: ' + node1 + ', ' + node2)

    def is_bi_connect(self, node1, node2):
        """
        Returns if the given nodes share a direct connection either way
        :param node1: the name of a node
        :param node2: the name of an other node
        :return: if these two nodes are directly connected at all
        """
        return self.is_connected(node1, node2) or self.is_connected(node2, node1)

    def id(self, node):
        """
        Returns the id for a given name, if the name is not used -1 is returned
        :param node: the name of a node
        :return: the id for a given name
        """
        if self.is_node(node):
            return self.m_names.index(node)
        return -1

    def name(self, node_nr):
        """
        Returns the name for a given id, if the id is not valid a error is produced!
        :param node_nr: the id of a node
        :return: the name of the node
        """
        return self.m_names[node_nr]

    def is_child_of(self, father, child, visited_list=None):
        """
        Test if a node, named child is the child of a node named father
        :param father: the name of the father node
        :param child: the name of the child node
        :param visited_list: unused parameter, do not use this parameter!
        :return: True if child is a child of the father
        """
        if visited_list is None:
            visited_list = []
        if self.is_connected(father, child):
            return True
        if self.is_node(father) and self.is_node(child):
            id1, id2 = self.id(father), self.id(child)
            children = np.nonzero(self.m_adj_mat[id1, :])[0]
            visited_list.append(id1)
            for test_child in children:
                if test_child not in visited_list:
                    if self.is_child_of(self.name(test_child), child, visited_list):
                        return True
        else:
            print('One of this nodes in unknown: ' + father + ', ' + child)
        return False

    def is_node(self, node):
        """
        Checks if the given node name is known in the graph
        :param node: name of the node
        :return: True if the name of the node is known
        """
        return node in self.m_names

    def print_graph(self):
        """
        Print the adjacency matrix of the graph
        """
        print(self.m_names)
        print(self.m_adj_mat)

    def _child_ids_bi(self, node_nr):
        """
        Generates all the ids of direct relatives of a node including this node
        :param node_nr: the id of the selected node
        :return: a numpy array full of node numbers
        """
        go_to = np.where(self.m_adj_mat[node_nr, :] > 0)
        come_from = np.where(self.m_adj_mat[:, node_nr] > 0)
        return np.unique(np.hstack((go_to, come_from)))

    def _type_of_connection(self, left, middle, right):
        """
        Returns the type of connection between the nodes:
            left ---- middle ---- right
        For example:
            left ---> middle <--- right, is head-to-head
            left <--- middle <--- right, is head-to-tail (order unimportant)
            left <--- middle ---> right, is tail-to-tail
        :param left: the name of the left node
        :param middle: the name of the middle node
        :param right: the name of the right node
        :return: returns a string with the name of the connection type
        """
        left_id, middle_id, right_id = self.id(left), self.id(middle), self.id(right)
        if self.m_adj_mat[left_id, middle_id] > 0:
            if self.m_adj_mat[right_id, middle_id] > 0:
                return "head-to-head"
            else:
                return "head-to-tail"  # order is not important here
        else:
            if self.m_adj_mat[right_id, middle_id] > 0:
                return "head-to-tail"  # order is not important here
            else:
                return "tail-to-tail"

    def _is_blocked(self, path, block_node):
        """
        Test if a path is blocked by a node block_node
        :param path: the path which should be tested
        :param block_node: the name of the block node
        :return: true if the path is blocked by the block node
        """
        for node in path[1:-1]:
            index = path.index(node)
            type = self._type_of_connection(path[index - 1], path[index], path[index + 1])
            if type == "head-to-head":
                if node != block_node:
                    # there is no directed path between node and block node
                    return len(self.generate_all_paths_between(node, block_node, directed=True)) == 0
            else:
                if node == block_node:
                    return True
        return False

    def d_separated(self, first, second, block_node):
        """
        Returns true if all paths from first to second are blocked by the block_node
        :param first: the name of the first node
        :param second: the name of the second node
        :param block_node: the name of the block node
        :return: true if all paths from first to second are blocked by the block_node
        """
        if block_node == first or block_node == second:
            print("The block node can't be the start or end node!")
            return False
        print('Test if ' + first + ' is d-separated from ' + second + ' by ' + block_node)
        paths = self.generate_all_paths_between(first, second)
        if len(paths) > 0:
            for path in paths:
                if not self._is_blocked(path, block_node):
                    return False
            return True
        else:
            print("No path between " + first + " and " + second)
            return False

    def generate_all_paths_between(self, node1, node2, visited_list=None, directed=False):
        """
        Generates all possible path between a node 'node1' and an other 'node2'
        :param node1: the name of the first node
        :param node2: the name of the second node
        :param visited_list: this parameter is used for recursion, don't use it!
        :param directed: if the connections on the way should be directed or not
        :return: a list of all possible paths between node1 and node2
        """
        if visited_list is None:
            visited_list = []
        if self.is_node(node1) and self.is_node(node2):
            id1, id2 = self.id(node1), self.id(node2)
            if id1 in visited_list:
                return []
            visited_list.append(id1)
            if id1 == id2:
                return [[node2]]
            if directed:
                children = np.nonzero(self.m_adj_mat[id1, :])[0]
            else:
                children = self._child_ids_bi(id1)
            all_paths = []
            for child in children:
                if child not in visited_list:
                    new_visited_list = copy.deepcopy(visited_list)
                    paths = self.generate_all_paths_between(self.name(child), node2, new_visited_list)
                    if len(paths) > 0:
                        if len(paths) > 1:
                            for small_ret in paths:
                                small_ret.append(node1)
                        else:
                            paths = paths[0]
                            paths.append(node1)
                        all_paths.append(paths)
            return all_paths
        else:
            print("One of this nodes in unknown: " + node1 + ", " + node2)
        return []


# init the graph
g = Graph()
# init all the nodes
g.add_nodes(['a', 'b', 'c', 'd', 'e'])
# connect all nodes (be aware of the direction!)
g.connect('a', 'b')
g.connect('c', 'b')
g.connect('c', 'd')
g.connect('b', 'e')

# visualize the graph as adjacency matrix
g.print_graph()

# Perform basic queries
print('Is a and b connected: ' + str(g.is_connected('a', 'b')))


def test_d_sep_for(first, second, block_node, graph):
    # Test if the nodes are d separated
    ret = graph.d_separated(first, second, block_node)
    if ret:
        print('Yes, they are\n')
    else:
        print('No, they are not\n')


test_d_sep_for('b', 'd', 'c', g)
test_d_sep_for('a', 'c', 'e', g)
test_d_sep_for('a', 'c', 'd', g)
test_d_sep_for('e', 'd', 'b', g)
test_d_sep_for('e', 'd', 'a', g)
