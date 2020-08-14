import bisect
"""
UCS uses path_cost to guide search policy

"""



infinity = 1.0e400


# ________________________________________________________________

class Graph:
    def __init__(self, dictionary=None, directed=True):
        self.dictionary = dictionary or {}
        self.directed = directed
        if not directed: self.make_undirected()

    def make_undirected(self):
        tmp = list(self.dictionary.keys())
        for a in tmp:
            for (b, distance) in self.dictionary[a].items():
                self.connect1(b, a, distance)

    def connect(self, A, B, distance=1):
        self.connect1(A, B, distance)
        if not self.directed: self.connect1(B, A, distance)

    def connect1(self, A, B, distance):
        self.dictionary.setdefault(A, {})[B] = distance

    def get(self, a, b=None):
        """Return a link distance or a dict of {node: distance} entries.
        .get(a,b) returns the distance or None;
        .get(a) returns a dict of {node: distance} entries, possibly {}."""
        links = self.dictionary.setdefault(a, {})
        if b is None:
            return links
        else:
            return links.get(b)

    def nodes(self, verbose=False):
        if verbose:
            return self.dictionary
        else:
            return self.dictionary.keys()


# Actually, this is record of solution, by tracking back its parents


class Node:
    """
    Actually, Node class is used to record solution
    Same child node can be created multiple times, due to different parents.
    By tracking back node's parents, it gives the solutions. 
    This is indeed space consuming solution. But so far, I have not found a better
    way to describe solution yet. 

    It also serves search algorithm by providing path_cost information.  
    """

    def __init__(self, state, parent=None, path_cost=0):
        self.state = state
        self.path_cost = path_cost
        self.parent = parent

    def path(self):
        "Create a list of nodes from the root to this node."
        x, result = self, [self]
        while x.parent:
            result.append(x.parent)
            x = x.parent
        return result

    def __repr__(self):
        return self.state

    def __eq__(self, other):
        return self.state == other.state


class Problem:
    def __init__(self, initial, goal, graph):
        self.graph = graph  # dictionary of graph
        self.initial = initial
        self.goal = goal

    def goal_test(self, node):
        return node == self.goal

    def expand(self, A):
        return [Node(next_state,
                     A,
                     (A.path_cost + self.graph.get(A.state, next_state) or infinity))
                for next_state in self.graph.get(A.state).keys()]


# ________________________________________________________________

class PriorityQueue:
    def __init__(self):
        self.A = []

    def append(self, items):
        for item in items:
            print("--> adding", item, item.path_cost)
            bisect.insort(self.A, (item.path_cost, item))

    def pop(self):
        # return with the lowest cost (UCS def)
        return self.A.pop(0)[1]  #[0] is path_cost

    def __repr__(self):
        return str(self.A)

    def __len__(self):
        return len(self.A)


def Search(problem, fringe):
    closed = {}
    fringe.append([problem.initial])

    while fringe:
        node = fringe.pop()

        print("check", node, "parent", node.parent, "Closed", closed)

        if problem.goal_test(node):
            return node
        if node.state not in closed:
            closed[node.state] = True
            fringe.append(problem.expand(node))

    return None


# ________________________________________________________________

'''
Test
'''
romania = Graph(dict(
    A=dict(Z=75, S=140, T=118),
    B=dict(U=85, P=101, G=90, F=211),
    C=dict(D=120, R=146, P=138),
    D=dict(M=75),
    E=dict(H=86),
    F=dict(S=99),
    H=dict(U=98),
    I=dict(V=92, N=87),
    L=dict(T=111, M=70),
    O=dict(Z=71, S=151),
    P=dict(R=97),
    R=dict(S=80),
    U=dict(V=142)), directed=False)

romania.locations = dict(
    A=(91, 492), B=(400, 327), C=(253, 288), D=(165, 299),
    E=(562, 293), F=(305, 449), G=(375, 270), H=(534, 350),
    I=(473, 506), L=(165, 379), M=(168, 339), N=(406, 537),
    O=(131, 571), P=(320, 368), R=(233, 410), S=(207, 457),
    T=(94, 410), U=(456, 350), V=(509, 444), Z=(108, 531))

# # code below is used to check undirected graph.
# ret =  romania.nodes(verbose=True)
# for k in ret.keys():
#     print(k, ret.get(k))

# # code below checks extending function of problem.
# problem = Problem('A','B',romania)
# R = Node('R')
# for node in problem.extend(R):
#     print(node.parent,"-->", node, node.path_cost)

problem = Problem(Node('A'), Node('B'), romania)

node = Search(problem, PriorityQueue())  # Priority on path_cost

for x in node.path():
    print(x.state, x.parent, x.path_cost)