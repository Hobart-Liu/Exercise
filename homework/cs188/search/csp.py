"""
Constraint Satisfaction Problems

Notation:
Name 'node' is used to present 'variable' 

test case:

use RGB to fill the color

G - H - I
| / | / |
D - E - F
| / | / |
A - B - C 



"""

import copy



class Node():
    """
    name        A name of of this variable
    domain      A list of possible values of this node [var, ...]
    """

    def __init__(self, name, domain):
        self._name = name
        self._value = None
        self._domain = domain
        self._successor = []

    def getName(self):
        return self._name

    def getValue(self):
        return self._value

    def setValue(self, value):
        self._value = value

    def getDomain(self):
        return self._domain

    def setSuccessor(self, nodes):
        self._successor = nodes

    def appendSuccessor(self, node):
        self._successor.append(node)

    def getSuccessor(self):
        return self._successor


    def __repr__(self):
        return "Node " + self._name


class FIFOQueue:
    def __init__(self):
        self._queue = []
        self._name = []

    def append(self, name, value):
        self._name.append(name)
        self._queue.append(value)

    def pop(self):
        name = self._name.pop()
        value = self._queue.pop()
        return name, value

    def popmin(self):
        idx = self._name.index(min(self._name))
        name = self._name.pop(idx)
        value = self._queue.pop(idx)
        return name, value

    def isEmpty(self):
        return True if len(self._queue)==0 else False

    def getList(self):
        return list(zip(self._name, [x.getValue() for x in self._queue]))

    def __len__(self):
        return len(self._queue)

    def __repr__(self):
        return str(self._name)


class CSPProblem():

    def __init__(self, nodes=None):
        self._nodes = FIFOQueue()
        self._constraints = {}
        self._constraints_all = []
        self._solution = FIFOQueue()


        if nodes:
            for name in nodes:
                self.addNode(nodes.get(name))

        self._nnodes = len(self._nodes)

        self._display = nodes  # save nodes reference for display

    def getNodes(self):
        return self._nodes
    
    def addNode(self, node):
        self._nodes.append(node.getName(), node)
        self._nnodes = len(self._nodes)

    def addConstraint(self, constraint, A, B):
        self._constraints.setdefault(A,{})[B] = constraint
        self._constraints.setdefault(B,{})[A] = constraint

    def addConstraint(self, constraint):
        self._constraints_all.append(constraint)
        
    def getConstraints(self):
        return self._constraints

    def getUnassignedNode(self):
        return self._nodes.popmin();

    def getSolution(self):
        if self.recursive_backtracking():
            return self._solution
        else:
            return None

    def recursive_backtracking(self):

        if self._nodes.isEmpty(): return True

        name, node = self.getUnassignedNode()

        for v in node.getDomain():
            node.setValue(v)
            if self.countConflicts(node) == 0:
                self._solution.append(node.getName(), node)
                self.display("set " + node.getName())

                if self.recursive_backtracking():
                    return True
                else:
                    self._solution.pop()



        node.setValue(None)
        self.addNode(node)
        self.display("roll back " + node.getName())


        return False


    def countConflicts(self,node):
        count = 0
        cons = self._constraints.get(node,[])
        for n in cons:
            count += cons[n](node, n)

        for f in self._constraints_all:
            for n in node.getSuccessor():
                count += f(node, n)

        return count


    def display(self, msg):
        val = [self._display.get(name).getValue() or ' ' for name in ['G', 'H', 'I', 'D', 'E', 'F', 'A', 'B', 'C']]

        print(msg.center(20, '-'))
        print("%s - %s - %s" %(val[0], val[1], val[2]))
        print("| / | / |")
        print("%s - %s - %s" %(val[3], val[4], val[5]))
        print("| / | / |")
        print("%s - %s - %s" %(val[6], val[7], val[8]))


domain_value = ['r','g','b']


class DefaultDict(dict):
    """Dictionary with a default value for unknown keys."""

    def __init__(self, default):
        self.default = default

    def __getitem__(self, key):
        if key in self: return self.get(key)
        return self.setdefault(key, copy.deepcopy(self.default))

    def __copy__(self):
        copy = DefaultDict(self.default)
        copy.update(self)
        return copy

def parse_neighbors(neighbors, vars=[]):
    """Convert a string of the form 'X: Y Z; Y: Z' into a dict mapping
    regions to neighbors.  The syntax is a region name followed by a ':'
    followed by zero or more region names, followed by ';', repeated for
    each region name.  If you say 'X: Y' you don't need 'Y: X'.
    >>> parse_neighbors('X: Y Z; Y: Z')
    {'Y': ['X', 'Z'], 'X': ['Y', 'Z'], 'Z': ['X', 'Y']}
    """
    dict = DefaultDict([])
    specs = [spec.split(':') for spec in neighbors.split(';')]
    for (A, Aneighbors) in specs:
        A = A.strip();
        dict.setdefault(A, [])
        for B in Aneighbors.split():
            dict[A].append(B)
            dict[B].append(A)
    return dict

# ret = parse_neighbors('NT: WA Q; NSW: Q V; T: ;SA: WA NT Q NSW V')

ret = parse_neighbors('A: B D E; B: C E F; C: F;D:E G H;E: F I H;F: I; H:G I')

print("G - H - I")
print("| / | / |")
print("D - E - F")
print("| / | / |")
print("A - B - C")

nodes ={}


for s in ret.keys():
    nodes.setdefault(s,Node(s, domain_value))


for name in nodes:
    node = nodes.get(name)
    neighbours = [nodes.get(s) for s in ret.get(node.getName())]
    node.setSuccessor(neighbours)


problem = CSPProblem(nodes)

def f(A, B):
    if A.getValue() == B.getValue():
        return 1
    else:
        return 0

problem.addConstraint(f)

solution = problem.getSolution()
print(solution.getList())
