# ----------
# User Instructions:
# 
# Implement the function optimum_policy2D below.
#
# You are given a car in grid with initial state
# init. Your task is to compute and return the car's 
# optimal path to the position specified in goal; 
# the costs for each motion are as defined in cost.
#
# There are four motion directions: up, left, down, and right.
# Increasing the index in this array corresponds to making a
# a left turn, and decreasing the index corresponds to making a 
# right turn.

forward = [[-1,  0], # go up
           [ 0, -1], # go left
           [ 1,  0], # go down
           [ 0,  1]] # go right
forward_name = ['up', 'left', 'down', 'right']

# action has 3 values: right turn, no turn, left turn
action = [-1, 0, 1]
action_name = ['R', '#', 'L']

# EXAMPLE INPUTS:
# grid format:
#     0 = navigable space
#     1 = unnavigable space 
grid = [[1, 1, 1, 0, 0, 0],
        [1, 1, 1, 0, 1, 0],
        [0, 0, 0, 0, 0, 0],
        [1, 1, 1, 0, 1, 1],
        [1, 1, 1, 0, 1, 1]]

init = [4, 3, 0] # given in the form [row,col,direction]
                 # direction = 0: up
                 #             1: left
                 #             2: down
                 #             3: right
                
goal = [2, 0] # given in the form [row,col]

cost = [2, 1, 20] # cost has 3 values, corresponding to making 
                  # a right turn, no turn, and a left turn

# EXAMPLE OUTPUT:
# calling optimum_policy2D with the given parameters should return 
# [[' ', ' ', ' ', 'R', '#', 'R'],
#  [' ', ' ', ' ', '#', ' ', '#'],
#  ['*', '#', '#', '#', '#', 'R'],
#  [' ', ' ', ' ', '#', ' ', ' '],
#  [' ', ' ', ' ', '#', ' ', ' ']]
# ----------

# ----------------------------------------
# modify code below
# ----------------------------------------

def optimum_policy2D(grid,init,goal,cost):


    closed = [[0 for col in range(len(grid[0]))] for row in range(len(grid))]
    closed[init[0]][init[1]] = 1

    policy2D = [[" " for col in range(len(grid[0]))] for row in range(len(grid))]
    count = 0

    x = init[0]
    y = init[1]
    heading = init[2]  # 0 up, 1 left, 2 down, 3 right
    lastx = x
    lasty = y
    lastaction = 1  # 0 right, 1 no turn, 2 left
    f = cost[lastaction] 


    open = [[f, x, y, heading, lastx, lasty, lastaction]]

    found = False  # flag that is set when search is complete
    resign = False # flag set if we can't find expand
    
    while not found and not resign:
        if len(open) == 0:
            resign = True
            return "Fail"
        else:
            open.sort()
            open.reverse()
            next = open.pop()
            print "get item", next, "---", open
            
            x = next[1]
            y = next[2]
            heading = next[3]
            lastx = next[4]
            lasty = next[5]
            lastaction = next[6]
            #count += 1

            policy2D[lastx][lasty] = action_name[lastaction]
            
            if x == goal[0] and y == goal[1]:
                found = True
            else:
                for i in range(len(action)):
                    # take one action
                    # act2 = -1 right, 0 forward, 1 left
                    act2 = action[i]
                    
                    
                    # new heading,0 up, 1 left, 2 down, 3 right
                    # new heading = heading + act2 
                    # new heading = 0, act2 = -1, then new forward idx = -1 and just to 3  -> right
                    # new heading = 0, act2 = 1, then new forward idx = 1 -> left
                    heading2 = (heading + act2 + len(forward)) % len(forward)
                    
                    # note new heading (heading2) is also index of forward
                    
                    x2 = x + forward[heading2][0]
                    y2 = y + forward[heading2][1]

                    if x2 >= 0 and x2 < len(grid) and y2 >=0 and y2 < len(grid[0]):
                        if closed[x2][y2] == 0 and grid[x2][y2] == 0:
                            f2 = cost[i] 
                            open.append([f2, x2, y2, heading2, x, y, i])
                            print "put item", open
                            closed[x2][y2] = 1


    return policy2D


result = optimum_policy2D(grid,init,goal,cost)
for l in result:
    print l

