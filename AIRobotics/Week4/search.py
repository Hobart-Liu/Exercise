# ----------
# User Instructions:
# 
# Define a function, search() that returns a list
# in the form of [optimal path length, row, col]. For
# the grid shown below, your function should output
# [11, 4, 5].
#
# If there is no valid path from the start point
# to the goal, your function should return the string
# 'fail'
# ----------

# Grid format:
#   0 = Navigable space
#   1 = Occupied space

grid = [[0, 0, 1, 0, 0, 0],
        [0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 1, 0],
        [0, 0, 1, 1, 1, 0],
        [0, 0, 0, 0, 1, 0]]
init = [0, 0]
goal = [len(grid)-1, len(grid[0])-1]
cost = 1

delta = [[-1, 0], # go up
         [ 0,-1], # go left
         [ 1, 0], # go down
         [ 0, 1]] # go right

delta_name = ['^', '<', 'v', '>']

def search(grid,init,goal,cost):
    # ----------------------------------------
    # insert code here
    # ----------------------------------------
    path = "fail"
    
    row_max = len(grid) - 1
    col_max = len(grid[0]) - 1

    items = []
    item = [0] + init
    items.append(item)
    grid[0][0] = 1
#    print "put [%d %d %d] in items list, and start" %(item[0], item[1], item[2])
    
    for item in items:
        g_value = item[0]
        row = item[1]
        col = item[2]

        
        print "get item [%d %d %d]" %(item[0], item[1], item[2])
        
        # if it is not an end, set a mark
        if row == goal[0] and col == goal[1]:
            path = item
        else:
       
            for step in delta:
#                print step, 
                row1 = row + step[0]
                col1 = col + step[1]
                if 0 <= row1 <= row_max and 0 <= col1 <= col_max:
#                    print row1, col1
                    if grid[row1][col1] != 1:
                        print "put item [%d %d %d] in items list and mark it with 1" %(g_value+1, row1, col1)
                        grid[row1][col1] = 1
                        for l in grid:
                            print l
                        items.append([g_value+1, row1, col1])
#                else:
#                    print "out of bound"
        
     
#    for l in items:
#        print l

    return path


#items = ["a", "b", "c", "d"]
#for item in items:
#    if item == "a":
#        items.append("e")
#    print item


print search(grid, init, goal, cost)
