# -----------
# Test only
#
#






## The code below is going to print matrix as such
#
##[9, 8, 7, 6, 5, 4]
##[8, 7, 6, 5, 4, 3]
##[7, 6, 5, 4, 3, 2]
##[6, 5, 4, 3, 2, 1]
##[5, 4, 3, 2, 1, 0]
#
#def make_heuristic(grid, goal, cost):
#    heuristic = [[0 for row in range(len(grid[0]))] 
#                      for col in range(len(grid))]
#    for i in range(len(grid)):    
#        for j in range(len(grid[0])):
#            heuristic[i][j] = abs(i - goal[0]) + \
#                abs(j - goal[1])
#
#    return heuristic
#    
#
#
#grid = [[0, 1, 0, 0, 0, 0],
#        [0, 1, 0, 1, 1, 0],
#        [0, 1, 0, 1, 0, 0],
#        [0, 0, 0, 1, 0, 1],
#        [0, 1, 0, 1, 0, 0]]
#
#
#goal = [len(grid)-1, len(grid[0])-1]
#
#cost = 1
#
#result = make_heuristic(grid, goal, cost)
#
#for l in result:
#    print l
