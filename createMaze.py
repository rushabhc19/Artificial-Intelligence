###TASK: Finding the path through a maze using Greedy Best-First Search and A∗search###

import sys
import stdio
import heapq
import collections
import math


class Maze(object):
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.walls = []
    
    def in_bounds(self, id):
        (x, y) = id
        return 0 <= x < self.width and 0 <= y < self.height
    
    def passable(self, id):
        return id not in self.walls
    
    def neighbors(self, id):
        (x, y) = id
        results = [(x+1, y), (x, y-1), (x-1, y), (x, y+1)]
        results = filter(self.in_bounds, results)
        results = filter(self.passable, results)
        return results
    
class MazeWithWeights(Maze):
    def __init__(self, width, height):
        super(MazeWithWeights, self).__init__(width, height)
        self.weights = {}
    
    def cost(self, from_node, to_node):
        return self.weights.get(to_node, 1)

####draw maze####

def draw_tile(graph, id, style, width):
    r = "."
    if 'number' in style and id in style['number']: r = "%d" % style['number'][id]
    if 'point_to' in style and style['point_to'].get(id, None) is not None:
        (x1, y1) = id
        (x2, y2) = style['point_to'][id]
        if x2 == x1 + 1: r = u"\u2192"
        if x2 == x1 - 1: r = u"\u2190"
        if y2 == y1 + 1: r = u"\u2193"
        if y2 == y1 - 1: r = u"\u2191"
    if 'start' in style and id == style['start']: r = "S"
    if 'goal' in style and id == style['goal']: r = "G"
    if 'path' in style and id in style['path']: r = "o"
    if id in graph.walls: r = u"\u254b" #u"\u2588"
    return r

def draw_grid(graph, width=2, **style):
    for y in range(graph.height):
        for x in range(graph.width):
            print"%%-%ds" % width % draw_tile(graph, (x, y), style, width),
        print "";
###A* algorithm###

class PriorityQueue:
    def __init__(self):
        self.elements = []
    
    def empty(self):
        return len(self.elements) == 0
    
    def put(self, item, priority):
        heapq.heappush(self.elements, (priority, item))
    
    def get(self):
        return heapq.heappop(self.elements)[1]

def reconstruct_path(came_from, start, goal):
    current = goal
    path = [current]
    while current != start:
        current = came_from[current]
        path.append(current)
    path.append(start) # optional
    path.reverse() # optional
    return path

#FOR A* ADVANCED 
def heuristic1(a, b):
    (x1, y1) = a
    (x2, y2) = b
    dx = abs(x1 - x2)
    dy = abs(y1 - y2)
    D = 2
    return D * math.sqrt(dx * dy + dx * dy)

#FOR A* AND GREEDY SEARCH
def heuristic2(a, b):
    (x1, y1) = a
    (x2, y2) = b
    return  abs(x1 - x2) + abs(y1 - y2)

def greedy_search(graph, start, goal):
    frontier = PriorityQueue()
    frontier.put(start, 0)
    came_from = {}
    cost_so_far = {}
    came_from[start] = None
    cost_so_far[start] = 0
    
    while not frontier.empty():
        current = frontier.get()

        
        if current == goal:
            break
        
        for next in graph.neighbors(current):
            new_cost = cost_so_far[current] + graph.cost(current, next)
            if next not in cost_so_far or new_cost < cost_so_far[next]:
                cost_so_far[next] = new_cost
                priority = heuristic2(goal, next)
                frontier.put(next, priority)
                came_from[next] = current

    
    return came_from, cost_so_far

def a_star_search1(graph, start, goal):
    frontier = PriorityQueue()
    frontier.put(start, 0)
    came_from = {}
    cost_so_far = {}
    came_from[start] = None
    cost_so_far[start] = 0
    
    while not frontier.empty():
        current = frontier.get()
        
        if current == goal:
            break
        
        for next in graph.neighbors(current):
            new_cost = cost_so_far[current] + graph.cost(current, next)
            if next not in cost_so_far or new_cost < cost_so_far[next]:
                cost_so_far[next] = new_cost
                priority = new_cost + heuristic2(goal, next)
                frontier.put(next, priority)
                came_from[next] = current

    
    return came_from, cost_so_far

def a_star_search2(graph, start, goal):
    frontier = PriorityQueue()
    frontier.put(start, 0)
    came_from = {}
    cost_so_far = {}
    came_from[start] = None
    cost_so_far[start] = 0
    
    while not frontier.empty():
        current = frontier.get()
        
        if current == goal:
            break
        
        for next in graph.neighbors(current):
            new_cost = cost_so_far[current] + graph.cost(current, next)
            if next not in cost_so_far or new_cost < cost_so_far[next]:
                cost_so_far[next] = new_cost
                priority = new_cost + heuristic1(goal, next)
                frontier.put(next, priority)
                came_from[next] = current

    
    return came_from, cost_so_far
### DRAW MAZE
maze = MazeWithWeights(10, 10)
maze.walls = [(1, 1), (2, 1), (3, 1), (4, 1), (5, 1), (6, 1), (7, 1), (8, 1), (8, 2), (8, 3), (8, 4), (8, 5), (8, 6), (8, 7), (8, 8),
                 (0, 3),(1, 3),(1, 4),(1, 5),(1, 6),(1, 8),(2, 8),(3, 8),(4, 8),(3, 9),(3,3),(5,2),(7,3),(3,4),(4,4),(5,4),(4,5),(4,6),(2,6),(6,5),(6,6),(6,7),(6,8),(6,9)]

# APPLY A* ALGO
s=(0, 9) #start point
g=(9, 0) #goal point
draw_grid(maze, start=s, goal=g)
came_from, cost_so_far = a_star_search1(maze, s, g)
draw_grid(maze, width=3, point_to=came_from, start=s, goal=g)
print('')
draw_grid(maze, width=3, number=cost_so_far, start=s, goal=g)
print('')
draw_grid(maze, width=3, path=reconstruct_path(came_from, start=s, goal=g))

s=(0, 9) #start point
g=(9, 0) #goal point
draw_grid(maze, start=s, goal=g)
came_from, cost_so_far = a_star_search2(maze, s, g)
draw_grid(maze, width=3, point_to=came_from, start=s, goal=g)
print('')
draw_grid(maze, width=3, number=cost_so_far, start=s, goal=g)
print('')
draw_grid(maze, width=3, path=reconstruct_path(came_from, start=s, goal=g))

s=(0, 9) #start point
g=(9, 0) #goal point
draw_grid(maze, start=s, goal=g)
came_from, cost_so_far = a_star_search2(maze, s, g)
draw_grid(maze, width=3, point_to=came_from, start=s, goal=g)
print ('')
print('A_STAR_SEARCH ADVANCED')
draw_grid(maze, width=3, number=cost_so_far, start=s, goal=g)
print('')
draw_grid(maze, width=3, path=reconstruct_path(came_from, start=s, goal=g))
