# search.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util
from game import Directions
from typing import List

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()




def tinyMazeSearch(problem: SearchProblem) -> List[Directions]:
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]


def depthFirstSearch(problem: SearchProblem) -> List[Directions]:
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    """
    "*** YOUR CODE HERE ***"
    # print("Start:", problem.getStartState())
    # print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    # print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    startNode = problem.getStartState() # Get the starting state of the problem
    # print(startNode)
    # If the start state is already the goal, return an empty action list
    if problem.isGoalState(startNode):
        return []

    # Initialize a stack for DFS, storing (node, actions) tuples
    stack= util.Stack()
    visited = []  # track visited nodes
    stack.push((startNode, [])) # Push the startnode and an empty action

    while not stack.isEmpty():
        currentNode, actions = stack.pop() # Pop the current node and the actions
        if currentNode not in visited:
            # Mark the current node as visited
            visited.append(currentNode)
            # if the current node is the goal state
            if problem.isGoalState(currentNode):
                return actions

            for nextNode, action, _ in problem.getSuccessors(currentNode):
                newAction = actions + [action]
                stack.push((nextNode, newAction))
    util.raiseNotDefined()


def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first using BFS."""

    
    # Initialize an empty Queue and an empty list for visited nodes
    frontier = util.Queue()
    visited = set()  # Use a set for faster lookups

    # Initialize the starting position and path
    start_state = problem.getStartState()
    frontier.push((start_state, [], 0))  # (state, path, cost)

    def add_to_frontier(state, path, cost):
        """Helper function to add a state to the frontier if not visited."""
        if state not in visited:
            frontier.push((state, path, cost))

    while not frontier.isEmpty():
        # Pop the front element from the queue
        current_state, path, action_cost = frontier.pop()

        # Check if we've reached the goal
        if problem.isGoalState(current_state):
            return path

        # If the state hasn't been visited, process it
        if current_state not in visited:
            visited.add(current_state)  # Mark the state as visited

            # Explore successors
            for successor, action, step_cost in problem.getSuccessors(current_state):
                new_path = path + [action]
                add_to_frontier(successor, new_path, action_cost + step_cost)

    # Raise an error if no solution is found
    util.raiseNotDefined()




def uniformCostSearch(problem: SearchProblem) -> List[Directions]:
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"

    from util import PriorityQueue

    pq = PriorityQueue()
    visited = {}
 
    startPosition = problem.getStartState()
    startStepCost = 0


    pq.push((startPosition, []), startStepCost)  
    visited[startPosition] = 0;

    while not pq.isEmpty():

        currentNode = pq.pop()
        position = currentNode[0]
        path = currentNode[1]

        if problem.isGoalState(position):
            return path;

        successors = problem.getSuccessors(position)

        for successor, action, StepCost in successors:

            newPath = path + [action]
            g = problem.getCostOfActions(newPath)

            if successor not in visited or g < visited[successor]:
                visited[successor] = g
                pq.push((successor, newPath), g)


    util.raiseNotDefined()


def nullHeuristic(state, problem=None) -> float:
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem: SearchProblem, heuristic=nullHeuristic) -> List[Directions]:
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"

    from util import PriorityQueue

    pq = PriorityQueue()

    visited = {}
    startPosition = problem.getStartState()
    startPriority = heuristic(startPosition, problem)

    pq.push((startPosition, []), startPriority) 
    visited[startPosition] = 0;

    while not pq.isEmpty():

        currentNode = pq.pop()
        position = currentNode[0]
        path = currentNode[1]

        if problem.isGoalState(position):
            return path;

        successors = problem.getSuccessors(position)

        for successor, action, stepCost in successors:
                 
            newPath = path + [action]
            newPosition = successor

            h = heuristic(newPosition, problem)
            g = problem.getCostOfActions(newPath)
            f = g + h

            if successor not in visited or g< visited[successor]:
                visited[successor] = g     
                pq.push((successor,newPath), f)

    util.raiseNotDefined()

# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
