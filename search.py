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

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """
    "*** YOUR CODE HERE ***"


    util.raiseNotDefined()


#  """
#       state: Search state

#       对于给定的状态，该函数应返回一个包含多个三元组的列表，每个三元组包含：
#       (successor, action, stepCost)
#       - successor：后继状态（下一个状态）
#       - action：从当前状态移动到后继状态所采取的动作
#       - stepCost：从当前状态移动到后继状态的代价
#  """
def breadthFirstSearch(problem: SearchProblem) -> List[Directions]:
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    from util import Queue

    #create an queue to store the state and path
    queue = Queue()
    # 存储已经访问过的状态
    visited = set()

    # 定义起始状态
    start_position = problem.getStartState()

    # 将起始状态压入队列,路径为空
    queue.push((start_position, [], 0))   # (位置, 路径, 代价)

    # 开始搜索
    while not queue.isEmpty():

        currentNode = queue.pop()
        position, path, stepCost = currentNode

        #如果当前状态是目标状态， 返回路径， 说明已经到达终点
        if problem.isGoalState(position):
            return path;
        #如果该状态没有访问过，标记之后，进行扩展
        if position not in visited:
            visited.add(position)   #标记该状态为已访问

            # 获取当前节点的所有后继节点
            successors = problem.getSuccessors(position)

            # 获取该状态的所有后继状态，压入队列. problem.getSuccessors(state) 返回的是一个列表，其中每一项是一个三元组
            for successor, action, stepCost in successors:
                if successor not in visited:
                    new_path = path + [action]
                    queue.push((successor, new_path, stepCost))

    util.raiseNotDefined()


def uniformCostSearch(problem: SearchProblem) -> List[Directions]:
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    

    util.raiseNotDefined()

def nullHeuristic(state, problem=None) -> float:
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0


# f = g + h, g:从起点到当前节点的实际路径代价, h:从当前节点到目标节点的曼哈顿距离
# 按照bfs的基本结构的话， 问题就是已经访问的节点无法再次访问， 有最优解但导致路径中断
def aStarSearch(problem: SearchProblem, heuristic=nullHeuristic) -> List[Directions]:
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"

    from util import PriorityQueue

    #create an priorityQueue to store the state and path
    pq = PriorityQueue()
    # 存储已经访问过的状态，字典存储，键值对：节点和节点的最短路径
    visited = {}
    # priority = 0 # 这个不知道为啥影响那么大

    # 定义起始状态
    startPosition = problem.getStartState()
    startPriority = heuristic(startPosition, problem)

    # 将起始状态压入队列,路径为空
    pq.push((startPosition, []), startPriority)  # (位置, 路径, 优先级)
    visited[startPosition] = 0;
    #print(pq.pop())  # 查看返回值的结构  ('A', [])

    # 开始搜索
    while not pq.isEmpty():

        currentNode = pq.pop()
        position = currentNode[0]
        path = currentNode[1]
        # priority, (position, path) = pq.pop()

        #如果当前状态是目标状态， 返回路径， 说明已经到达终点
        if problem.isGoalState(position):
            return path;
        #如果该状态没有访问过，标记之后，进行扩展
        # if position not in visited:
        #     visited.add(position)   #标记该状态为已访问

        # 获取当前节点的所有后继节点
        successors = problem.getSuccessors(position)

        # 要计算后继节点的cost，那么位置就得更新后再计算，得到cost即priority之后，入列
        for successor, action, stepCost in successors:
                 
            # move first
            newPath = path + [action]
            newPosition = successor

            h = heuristic(newPosition, problem)
            g = problem.getCostOfActions(newPath)
            f = g + h

            if successor not in visited or g< visited[successor]:
                visited[successor] = g     # # 更新访问记录中的最小代价
                pq.push((successor,newPath), f)

    util.raiseNotDefined()

# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
