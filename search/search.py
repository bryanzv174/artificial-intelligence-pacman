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
# Import the stack from util
from util import Stack
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


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]


def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first (DFS).
    """
    # We use a stack (LIFO) for the DFS algorithm
    fringe = Stack()

    # We initialize the stack with the initial state and an empty list of actions
    fringe.push((problem.getStartState(), []))

    # An array to store visited states
    visited = set()

    # As long as the stack is not empty, we continue exploring
    while not fringe.isEmpty():
        # We pop the node (current state, actions so far) from the stack
        state, actions = fringe.pop()

        # If the current state is the target, we return the list of actions
        if problem.isGoalState(state):
            return actions

        # If the state has not been visited
        if state not in visited:
            # We mark the state as visited
            visited.add(state)

            # We expand the successors of the current state
            for successor, action, stepCost in problem.getSuccessors(state):
                if successor not in visited:
                    # We add the successor to the stack with the updated actions
                    fringe.push((successor, actions + [action]))

    # If no solution is found, we return an empty list
    return []

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


from util import Queue


from util import Queue

def breadthFirstSearch(problem):
    """
    Search the shallowest nodes in the search tree first (BFS).
    """
    # We create a queue (FIFO) to store the nodes to explore
    fringe = Queue()

    # We initialize the queue with the initial state and an empty list of actions.
    start_state = problem.getStartState()
    fringe.push((start_state, []))

    # An array to store visited states
    visited = set()

    # Since the queue is not empty, continue exploring.
    while not fringe.isEmpty():

        # We remove the node (current state, actions so far) from the queue

        state, actions = fringe.pop()

        # If the current status is the target, we return the action list.
        if problem.isGoalState(state):
            return actions

        # If the status has not visited
        if state not in visited:

            # Mark the status as visited.
            visited.add(state)

            # Expand the successors of the current status
            for successor, action, stepCost in problem.getSuccessors(state):
                if successor not in visited:

                    # add the successor to the queue with the update actions
                    fringe.push((successor, actions + [action]))

    # If don't find solution, return a empty list
    return []


from util import PriorityQueue


def uniformCostSearch(problem):
    """Search the node of least total cost first (UCS)."""
    # We use a priority queue for the UCS algorithm
    fringe = PriorityQueue()

    # Initialize the priority queue with the initial state and a cost of 0
    fringe.push((problem.getStartState(), []), 0)

    # A set to store visited states
    visited = set()

    # While the priority queue is not empty, we keep exploring
    while not fringe.isEmpty():
        # We pop the node with the lowest accumulated cost from the priority queue
        state, actions = fringe.pop()

        # If the current state is the goal, return the list of actions
        if problem.isGoalState(state):
            return actions

        # If the state has not been visited
        if state not in visited:
            # Mark the state as visited
            visited.add(state)

            # Expand the successors of the current state
            for successor, action, stepCost in problem.getSuccessors(state):
                if successor not in visited:
                    # Calculate the new accumulated cost
                    new_cost = problem.getCostOfActions(actions + [action])
                    # Add the successor to the priority queue with the updated accumulated cost
                    fringe.push((successor, actions + [action]), new_cost)

    # If no solution is found, return an empty list
    return []


def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0


from util import PriorityQueue


def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first (A*)."""
    # We use a priority queue for the A* algorithm
    fringe = PriorityQueue()

    # Initialize the queue with the initial state, accumulated cost (g), and the heuristic value (h)
    start_state = problem.getStartState()
    fringe.push((start_state, [], 0), heuristic(start_state, problem))

    # A set to store visited states
    visited = set()

    # While the priority queue is not empty, we keep exploring
    while not fringe.isEmpty():
        # We pop the node with the lowest cost g + h
        state, actions, cost = fringe.pop()

        # If the current state is the goal, return the list of actions
        if problem.isGoalState(state):
            return actions

        # If the state has not been visited
        if state not in visited:
            # Mark the state as visited
            visited.add(state)

            # Expand the successors of the current state
            for successor, action, stepCost in problem.getSuccessors(state):
                if successor not in visited:
                    # Calculate the new accumulated cost
                    new_cost = cost + stepCost
                    # Calculate f(n) = g(n) + h(n)
                    priority = new_cost + heuristic(successor, problem)
                    # Add the successor to the priority queue with the new priority
                    fringe.push((successor, actions + [action], new_cost), priority)

    # If no solution is found, return an empty list
    return []


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
