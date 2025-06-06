# multiAgents.py
# --------------
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



from util import manhattanDistance
from game import Directions
import random, util
from game import Agent

class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood().asList()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        # Start with the game score
        score = successorGameState.getScore()

        # Calculate distance to the nearest food
        foodDistances = [manhattanDistance(newPos, food) for food in newFood]
        if foodDistances:
            score += 10.0 / min(foodDistances)  # Favor closer food

        # Consider ghosts: try to avoid getting too close
        for ghostState, scaredTime in zip(newGhostStates, newScaredTimes):
            ghostPos = ghostState.getPosition()
            ghostDist = manhattanDistance(newPos, ghostPos)
            if scaredTime > 0:
                # Scared ghosts are good to approach
                score += 200.0 / ghostDist
            else:
                # Active ghosts should be avoided
                if ghostDist < 2:  # Penalize being too close to a ghost
                    score -= 1000

        # Favor states with less food left
        score -= 4 * len(newFood)

        return score


def scoreEvaluationFunction(currentGameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
      This class provides some common elements to all of your
      multi-agent searchers.  Any methods defined here will be available
      to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

      You *do not* need to make any changes here, but you can if you want to
      add functionality to all your adversarial search agents.  Please do not
      remove anything, however.

      Note: this is an abstract class: one that should not be instantiated.  It's
      only partially specified, and designed to be extended.  Agent (game.py)
      is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)


class MinimaxAgent(MultiAgentSearchAgent):
    def getAction(self, gameState):


        def minimax(agentIndex, depth, gameState):
            # If the state is terminal (win/lose) or max depth is reached, return evaluation
            if gameState.isWin() or gameState.isLose() or depth == self.depth:
                return self.evaluationFunction(gameState)

            # Pac-Man's turn (Maximizing player)
            if agentIndex == 0:
                return maxValue(agentIndex, depth, gameState)

            # Ghosts' turn (Minimizing players)
            else:
                return minValue(agentIndex, depth, gameState)

        def maxValue(agentIndex, depth, gameState):
            # Initialize the best value to negative infinity
            bestValue = float('-inf')
            bestAction = None

            # Get all legal actions for Pac-Man
            legalActions = gameState.getLegalActions(agentIndex)

            for action in legalActions:
                # Generate the successor state for each action
                successor = gameState.generateSuccessor(agentIndex, action)

                # Recursively call minimax for the next agent
                value = minimax(1, depth, successor)

                # Update the best value and action if necessary
                if value > bestValue:
                    bestValue = value
                    bestAction = action

            if depth == 0:  # Only return the action at the root
                return bestAction
            return bestValue

        def minValue(agentIndex, depth, gameState):
            # Initialize the best value to positive infinity
            bestValue = float('inf')

            # Get all legal actions for the ghost
            legalActions = gameState.getLegalActions(agentIndex)

            for action in legalActions:
                # Generate the successor state for each action
                successor = gameState.generateSuccessor(agentIndex, action)

                # Determine the next agent (either next ghost or Pac-Man at the next depth)
                nextAgent = (agentIndex + 1) % gameState.getNumAgents()
                if nextAgent == 0:  # If next agent is Pac-Man, increment depth
                    value = minimax(nextAgent, depth + 1, successor)
                else:
                    value = minimax(nextAgent, depth, successor)

                # Update the best value if necessary
                if value < bestValue:
                    bestValue = value

            return bestValue

        # Call minimax from the root (Pac-Man's turn)
        return minimax(0, 0, gameState)


class AlphaBetaAgent(MultiAgentSearchAgent):
    def getAction(self, gameState):
        def alphaBeta(agentIndex, depth, gameState, alpha, beta):
            if gameState.isWin() or gameState.isLose() or depth == self.depth:
                return self.evaluationFunction(gameState)

            if agentIndex == 0:  # Pacman's turn (maximizing)
                return maxValue(agentIndex, depth, gameState, alpha, beta)
            else:  # Ghosts' turn (minimizing)
                return minValue(agentIndex, depth, gameState, alpha, beta)

        def maxValue(agentIndex, depth, gameState, alpha, beta):
            v = float('-inf')
            bestAction = None
            for action in gameState.getLegalActions(agentIndex):
                successor = gameState.generateSuccessor(agentIndex, action)
                value = alphaBeta(1, depth, successor, alpha, beta)
                if value > v:
                    v = value
                    bestAction = action
                if v > beta:  # Prune only if strictly greater
                    return v
                alpha = max(alpha, v)
            return bestAction if depth == 0 else v

        def minValue(agentIndex, depth, gameState, alpha, beta):
            v = float('inf')
            nextAgent = (agentIndex + 1) % gameState.getNumAgents()
            nextDepth = depth + 1 if nextAgent == 0 else depth
            for action in gameState.getLegalActions(agentIndex):
                successor = gameState.generateSuccessor(agentIndex, action)
                value = alphaBeta(nextAgent, nextDepth, successor, alpha, beta)
                v = min(v, value)
                if v < alpha:  # Prune only if strictly less
                    return v
                beta = min(beta, v)
            return v

        # Start alpha-beta from Pacman (agent index 0) at the root
        return alphaBeta(0, 0, gameState, float('-inf'), float('inf'))

class ExpectimaxAgent(MultiAgentSearchAgent):
    def getAction(self, gameState):
        def expectimax(agentIndex, depth, gameState):
            # Si el estado es terminal (ganar/perder) o alcanzamos la profundidad máxima
            if gameState.isWin() or gameState.isLose() or depth == self.depth:
                return self.evaluationFunction(gameState)

            # Turno de Pacman (maximizar)
            if agentIndex == 0:
                return maxValue(agentIndex, depth, gameState)

            # Turno de los fantasmas (expectativa)
            else:
                return expValue(agentIndex, depth, gameState)

        def maxValue(agentIndex, depth, gameState):
            v = float('-inf')
            bestAction = None
            for action in gameState.getLegalActions(agentIndex):
                successor = gameState.generateSuccessor(agentIndex, action)
                value = expectimax(1, depth, successor)
                if value > v:
                    v = value
                    bestAction = action
            if depth == 0:  # Solo devolver la acción en la raíz
                return bestAction
            return v

        def expValue(agentIndex, depth, gameState):
            v = 0
            legalActions = gameState.getLegalActions(agentIndex)
            prob = 1.0 / len(legalActions)  # Probabilidad uniforme para cada acción

            nextAgent = (agentIndex + 1) % gameState.getNumAgents()
            nextDepth = depth + 1 if nextAgent == 0 else depth

            for action in legalActions:
                successor = gameState.generateSuccessor(agentIndex, action)
                value = expectimax(nextAgent, nextDepth, successor)
                v += prob * value  # Sumar el valor esperado
            return v

        # Iniciar expectimax desde Pacman (agente 0)
        return expectimax(0, 0, gameState)


def betterEvaluationFunction(currentGameState):
    pos = currentGameState.getPacmanPosition()
    food = currentGameState.getFood().asList()
    ghosts = currentGameState.getGhostStates()
    capsules = currentGameState.getCapsules()

    # Empezamos con el puntaje del estado actual
    score = currentGameState.getScore()

    # Característica 1: Incentivo por acercarse a la comida más cercana
    foodDistances = [manhattanDistance(pos, foodPos) for foodPos in food]
    if foodDistances:
        score += 10.0 / min(foodDistances)  # Incentivar la comida cercana

    # Característica 2: Penalización por la cantidad de comida restante
    score -= 4 * len(food)  # Penalizar la cantidad de comida restante

    # Característica 3: Incentivo fuerte por cápsulas de poder cercanas
    capsuleDistances = [manhattanDistance(pos, cap) for cap in capsules]
    if capsuleDistances:
        score += 50.0 / min(capsuleDistances)  # Fuerte incentivo por cápsulas cercanas

    # Característica 4: Incentivo o penalización basado en la distancia a los fantasmas
    for ghost in ghosts:
        ghostDistance = manhattanDistance(pos, ghost.getPosition())
        if ghost.scaredTimer > 0:
            # Incentivo por acercarse a fantasmas asustados para ganar puntos
            score += 200 / ghostDistance if ghostDistance > 0 else 200
        else:
            # Si hay una cápsula cerca, reducimos la penalización para tomar el riesgo
            if capsuleDistances and min(capsuleDistances) < 3:
                score -= 200 / (ghostDistance + 1)  # Penalización reducida si hay una cápsula cerca
            else:
                # Penalización fuerte si los fantasmas están cerca y no hay cápsula en camino
                if ghostDistance < 2:
                    score -= 1000
                else:
                    score -= 2 / (ghostDistance + 1)

    return score

# Abbreviation
better = betterEvaluationFunction
