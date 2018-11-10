# qlearningGhostAgents.py
# ------------------
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


from game import *
from learningGhostAgents import ReinforcementGhostAgent
from ghostfeatureExtractors import *
import sys
import random
import util
import math
import pickle


class QLearningGhostAgent(ReinforcementGhostAgent):
    """
      Q-Learning Ghost Agent

      Functions you should fill in:
        - computeValueFromQValues
        - computeActionFromQValues
        - getQValue
        - getAction
        - update

      Instance variables you have access to
        - self.epsilon (exploration prob)
        - self.alpha (learning rate)
        - self.discount (discount rate)

      Functions you should use
        - self.getLegalActions(state)
          which returns legal actions for a state
    """

    def __init__(self, epsilon=0.05, gamma=0.8, alpha=0.2, numTraining=0, agentIndex=1, extractor='GhostIdentityExtractor', **args):
        "You can initialize Q-values here..."
        args['epsilon'] = epsilon
        args['gamma'] = gamma
        args['alpha'] = alpha
        args['numTraining'] = numTraining
        args['agentIndex'] = agentIndex
        self.index = agentIndex
        self.q_values = util.Counter()

        self.featExtractor = util.lookup(extractor, globals())()
        self.weights = util.Counter()
        ReinforcementGhostAgent.__init__(self, **args)

    def computeValueFromQValues(self, state):
        possibleActions = self.getLegalActions(state)
        if possibleActions:
            maxv = float("-inf")
            for action in possibleActions:
                q = self.getQValue(state, action)
                if q >= maxv:
                    maxv = q
            return maxv
        return 0.0

    def computeActionFromQValues(self, state):
        possibleActions = self.getLegalActions(state)
        if possibleActions:
            maxv = float("-inf")
            bestAction = None
            for action in possibleActions:
                q = self.getQValue(state, action)
                if q >= maxv:
                    maxv = q
                    bestAction = action
            return bestAction
        return None

    def getWeights(self):
        return self.weights

    # def getQValue(self, state, action):
    #    pass

    def getQValue(self, state, action):
        """
          Should return Q(state,action) = w * featureVector
          where * is the dotProduct operator
        """
        "*** YOUR CODE HERE ***"
        f = self.featExtractor.getFeatures(state, action)
        qv = 0
        for feature in f:
            qv = qv + self.weights[feature] * f[feature]
        return qv

    def update(self, state, action, nextState, reward):
        """
           Should update your weights based on transition
        """
        "*** YOUR CODE HERE ***"
        R = reward
        f = self.featExtractor.getFeatures(state, action)
        alphadiff = self.alpha * \
            ((R + self.discount * self.getValue(nextState)) -
             self.getQValue(state, action))
        for feature in f.keys():
            self.weights[feature] = self.weights[feature] + \
                alphadiff * f[feature]

    def final(self, state):
        "Called at the end of each game."
        # call the super-class final method
        ReinforcementGhostAgent.final(self, state)
        # did we finish training?
        if self.episodesSoFar == self.numTraining:
            # you might want to print your weights here for debugging
            "*** YOUR CODE HERE ***"
            # sys.exit(1)
            pass

    def getAction(self, state):
        # Uncomment the following if you want one of your agent to be a random agent.
        # if self.agentIndex == 1:
            # return random.choice(self.getLegalActions(state))
        possibleActions = self.getLegalActions(state)
        action = None
        "*** YOUR CODE HERE ***"
        if possibleActions:
            if util.flipCoin(self.epsilon) == True:
                action = random.choice(possibleActions)
            else:
                action = self.getPolicy(state)
        return action

    def getPolicy(self, state):
        return self.computeActionFromQValues(state)

    def getValue(self, state):
        return self.computeValueFromQValues(state)
