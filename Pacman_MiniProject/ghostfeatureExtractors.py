# featureExtractors.py
# --------------------
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


"Feature extractors for Ghost Agent game states"

from game import Directions, Actions
import util


def ghostDistance(pacman_pos, ghost_pos, walls):
    fringe = [(pacman_pos[0], pacman_pos[1], 0)]
    expanded = set()
    while fringe:
        pos_x, pos_y, dist = fringe.pop(0)
        if (pos_x, pos_y) in expanded:
            continue
        expanded.add((pos_x, pos_y))
        # if we find a capsule at this location then exit
        if (pos_x, pos_y) == (int(ghost_pos[0]), int(ghost_pos[1])):
            return dist
        # otherwise spread out from the location to its neighbours
        nbrs = Actions.getLegalNeighbors((pos_x, pos_y), walls)
        for nbr_x, nbr_y in nbrs:
            fringe.append((nbr_x, nbr_y, dist+1))
    # no capsule found
    return None


def pacmanDistance(pacman_pos, ghost_pos, walls):
    fringe = [(ghost_pos[0], ghost_pos[1], 0)]
    expanded = set()
    while fringe:
        pos_x, pos_y, dist = fringe.pop(0)
        if (pos_x, pos_y) in expanded:
            continue
        expanded.add((pos_x, pos_y))
        # if we find a capsule at this location then exit
        if (pos_x, pos_y) == (int(pacman_pos[0]), int(pacman_pos[1])):
            return dist
        # otherwise spread out from the location to its neighbours
        nbrs = Actions.getLegalNeighbors((pos_x, pos_y), walls)
        for nbr_x, nbr_y in nbrs:
            fringe.append((nbr_x, nbr_y, dist+1))
    # no capsule found
    return None


def closestCapsule(pos, capsules, walls):
    fringe = [(pos[0], pos[1], 0)]
    expanded = set()
    while fringe:
        pos_x, pos_y, dist = fringe.pop(0)
        if (pos_x, pos_y) in expanded:
            continue
        expanded.add((pos_x, pos_y))
        # if we find a capsule at this location then exit
        if (pos_x, pos_y) in capsules:
            return dist
        # otherwise spread out from the location to its neighbours
        nbrs = Actions.getLegalNeighbors((pos_x, pos_y), walls)
        for nbr_x, nbr_y in nbrs:
            fringe.append((nbr_x, nbr_y, dist+1))
    # no capsule found
    return None


class GhostFeatureExtractor:
    def getFeatures(self, state, action):
        """
          Returns a dict from features to counts
          Usually, the count will just be 1.0 for
          indicator functions.
        """
        util.raiseNotDefined()


class GhostIdentityExtractor(GhostFeatureExtractor):
    def getFeatures(self, state, action):
        feats = util.Counter()
        feats[(state, action)] = 1.0
        return feats


class GhostAdvancedExtractor(GhostFeatureExtractor):
    """
    Returns simple features for a basic reflex Pacman:
    - whether food will be eaten
    - how far away the next food is
    - whether a ghost collision is imminent
    - whether a ghost is one step away
    """

    def getFeatures(self, state, action):
        # extract the grid of food and wall locations and get the ghost locations
        ghostState = state.getGhostState(self.index)
        food = state.getFood()
        walls = state.getWalls()
        ghosts = state.getGhostPositions()
        pacman = state.getPacmanPosition()
        ghost = ghostState.getGhostPosition()
        ghost2 = ghostState.getGhostPosition()
        capsules = state.getCapsules()
        closestCapsule = closestCapsule(pacman, capsules, walls)

        # find other ghost position
        for g in ghosts:
            if g != ghost:
                ghost2 = g

        features = util.Counter()

        features["bias"] = 1.0

        # compute the location of ghost after he takes the action
        x, y = state.getGhostPosition()
        dx, dy = Actions.directionToVector(action)
        next_x, next_y = int(x + dx), int(y + dy)

        #Ghost is scared
        ghostState = state.getGhostState(self.index)
        isScared = ghostState.scaredTimer > 0
        if isScared:
            features["scared"] = float(pacmanDistance(
                pacman, (next_x, next_y), walls)) / (walls.width * walls.height)
            features["distancefrompacmanunscared"] = 0
            features["eats-pacman"] = 0
        else:
            features["eats-pacman"] = 1.0
            features["distancefrompacmanunscared"] = float(pacmanDistance(
                pacman, (next_x, next_y), walls)) / (walls.width * walls.height)
            features["scared"] = 0

        print("MY DICK IS SCARED")

        # count the number of ghosts 1-step away
        # features["#-of-ghosts-1-step-away"] = sum((next_x, next_y) in Actions.getLegalNeighbors(g, walls) for g in ghosts)

        # distance of ghost from each other
        #features["distancebetweenghosts"] = float(ghostDistance(ghost, ghost2, walls)) / (walls.width * walls.height)

        # Next distance from each other
        #features["distancebetweenghostsInNextAction"] = float(ghostDistance((next_x, next_y), ghost2, walls)) / (walls.width * walls.height)

        # distance of capsule from pacman in relation to distance of ghost from pacman
        #features["distanceofghostfrompacman"] = (float(pacmanDistance(pacman, ghost, walls)) / (walls.width * walls.height))
        #features["pactmanfromcapsule"] = (float(pacmanDistance(pacman, capsule, walls)) / (walls.width * walls.height))

        # if there is no ghosts nearyby go for pacman
        # if not features["#-of-ghosts-1-step-away"] and pacman[next_x][next_y]:
        #   features["eats-pacman"] = 1.0

        #dist = pacmanDistance(pacman, ghost, walls)
        # if dist is not None:
        # make the distance a number less than one otherwise the update
        # will diverge wildly
        #features["pacman"] = float(dist) / (walls.width * walls.height)
        features.divideAll(10.0)
        return features
