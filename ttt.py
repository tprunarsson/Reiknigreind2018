import numpy as np

def hashit(state):
    base3 = np.matmul(np.power(3, range(0, 9)), state.transpose())
    return int(base3)

def legal_moves(board):
    return np.where(board == 0)[0]

def epsilongreedy(board, player, epsilon):
    moves = legal_moves(board)
    if (np.random.uniform() < epsilon):
        return np.random.choice(moves, 1)
    na = np.size(moves)
    va = np.zeros(na)
    for i in range(0, na):
        board[moves[i]] = player
        va[i] = value[hashit(board)]
        board[moves[i]] = 0  # undo
    return moves[np.argmax(va)]

def iswin(board, m):
    if np.all(board[[0, 1, 2]] == m) | np.all(board[[3, 4, 5]] == m):
        return 1
    if np.all(board[[6, 7, 8]] == m) | np.all(board[[0, 3, 6]] == m):
        return 1
    if np.all(board[[1, 4, 7]] == m) | np.all(board[[2, 5, 8]] == m):
        return 1
    if np.all(board[[0, 4, 8]] == m) | np.all(board[[2, 4, 6]] == m):
        return 1
    return 0

def getotherplayer(player):
    if (player == 1):
        return 2
    return 1

def learnit(numgames, epsilon, alpha, debug=False):
    # play games for training
    for games in range(0, numgames):
        state = np.zeros(9)        # initialize the board
        s = [hashit(state), hashit(state)] # this state value is never used
        # player to start is "+1"
        player = 1
        # start turn playing game, maximum 9 moves
        for move in range(0, 9):
            # use policy to find action
            action = epsilongreedy(np.copy(state), player, epsilon)
            # perform action and observe new state (for other player)
            state[action] = player
            # print the board, if in debug mode
            if debug:
                print(state.reshape(3, 3))
            # has this player won?
            if (1 == iswin(state, player)):
                value[s[player - 1]] = value[s[player - 1]] + alpha * (1.0 - value[s[player - 1]])
                s[player - 1] = hashit(state) # index to winning state
                value[s[player - 1]] = 1.0 # winner (reward one)
                value[s[getotherplayer(player) - 1]] = 0.0 # looser, reward zero
                # also do a temporal update for the previous state
                break
            # do a temporal difference update
            value[s[player - 1]] = value[s[player - 1]] + alpha * (value[hashit(state)] - value[s[player - 1]])
            s[player - 1] = hashit(state) # store this new state for player
            player = getotherplayer(player) # swap players

# do trails here, note because we initialize everything to 0.5 we don't need
# to update for draws 
value = np.ones(hashit(2 * np.ones(9))) / 2.0
alpha = 0.1
epsilon = 0.1
# train the value function
learnit(10000, epsilon, alpha)
# play one game determinstically using the value function
learnit(1, 0, 0, True)
