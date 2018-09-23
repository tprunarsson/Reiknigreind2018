import pandas as pd
import numpy as np
import ggplot as gg

# this function is used to find an index to our value table
def hashit(board):
    base3 = np.matmul(np.power(3, range(0, 9)), board.transpose())
    return int(base3)

# this function finds all legal actions (moves) for given state A(s)
def legal_moves(board):
    return np.where(board == 0)[0]

# this is the improving policy used by player
def epsilongreedy(board, player, epsilon, debug = False):
    moves = legal_moves(board)
    if (np.random.uniform() < epsilon):
        if (1 == debug):
            print("explorative move")
        return np.random.choice(moves, 1)
    na = np.size(moves)
    va = np.zeros(na)
    for i in range(0, na):
        board[moves[i]] = player
        va[i] = value[hashit(board)]
        board[moves[i]] = 0  # undo
    return moves[np.argmax(va)]

# play random games to the end starting from the current board
def rollout(board, theplayer):
    player = getotherplayer(theplayer) # its the other player's turn
    for move in range(np.sum(board>0), 9):
        # use a random policy to find action, epsilon = 1.0
        action = epsilongreedy(np.copy(board), player, 1.0)
        board[action] = player # perform the move       
        if (1 == iswin(board, player)): # has this player won?
            if (player == theplayer): # this will be player 2
                reward = 1.0
            else:
                reward = 0.0
            break
        # check if we have a draw
        if (8 == move):
           reward = 0.5 # draw (equal reward for both)
        player = getotherplayer(player) # swap players
    return reward

# the move selected by player 2 using rollouts
def mcgreedy(board, player, n):
    moves = legal_moves(board)
    na = np.size(moves)
    va = np.zeros(na)
    for i in range(na):
        board[moves[i]] = player
 # The code in the lines below taken out are for an idea discussed in class, on
 # on keeping the rollout values for computational speedup.        
 #       if (value[hashit(board)] > 0): # make sure the function value is set to zero initially
 #           va[i] = value[hashit(board)] # we are using previous rollouts from this state
 #       else:
        for j in range(n):
            va[i] += rollout(np.copy(board), player)
 #       value[hashit(board)] = va[i] # keep this for future use
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

def learnitMC(numgames, epsilon, alpha, n, debug = False):
    # play games for training
    for games in range(0, numgames):
        board = np.zeros(9)          # initialize the board
        sold = [0, hashit(board), 0] # first element not used
        # player to start is "1" the other player is "2"
        player = 1
        # start turn playing game, maximum 9 moves
        for move in range(0, 9):
            # use a policy to find action
            if (1 == player):
                action = epsilongreedy(np.copy(board), player, epsilon, debug)
            else:
                action = mcgreedy(np.copy(board), player, n)
            # perform move and update board (for other player)
            board[action] = player            
            if debug: # print the board, when in debug mode
                symbols = np.array([" ", "X", "O"])
                print("player ", symbols[player], ", move number ", move+1, ":")
                print(symbols[board.astype(int)].reshape(3,3))
            if (1 == iswin(board, player)): # has this player won?
                if (player == 1):
                    value[sold[player]] = value[sold[player]] + alpha * (1.0 - value[sold[player]])
                    sold[player] = hashit(board) # index to winning state
                    value[sold[player]] = 1.0 # winner (reward one)
                else:
                    value[sold[getotherplayer(player)]] = 0.0 # looser (reward zero)
                break
            # do a temporal difference update, once both players have made at least one move
            if ((1 < move) & (player == 1)): # only update for player 1
                value[sold[player]] = value[sold[player]] + alpha * (value[hashit(board)] - value[sold[player]])
            sold[player] = hashit(board) # store this new state for player
            # check if we have a draw, then set the final states for both players to 0.5
            if (8 == move): # this is OK to update for both players
                value[sold] = 0.5 # draw (equal reward for both)
            player = getotherplayer(player) # swap players

# the unmodified code from ttt.py
def learnit(numgames, epsilon, alpha, debug = False):
    # play games for training
    for games in range(0, numgames):
        board = np.zeros(9)          # initialize the board
        sold = [0, hashit(board), 0] # first element not used
        # player to start is "1" the other player is "2"
        player = 1
        # start turn playing game, maximum 9 moves
        for move in range(0, 9):
            # use a policy to find action
            action = epsilongreedy(np.copy(board), player, epsilon, debug)
            # perform move and update board (for other player)
            board[action] = player            
            if debug: # print the board, when in debug mode
                symbols = np.array([" ", "X", "O"])
                print("player ", symbols[player], ", move number ", move+1, ":")
                print(symbols[board.astype(int)].reshape(3,3))
            if (1 == iswin(board, player)): # has this player won?
                value[sold[player]] = value[sold[player]] + alpha * (1.0 - value[sold[player]])
                sold[player] = hashit(board) # index to winning state
                value[sold[player]] = 1.0 # winner (reward one)
                value[sold[getotherplayer(player)]] = 0.0 # looser (reward zero)
                break
            # do a temporal difference update, once both players have made at least one move
            if (1 < move):
                value[sold[player]] = value[sold[player]] + alpha * (value[hashit(board)] - value[sold[player]])
            sold[player] = hashit(board) # store this new state for player
            # check if we have a draw, then set the final states for both players to 0.5
            if (8 == move):
                value[sold] = 0.5 # draw (equal reward for both)
            player = getotherplayer(player) # swap players

# one competition game, n is the number of rollouts for player 2
# epsilon is the exploration of player 1 (may want to switch this off)
def competition(n, epsilon):
    board = np.zeros(9)          # initialize the board
    # player to start is "1" the other player is "2"
    player = 1
    # start turn playing game, maximum 9 moves
    for move in range(0, 9):
    # use a policy to find action, switch off exploration
        if (1 == player):
            action = epsilongreedy(np.copy(board), player, epsilon)
        else:
            action = mcgreedy(np.copy(board), player, n)
        # perform move and update board (for other player)
        board[action] = player            
        if (1 == iswin(board, player)): # has this player won?
            if (player == 1):
                reward = 1.0 # we only record the winnings for player 1
            else:
                reward = 0.0
            break
        # check if we have a draw, then set the final states for both players to 0.5
        if (8 == move):
            reward = 0.5 # draw (equal reward for both)
        player = getotherplayer(player) # swap players
    return reward

# global after-state value function, note this table is too big and contrains states that
# will never be used, also each state is unique to the player (no one after-state seen by both players) 
value = np.zeros(hashit(2 * np.ones(9)))
alpha = 0.1 # step size
epsilon = 0.1 # exploration parameter
# when 10 player 1 is winning more than drawing
# when 30 player 1 is drawing more than winning
n = 100 # number of rollouts, change this to n= 10, 30 or 100
# train the value function using 10 x 1000 games
epochs = 100
training_steps = 100
competition_games = 100
wins_for_player_1 = np.zeros(epochs)
draw_for_players = np.zeros(epochs)

data = []
for i in range(epochs):
    for j in range(competition_games):
         reward = competition(n, 0.0) # switch off exploration
         if (reward == 1):
             wins_for_player_1[i] += 1.0
         elif (reward == 0.5):
             draw_for_players[i] += 1.0
             
    print(i, wins_for_player_1[i], draw_for_players[i])
    data.append({'Type': 0, 'Wins': wins_for_player_1[i], 'Training': training_steps*(i-1)})
    data.append({'Type': 1, 'Wins': draw_for_players[i], 'Training': training_steps*(i-1)})
    learnitMC(training_steps, epsilon, alpha, n)
 #   learnit(training_steps, epsilon, alpha) # the original learning code.

# Pandas gives you the power of R
learningdf = pd.DataFrame(data)
# I use ggplot when I generate figures in R and would like to use it with Python, HOWEVER:
# latest Pandas causes problems for ggplot so I needed these two patches:
# https://stackoverflow.com/questions/50591982/importerror-cannot-import-name-timestamp/52378663
# https://github.com/yhat/ggpy/issues/612
p = gg.ggplot(gg.aes(x='Training', y='Wins', group='Type'), data=learningdf)+ gg.xlab('Learning games') + \
    gg.ylab('Wins for player 1') + gg.ggtitle("n="+str(n)) + gg.geom_point() + gg.stat_smooth(method='loess')
p.make()
filename = "experiment_"+str(n)+".pdf"
p.save(filename)
