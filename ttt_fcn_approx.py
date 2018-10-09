import numpy as np
import scipy.linalg as la

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

# Calculate features for board
def getfeatures(board):
      
    player = 1
    lines = np.matrix( ((0, 1, 2),(3, 4, 5),(6, 7, 8),(0, 3, 6),(1, 4, 7),(2, 5, 8),(0, 4, 8),(2, 4, 6)) )
    playersinglets = (1==np.sum(board[lines] == player,axis = 1)) & (2==np.sum(board[lines] == 0,axis = 1))
    otherplayersinglets = (1==np.sum(board[lines] == getotherplayer(player),axis = 1)) & (2==np.sum(board[lines] == 0,axis = 1))
    playerdoublets = (2==np.sum(board[lines] == player,axis = 1)) & (1==np.sum(board[lines] == 0,axis = 1))
    otherplayerdoublets = (2==np.sum(board[lines] == getotherplayer(player),axis = 1)) & (1==np.sum(board[lines] == 0,axis = 1))
    playertriplets = (3==np.sum(board[lines] == player,axis = 1))
    otherplayertriplets = (3==np.sum(board[lines] == getotherplayer(player),axis = 1))
 
    # cross-points:
    playercrosspoint = 0
    otherplayercrosspoint = 0
    for location in range(9):
        I = np.any(lines==location, axis=1) # find line containing this location
        playercrosspoint = playercrosspoint + (np.sum(playersinglets[np.where(I)[0]]) > 1)
        otherplayercrosspoint = otherplayercrosspoint + np.sum(otherplayersinglets[np.where(I)[0]]) > 1
    
    theboard = board
    theboard[theboard == 2] = -1
    #features = np.array((bias, sum(playersinglets), sum(playerdoublets), sum(otherplayersinglets), sum(otherplayerdoublets), sum(playertriplets), sum(otherplayertriplets), playercrosspoint, otherplayercrosspoint))
    features_player = np.array((sum(playersinglets), sum(playerdoublets), sum(playertriplets), playercrosspoint))
    features_otherplayer = np.array((sum(otherplayersinglets), sum(otherplayerdoublets), sum(otherplayertriplets), otherplayercrosspoint))

    features = np.hstack((features_player, features_otherplayer, theboard))
    features = np.hstack((theboard))
# the features are linearly dependent for some reason, so removing the last position on the board somehow fixes this
# I have not yet got my head around this
    features = features[:-1]
    return features


def sigmoid (a):
    return(np.exp(a) / (1.0+np.exp(a)))

def dsigmoid (a):
    return(sigmoid(a)*(1-sigmoid(a)))


# this is the improving policy used by player
def epsilonfunctiongreedy(board, player, epsilon, weights, debug = False):
    moves = legal_moves(board)
    if (np.random.uniform() < epsilon):
        if (1 == debug):
            print("explorative move")
        return np.random.choice(moves, 1)
    na = np.size(moves)
    va = np.zeros(na)
    for i in range(0, na):
        board[moves[i]] = player
        phi = getfeatures(board)
        va[i] = sigmoid(np.matmul(phi, weights[1:]) + weights[0])
        board[moves[i]] = 0  # undo move
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
            ### Added to code to keep track of all states visited
            if (player == 1) & (state_visited_by_player1[hashit(board)] == 0) :
                BOARD[hashit(board),] = np.copy(board)
                state_visited_by_player1[hashit(board)] = 1
            ### end Added code
            if debug: # print the board, when in debug mode
                symbols = np.array([" ", "X", "O"])
                print("player ", symbols[player], ", move number ", move+1, ":")
                print(symbols[board.astype(int)].reshape(3,3))
            if (1 == iswin(board, player)): # has this player won?
                value[sold[player]] = value[sold[player]] + alpha * (1.0 - value[sold[player]])
                sold[player] = hashit(board) # index to winning state
                value[sold[player]] = 1.0 # winner (reward one)
                value[sold[getotherplayer(player)]] = 0.0 # looser (reward zero)
            # do a temporal difference update, once both players have made at least one move
            if (1 < move):
                value[sold[player]] = value[sold[player]] + alpha * (value[hashit(board)] - value[sold[player]])
            sold[player] = hashit(board) # store this new state for player
            # check if we have a draw, then set the final states for both players to 0.5
            if (8 == move):
                value[sold] = 0.5 # draw (equal reward for both)
            player = getotherplayer(player) # swap players

# the modified code from ttt.py where player one learns using a function approximator
# here we swithch of the learning for player 2 and see if player 1 can "fix" its function
def learnit_fcn_approx(numgames, epsilon, alpha, weights, debug = False):
    # play games for training
    for games in range(0, numgames):
        board = np.zeros(9)          # initialize the board
        oldboard = board
        # player to start is "1" the other player is "2"
        player = 1
        # start turn playing game, maximum 9 moves
        for move in range(0, 9):
            # use a policy to find action
            action = epsilongreedy(np.copy(board), player, epsilon, debug)
            # perform move and update board (for other player)
            board[action] = player         
            if (1 == iswin(board, player)): # has this player won?
                if (player == 1):
                    delta = (sigmoid(np.matmul(phiold, weights[1:]) + weights[0]) - 1)
                    weights[1:] = weights[1:] - alpha * delta * dsigmoid(np.matmul(phiold, weights[1:]) + weights[0]) * phi
                    weights[0] = weights[0] - alpha * delta * dsigmoid(np.matmul(phiold, weights[1:]) + weights[0]) * weights[0]
                if (player == 2):
                    delta = (sigmoid(np.matmul(phiold, weights[1:]) + weights[0]) - 0)
                    weights[1:] = weights[1:] - alpha * delta * dsigmoid(np.matmul(phiold, weights[1:]) + weights[0]) * weights[1:]
                    weights[0] = weights[0] - alpha * delta * dsigmoid(np.matmul(phiold, weights[1:]) + weights[0]) * 1

                break
            # do a temporal difference update, once both players have made at least one move
            if (1 < move) & (player == 1):
                #value[sold[player]] = value[sold[player]] + alpha * (value[hashit(board)] - value[sold[player]])
                phi = getfeatures(board)
                delta = (sigmoid(np.matmul(phiold, weights[1:]) + weights[0]) - sigmoid(np.matmul(phi, weights[1:]) + weights[0]))
                weights[1:] = weights[1:] - alpha * delta * dsigmoid(np.matmul(phiold, weights[1:]) + weights[0])*phiold
                weights[0] = weights[0] - alpha * delta * dsigmoid(np.matmul(phiold, weights[1:]) + weights[0])*1
            if (player == 1):
                phiold = getfeatures(board)
           # check if we have a draw, then set the final states for both players to 0.5
            if (8 == move):
                delta = (sigmoid(np.matmul(phiold, weights[1:]) + weights[0]) - 0.5)
                weights[1:] = weights[1:] - alpha * delta * dsigmoid(np.matmul(phiold, weights[1:]) + weights[0])*phiold
                weights[0] = weights[0] - alpha * delta * dsigmoid(np.matmul(phiold, weights[1:]) + weights[0])*1

            player = getotherplayer(player) # swap players
 

# one competition game, n is the number of rollouts for player 2
# epsilon is the exploration of player 1 (may want to switch this off)
def competition(weights, epsilon = 0.0, debug = False):
    board = np.zeros(9)          # initialize the board
    # player to start is "1" the other player is "2"
    player = 1
    # start turn playing game, maximum 9 moves
    for move in range(0, 9):
        # use a policy to find action, switch off exploration
        if (2 == player):
            action = epsilongreedy(np.copy(board), player, epsilon)
        else:
            action = epsilonfunctiongreedy(np.copy(board), player, epsilon, weights)
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
        if debug: # print the board, when in debug mode
            symbols = np.array([" ", "X", "O"])
            print("player ", symbols[player], ", move number ", move+1, ":")
            print(symbols[board.astype(int)].reshape(3,3))
 
    return reward

# global after-state value function, note this table is too big and contrains states that
# will never be used, also each state is unique to the player (no one after-state seen by both players) 
value = np.zeros(hashit(2 * np.ones(9)))
state_visited_by_player1 = np.zeros(hashit(2 * np.ones(9)))
BOARD = np.zeros( (hashit(2 * np.ones(9)), 9) )

alpha = 0.1 # step size
epsilon = 0.1 # exploration parameter

training_steps = 10000
competition_games = 100

# train the value function using a table representation
learnit(10*training_steps, epsilon, alpha)

# approximate the table using the features of the board states visited by player 1
I = np.where(state_visited_by_player1 == 1)[0]
dummy = getfeatures(np.zeros(9))
N = np.size(dummy) + 1
M = np.size(I)
PHI = np.zeros( (M,N))
V = np.zeros((M))
for i in range(M):
    PHI[i,] = np.concatenate( (np.ones(1), getfeatures(np.copy(BOARD[I[i],]))) )
    V[i] = value[I[i]]
# check the rank of this matrix (I had some problems with it)
np.linalg.matrix_rank(np.matmul(PHI.T,PHI))

weights = la.solve(PHI.T @ PHI, PHI.T @ V)

# remove the bias above:
PHI = PHI[:,1:]

# how about using R from Python?
import rpy2.robjects as robjects
import rpy2.robjects.packages as rpackages
from rpy2.robjects.vectors import StrVector
from rpy2.robjects import pandas2ri
import pandas as pd

N = N - 1 # we have removed the bias
cols = ['Y'] + ['X{number}'.format(number=num) for num in range(1,N+1)]
VandPHI = np.concatenate( (V[:,None], PHI), axis = 1)
data_pd_df = pd.DataFrame(VandPHI ,columns = cols)
pandas2ri.activate()
data_r_df = pandas2ri.py2ri(data_pd_df)

stats = rpackages.importr('stats')
base = rpackages.importr('base')
mass=rpackages.importr('MASS')


model = stats.lm('Y ~ .', data=data_r_df)
print(base.summary(model).rx2('coefficients'))
weights = np.array(stats.coefficients(model))

#weights = base.summary(model).rx2('coefficients')[0:N]
Vhat = np.matmul(np.concatenate((np.ones((M,1)),PHI), axis = 1),weights)
mse=((V-Vhat.T)**2).mean()

glmodel = stats.glm('Y ~ .', data=data_r_df, family = "binomial")
print(base.summary(glmodel).rx2('coefficients'))
aic=mass.stepAIC(glmodel, trace = False)
print(aic.rx2('anova'))

#lmweights = base.summary(glmodel).rx2('coefficients')[0:N]
lmweights = np.array(stats.coefficients(glmodel))

Vprobs = stats.predict(glmodel, data = data_r_df, type = "response")
mse_glm = ((V-Vprobs)**2).mean()

Vlmhat = np.matmul(np.concatenate((np.ones((M,1)),PHI), axis = 1),lmweights)
Vlmhat = np.exp(Vlmhat)/(np.exp(Vlmhat)+1.0)
mselm = ((V-Vlmhat.T)**2).mean()

wins_for_player_1 = 0
draw_for_players = 0
loss_for_player_1 = 0

for j in range(competition_games):
    reward = competition(lmweights, epsilon, debug = False) # switch off exploration
    if (reward == 1):
        wins_for_player_1 += 1.0
    elif (reward == 0.5):
        draw_for_players += 1.0
    else:
        loss_for_player_1 += 1.0
             
print(wins_for_player_1, draw_for_players, loss_for_player_1)


learnit_fcn_approx(training_steps, epsilon, alpha, lmweights)

wins_for_player_1 = 0
draw_for_players = 0
loss_for_player_1 = 0

for j in range(competition_games):
    reward = competition(weights, epsilon, debug = False) # switch off exploration
    if (reward == 1):
        wins_for_player_1 += 1.0
    elif (reward == 0.5):
        draw_for_players += 1.0
    else:
        loss_for_player_1 += 1.0

print(wins_for_player_1, draw_for_players, loss_for_player_1)
    
weights = 0*weights
learnit_fcn_approx(training_steps, epsilon, alpha, weights)

wins_for_player_1 = 0
draw_for_players = 0
loss_for_player_1 = 0

for j in range(competition_games):
    reward = competition(lmweights, epsilon, debug = False) # switch off exploration
    if (reward == 1):
        wins_for_player_1 += 1.0
    elif (reward == 0.5):
        draw_for_players += 1.0
    else:
        loss_for_player_1 += 1.0

print(wins_for_player_1, draw_for_players, loss_for_player_1)
    
