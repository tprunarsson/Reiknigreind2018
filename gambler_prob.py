# Copyright (C)
# 2016-2018 Shangtong Zhang(zhangshangtong.cpp@gmail.com)
# 2016 Kenta Shimada(hyperkentakun@gmail.com)
# Permission given to modify the code as long as you keep this
# declaration at the top

# Modified significantly by tpr@hi.is

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

# goal, try using an odd number
GOAL = 100
STATES = np.arange(1,GOAL) # the states are 1,2,..,GOAL, last state is terminal

# probability of head
HEAD_PROB = 0.4

# how would you like to break ties ?!
BREAK_TIES = 1  # 1.) take tjhe first in lexiographical order 
                # 2.) take the last in lexiographical order
                # 3.) break ties randomly (this will show the family)

# initialize state value, all terminal states by definition have zero value
# the number of states is 1..(GOAL-1) plus two terminal states 0 and GOAL
state_value = np.zeros(GOAL-1+2) 

def policy_value(state, breakties):
    # get possible actions for current state, $1,$2,...,$state
    action = np.arange(1,min(state,GOAL-state) + 1)
    # initialize the memory for the array of returns
    action_returns = np.zeros(action.size)
    for i in range(0,action.size):
        if (state + action[i]) == GOAL: # the goal has been reached when we have $GOAL or more
            reward = 1.0 # you have reached a successful end state
        else:
            reward = 0.0 # you have not reached the goal yet
        # note here that the terminal states are 0 and GOAL and have value 0
        action_returns[i] = HEAD_PROB * (reward + state_value[state + action[i]]) + \
            (1.0 - HEAD_PROB) * state_value[state - action[i]]
    # the action with the highest return (greedy action has this value)
    max_value = np.max(action_returns)
    # round of the value function by 5 decimal places
    action_returns = np.round(action_returns,5)
    if (breakties == 1):
        max_action = action[np.argmax(action_returns)]
    elif (breakties == 2):
        max_action = action[np.flatnonzero(action_returns==action_returns.max())][-1]
    else:
        max_action = action[np.random.choice(np.flatnonzero(action_returns == action_returns.max()))]
    return max_value, max_action

def figure_4_3():
    # value iteration loop
    policy = np.zeros(STATES.size)
    while True:
        delta = 0.0 # initial value for \Delta
        for state in STATES:
            # get the value of the new state
            new_value, _ = policy_value(state, BREAK_TIES)
            # the worst error for the value function
            #delta = np.maximum(delta,np.abs(state_value[state] - new_value))
            delta += np.abs(state_value[state] - new_value) # better convergence criteria
            state_value[state] = new_value # update the state value function with new value

        if delta < 1e-9:
            break
    # extract the policy 
    for state in STATES:
        # update state value to this new value
        state_value[state], policy[state-1] = policy_value(state,BREAK_TIES)
 

    plt.figure(figsize=(20, 40))

    plt.subplot(2, 1, 1)
    plt.plot(state_value[1:GOAL-1])
    plt.xlabel('Capital')
    plt.ylabel('Value estimates')

    plt.subplot(2, 1, 2)
    plt.scatter(STATES, policy)
    plt.xlabel('Capital')
    plt.ylabel('Final policy (stake)')

    plt.savefig('figure_4_3.png')
    plt.show()
    plt.close()
    return policy, state_value

if __name__ == '__main__':
    policy, state_value = figure_4_3()
 
