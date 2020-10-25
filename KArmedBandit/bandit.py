import numpy as np

#Simple  e-greedy K-armend bandit problem implmentation

#Define number of iterations
ITERARIONS = 10000

#Define K
K = 8

#Define Epsilon
e = 0.01

# Define action rewards
rewards = np.zeros(K)
for i in range(K):
    reward = np.random.randint(1,K)
    rewards[i] = reward
    
print("Real action values = " + str(rewards))

# Initialize action values
action_values = np.zeros(K)

#Historic decisions and rewards
hist_decisions = []
hist_rewards = []

#Simulate decisions
for i in range(1,ITERARIONS):
    
    #Select action
    if np.random.normal(0,1,1) <= e:
        #decision = np.argmax(action_values)
        decision = np.random.choice(np.where(action_values == action_values.max())[0])
    else:   
        decision = np.random.randint(0,K)

    #Get reward
    reward = np.random.normal(rewards[decision],1)

    #Update action value
    new_value = action_values[decision] + 0.1*(reward - action_values[decision])
    action_values[decision] = new_value

    #Store decision and reward
    hist_decisions.append(decision)
    hist_rewards.append(reward)


#Print some data 
print("Action values aproximations = " + str(action_values))
print("Mean reward = " + str(np.mean(hist_rewards)))
print("Decisions taken:" + str(np.unique(hist_decisions,return_counts=True)))