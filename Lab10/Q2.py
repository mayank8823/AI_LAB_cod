import numpy as np

# Define parameters
discount_factor = 0.9
move_cost = 2
rental_reward = 10
max_bikes = 20
max_move = 5
requests_loc1 = 3
requests_loc2 = 4
returns_loc1 = 3
returns_loc2 = 2

# Initialize state-value function and policy
V = np.zeros((max_bikes + 1, max_bikes + 1))
policy = np.zeros((max_bikes + 1, max_bikes + 1), dtype=int)

# Define possible actions
actions = np.arange(-max_move, max_move + 1)
print(actions)

# Define Poisson distribution functions for rental and return
def poisson(n, lam):
    return np.power(lam, n) * np.exp(-lam) / np.math.factorial(n)

# Policy Iteration
while True:
    delta = 0
    for i in range(max_bikes + 1):
        for j in range(max_bikes + 1):
            old_v = V[i, j]
            action = policy[i, j]
            reward = 0
            expected_next_state_value = 0
            for rental_request_loc1 in range(requests_loc1 + 1):
                for rental_request_loc2 in range(requests_loc2 + 1):
                    for return_loc1 in range(returns_loc1 + 1):
                        for return_loc2 in range(returns_loc2 + 1):
                            num_rentals_loc1 = min(rental_request_loc1, i)
                            num_rentals_loc2 = min(rental_request_loc2, j)
                            rental_reward_loc1 = num_rentals_loc1 * rental_reward
                            rental_reward_loc2 = num_rentals_loc2 * rental_reward
                            prob_rental_loc1 = poisson(rental_request_loc1, requests_loc1)
                            prob_rental_loc2 = poisson(rental_request_loc2, requests_loc2)
                            prob_return_loc1 = poisson(return_loc1, returns_loc1)
                            prob_return_loc2 = poisson(return_loc2, returns_loc2)
                            prob = prob_rental_loc1 * prob_rental_loc2 * prob_return_loc1 * prob_return_loc2
                            next_state_loc1 = min(i - num_rentals_loc1 + return_loc1, max_bikes)
                            next_state_loc2 = min(j - num_rentals_loc2 + return_loc2, max_bikes)
                            expected_next_state_value += prob * V[next_state_loc1, next_state_loc2]
                            rental_reward_total = rental_reward_loc1 + rental_reward_loc2
                            reward += prob * rental_reward_total
            reward -= move_cost * abs(action)
            # print(i, j, reward)
            expected_value = reward + discount_factor * expected_next_state_value
            V[i, j] = expected_value
            delta = max(delta, abs(old_v - V[i, j]))
    if delta < 1e-6:
        break

# Policy Improvement
policy_stable = True
for i in range(max_bikes + 1):
    for j in range(max_bikes + 1):
        old_action = policy[i, j]
        q_values = np.zeros(actions.shape)
        for a_idx, action in np.ndenumerate(actions):
            if (0 <= (i - action) <= max_bikes) and (0 <= (j + action) <= max_bikes):
                q_values[a_idx] = -abs(action) * move_cost
                for rental_request_loc1 in range(requests_loc1 + 1):
                    for rental_request_loc2 in range(requests_loc2 + 1):
                        for return_loc1 in range(returns_loc1 + 1):
                            for return_loc2 in range(returns_loc2 + 1):
                                num_rentals_loc1 = min(rental_request_loc1, i - action)
                                num_rentals_loc2 = min(rental_request_loc2, j + action)
                                expected_rental_reward = (num_rentals_loc1 + num_rentals_loc2) * rental_reward
                                expected_bikes_loc1 = min(i - action - num_rentals_loc1 + return_loc1, max_bikes)
                                expected_bikes_loc2 = min(j + action - num_rentals_loc2 + return_loc2, max_bikes)
                                probability = poisson(rental_request_loc1, requests_loc1) * poisson(rental_request_loc2, requests_loc2) * poisson(return_loc1, returns_loc1) * poisson(return_loc2, returns_loc2)
                                q_values[a_idx] += probability * (expected_rental_reward + discount_factor * V[expected_bikes_loc1, expected_bikes_loc2])
        best_action = actions[np.argmax(q_values)]
        policy[i, j] = best_action
        if old_action != best_action:
            policy_stable = False


print(policy)
print(V)
print(reward)
print(q_values)

