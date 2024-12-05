



from itertools import combinations

# Example data
state_list = ["state1", "state2", "state3", "state4"]
accuracy_list = [0.85, 0.92, 0.78, 0.89]

# Combine the lists into pairs
paired_list = zip(state_list, accuracy_list)

# Sort the pairs by accuracy in descending order
ranked_list = sorted(paired_list, key=lambda x: x[1], reverse=True)
print(ranked_list)
all_comb = list(combinations(ranked_list, r=2))
print(all_comb)



print([state[0][0] for state in all_comb])
print([state[1][0] for state in all_comb])
# Separate the sorted data back into ranked states and accuracies
# ranked_states, ranked_accuracies = zip(*ranked_list)

# # Print results
# print("Ranked States:", ranked_states)
# print("Ranked Accuracies:", ranked_accuracies)

