### MDP Value Iteration and Policy Iteration
import argparse
import numpy as np
import gym
import time
from lake_envs import *

np.set_printoptions(precision=3)

parser = argparse.ArgumentParser(
	description="A program to run assignment 1 implementations.",
	formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)

parser.add_argument(
	"--env",
	help="The name of the environment to run your algorithm on.",
	choices=[
		"Deterministic-4x4-FrozenLake-v0",
		"Stochastic-4x4-FrozenLake-v0",
		"Deterministic-8x8-FrozenLake-v0"
	],
	default="Deterministic-4x4-FrozenLake-v0",
)


"""
For policy_evaluation, policy_improvement, policy_iteration and value_iteration,
the parameters P, nS, nA, gamma are defined as follows:

	P: nested dictionary
		From gym.core.Environment
		For each pair of states in [1, nS] and actions in [1, nA], P[state][action] is a
		tuple of the form (probability, nextstate, reward, terminal) where
			- probability: float
				the probability of transitioning from "state" to "nextstate" with "action"
			- nextstate: int
				denotes the state we transition to (in range [0, nS - 1])
			- reward: int
				either 0 or 1, the reward for transitioning from "state" to
				"nextstate" with "action"
			- terminal: bool
			  True when "nextstate" is a terminal state (hole or goal), False otherwise
	nS: int
		number of states in the environment
	nA: int
		number of actions in the environment
	gamma: float
		Discount factor. Number in range [0, 1)
"""

def bellman_operator_single(P, s, a, value_func, gamma):
	"""
	Applies a bellman backup for a given state, action, and value function.

	Parameters
	----------
	P:
		defined at beginning of file
	s: int
		The state you are in currently
	a: int
		The action you wish to take and evaluate
	value_func: np.array[nS]
		The current value function to be used in the Bellman backup
	gamma: float
		Discount in rewards
	Returns
	-------
	value_function: float
		The Q(s,a) result

	"""
	result = 0.
	for (trans_prob, s_, r, _) in P[s][a]:
		result += (
			# expectation over actions // if deterministic, it's unit mass
			trans_prob * (
				# reward
				r +
				# decayed sum of future value
				gamma * value_func[s_]
			)
		)
	return result



def policy_evaluation(P, nS, nA, policy, gamma=0.9, tol=1e-3, verbose=False):
	"""Evaluate the value function from a given policy.

	Parameters
	----------
	P, nS, nA, gamma:
		defined at beginning of file
	policy: np.array[nS]
		The policy to evaluate. Maps states to actions.
	tol: float
		Terminate policy evaluation when
			max |value_function(s) - prev_value_function(s)| < tol
	Returns
	-------
	value_function: np.ndarray[nS]
		The value function of the given policy, where value_function[s] is
		the value of state s
	"""

	value_function = np.zeros(nS)

	############################
	# YOUR IMPLEMENTATION HERE #
	# TODO: write tests

	# make copy for comparison, too tired to think of assignments/memory
	value_function_old = value_function.copy()

	k = 1 # init counter of iters
	while True:
		# init a new value function
		value_function_new = value_function_old.copy()
		for s in range(nS): # iter over states, per procedure
			# extract the action we wish to take
			action = policy[s] # infinite horizon, so det.
			# apply Bellman statewise
			value_function_new[s] = bellman_operator_single(
				P=P,
				s=s,
				a=action,
				value_func=value_function_old,
				gamma=gamma
			)
		# update counter
		k += 1
		if np.max(value_function_new - value_function_old) < tol:
			break
		value_function_old = value_function_new.copy()
	value_function = value_function_new
	############################
	return value_function


def policy_improvement(P, nS, nA, value_from_policy, policy, gamma=0.9):
	"""Given the value function from policy improve the policy.

	Parameters
	----------
	P, nS, nA, gamma:
		defined at beginning of file
	value_from_policy: np.ndarray
		The value calculated from the policy
	policy: np.array
		The previous policy.

	Returns
	-------
	new_policy: np.ndarray[nS]
		An array of integers. Each integer is the optimal action to take
		in that state according to the environment dynamics and the
		given value function.
	"""
	cur_policy = policy.copy()
	# can set to zero b/c we're monotonic increasing
	new_policy = np.zeros(nS, dtype="int")

	############################
	# YOUR IMPLEMENTATION HERE #

	for s in range(nS):
		Q_sa_vec = np.zeros(nA)
		for action in range(nA):
			# store state-action evaluation
			Q_sa_vec[action] = bellman_operator_single(
				P=P,
				s=s,
				a=action,
				value_func=value_from_policy,
				gamma=gamma
			)

		# new_policy[s] = np.where(Q_sa_vec == np.max(Q_sa_vec))[0][0]
		# do I need to randomize here
		new_policy[s] = np.argmax(Q_sa_vec)
	############################
	return new_policy


def policy_iteration(P, nS, nA, gamma=0.9, tol=10e-3):
	"""Runs policy iteration.

	You should call the policy_evaluation() and policy_improvement() methods to
	implement this method.

	Parameters
	----------
	P, nS, nA, gamma:
		defined at beginning of file
	tol: float
		tol parameter used in policy_evaluation()
	Returns:
	----------
	value_function: np.ndarray[nS]
	policy: np.ndarray[nS]
	"""

	value_function = np.zeros(nS)
	policy = np.zeros(nS, dtype=int)

	############################
	# YOUR IMPLEMENTATION HERE #
	# init a counter, and save old policy to compare against
	j, policy_old = 1, policy.copy()
	while True:
		# solve values, by policy eval
		value_from_policy = policy_evaluation(P, nS, nA, policy, gamma, tol)
		# solve policy, by policy improvement
		policy = policy_improvement(P, nS, nA, value_from_policy, policy, gamma)
		# update counter
		j += 1
		# check if policy changed
		if np.max(policy - policy_old) == 0:
			# one last eval at policy, for most up to date V
			value_function = policy_evaluation(P, nS, nA, policy, gamma, tol)
			# kill it
			break
		# otherwise start it at the top
		policy_old = policy.copy()
	############################
	return value_function, policy


def value_iteration(P, nS, nA, gamma=0.9, tol=1e-3):
	"""
	Learn value function and policy by using value iteration method for a given
	gamma and environment.

	Parameters:
	----------
	P, nS, nA, gamma:
		defined at beginning of file
	tol: float
		Terminate value iteration when
			max |value_function(s) - prev_value_function(s)| < tol
	Returns:
	----------
	value_function: np.ndarray[nS]
	policy: np.ndarray[nS]
	"""

	value_function = np.zeros(nS)
	value_function_old = value_function.copy()
	policy = np.zeros(nS, dtype=int)


	############################
	# YOUR IMPLEMENTATION HERE #
	while True:
		for s in range(nS):
			value_candidate = np.zeros(nA)
			for action in range(nA):
				value_candidate[action] = bellman_operator_single(
					P=P,
					s=s,
					a=action,
					value_func=value_function,
					gamma=gamma
				)
			# keep that one which does best
			value_function[s] = value_candidate[np.argmax(value_candidate)]
		if np.max(
			np.abs(
				value_function -
				value_function_old
			)
		) < tol:
			### choose the best policy, which can be done by recycling policy_improvement()
			policy = policy_improvement(P, nS, nA, value_function, policy, gamma)
			break
		else:
			value_function_old = value_function.copy()
	############################
	return value_function, policy


def render_single(env, policy, max_steps=100):
	"""
	This function does not need to be modified
	Renders policy once on environment. Watch your agent play!

	Parameters
	----------
	env: gym.core.Environment
	  Environment to play on. Must have nS, nA, and P as
	  attributes.
	Policy: np.array of shape [env.nS]
	  The action to take at a given state
  """

	episode_reward = 0
	ob = env.reset()
	for t in range(max_steps):
		env.render()
		time.sleep(0.25)
		a = policy[ob]
		ob, rew, done, _ = env.step(a)
		episode_reward += rew
		if done:
			break
	env.render()
	if not done:
		print("The agent didn't reach a terminal state in {} steps.".format(max_steps))
	else:
		print("Episode reward: %f" % episode_reward)


# Edit below to run policy and value iteration on different environments and
# visualize the resulting policies in action!
# You may change the parameters in the functions below
if __name__ == "__main__":
	# read in script argument
	args = parser.parse_args()

	# Make gym environment
	print("Starting environment.")
	env = gym.make(args.env)
	print("Environment made.")

	print("\n" + "-" * 25 + "\nBeginning Policy Iteration\n" + "-" * 25)

	V_pi, p_pi = policy_iteration(env.P, env.nS, env.nA, gamma=0.9, tol=1e-3)
	render_single(env, p_pi, 100)

	# print("\n" + "-" * 25 + "\nBeginning Value Iteration\n" + "-" * 25)

	V_vi, p_vi = value_iteration(env.P, env.nS, env.nA, gamma=0.9, tol=1e-3)
	render_single(env, p_vi, 100)



