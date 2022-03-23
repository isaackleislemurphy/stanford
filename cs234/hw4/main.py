from abc import ABC, abstractmethod

import numpy as np
import csv
import os

import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

from data import load_data, LABEL_KEY
from scipy.stats import norm, multivariate_normal

import pdb

inv = np.linalg.inv

DOSE_CHOICES = ["low", "medium", "high"]

# LINEAR_WEIGHTS = {
#     "Age in decades": -0.2614,
#     "Height in cm": 0.0087,
#     "Weight in kg": 0.0128,
#     "VKORC1AG": 0.8677,
#     "VKORC1AA": 1.6974,
#     "VKORC1UN": 0.4854,
#     "CYP2C912": 0.5211,
#     "CYP2C913": 0.9357,
#     "CYP2C922": 1.0616,
#     "CYP2C923": 1.9206,
#     "CYP2C933": 2.3312,
#     "CYP2C9UN": 0.2188,
#     "Asian": 0.1092,
#     "Black": 0.2760,
#     "Unknown race": 0.1032,
#     # enzymes
#     "Carbamazepine (Tegretol)": 1.1816,
#     "Phenytoin (Dilantin)": 1.1816,
#     "Rifampin or Rifampicin": 1.1816,
#     # amaidorine
#     "Amiodarone (Cordarone)": 0.5503,
#     # "ENZYME": 1.1816,
#     # "AMIADORONE": 0.5503,
# }


LINEAR_WEIGHTS = {
    "Age in decades": -0.2614,
    "Height (cm)": 0.0118,
    "Weight (kg)": 0.0134,
    "Asian": -0.6752,
    "Black": 0.4060,
    "Unknown race": 0.0443,
    # enzymes
    "Carbamazepine (Tegretol)": 1.2799,
    "Phenytoin (Dilantin)": 1.2799,
    "Rifampin or Rifampicin": 1.2799,
    # amaidorine
    "Amiodarone (Cordarone)": -0.5695,
}


def dose_class(weekly_dose):
    if weekly_dose < 21:
        return "low"
    elif 21 <= weekly_dose and weekly_dose <= 49:
        return "medium"
    else:
        return "high"


# Base classes
class BanditPolicy(ABC):
    @abstractmethod
    def choose(self, x):
        pass

    @abstractmethod
    def update(self, x, a, r):
        pass


class StaticPolicy(BanditPolicy):
    def update(self, x, a, r):
        pass


class RandomPolicy(StaticPolicy):
    def __init__(self, probs=None):
        self.probs = probs if probs is not None else [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]

    def choose(self, x):
        return np.random.choice(("low", "medium", "high"), p=self.probs)


# Baselines
class FixedDosePolicy(StaticPolicy):
    def choose(self, x):
        """
        Args:
                x: Dictionary containing the possible patient features.
        Returns:
                output: string containing one of ('low', 'medium', 'high')

        TODO:
        Please implement the fixed dose algorithm.
        """
        #######################################################
        #########   YOUR CODE HERE - ~1 lines.   #############
        return "medium"
        #######################################################
        #########


class ClinicalDosingPolicy(StaticPolicy):
    def choose(self, x):
        """
        Args:
                x: Dictionary containing the possible patient features.
        Returns:
                output: string containing one of ('low', 'medium', 'high')

        TODO:
        Please implement the Clinical Dosing algorithm.

        Hint:
                - You may need to do a little data processing here.
                - Look at the "main" function to see the key values of the features you can use. The
                        age in decades is implemented for you as an example.
                - You can treat Unknown race as missing or mixed race.
                - Use dose_class() implemented for you.
        """
        #######################################################
        #########   YOUR CODE HERE - ~2-10 lines.   #############
        # "The output of this algorithm must be squared to compute weekly dose in mg. The output
        # of this algorithm must be squared and then divided by 7 to compute the daily dose in mg."
        inner_prod = 4.0376 + np.sum(
            [x[key] * LINEAR_WEIGHTS[key] for key in LINEAR_WEIGHTS]
        )
        algorithm_out = inner_prod ** 2
        return dose_class(algorithm_out)
        #######################################################
        #########


# Upper Confidence Bound Linear Bandit
class LinUCB(BanditPolicy):
    def __init__(self, n_arms, features, alpha=1.0):
        """
        See Algorithm 1 from paper:
                "A Contextual-Bandit Approach to Personalized News Article Recommendation"

        Args:
                n_arms: int, the number of different arms/ actions the algorithm can take
                features: list of strings, contains the patient features to use
                alpha: float, hyperparameter for step size.

        TODO:
        Please initialize the following internal variables for the Disjoint Linear Upper Confidence Bound Bandit algorithm.
        Please refer to the paper to understadard what they are.
        Please feel free to add additional internal variables if you need them, but they are not necessary.

        Hints:
        Keep track of a seperate A, b for each action (this is what the Disjoint in the algorithm name means)
        """
        #######################################################
        #########   YOUR CODE HERE - ~5 lines.   #############
        self.n_arms = n_arms
        self.features = features
        self.d = len(features)  # + 1
        self.alpha = alpha
        self.A = [np.eye(self.d) for _ in range(self.n_arms)]
        self.b = [np.zeros(self.d) for _ in range(self.n_arms)]

        #######################################################
        #########          END YOUR CODE.          ############

    def get_xvec(self, x):
        """
        Converts x as a dictionary into x as a feature vector.

        Args:
                x: Dictionary containing the possible patient features.
        Returns:
                output: np.array of x as a feature vector
        """
        return np.array(
            [
                # [1] + [
                x[feat]
                for feat in self.features
            ]
        )

    def choose(self, x):
        """
        See Algorithm 1 from paper:
                "A Contextual-Bandit Approach to Personalized News Article Recommendation"

        Args:
                x: Dictionary containing the possible patient features.
        Returns:
                output: string containing one of ('low', 'medium', 'high')

        TODO:
        Please implement the "forward pass" for Disjoint Linear Upper Confidence Bound Bandit algorithm.
        """
        #######################################################
        #########   YOUR CODE HERE - ~7 lines.   #############
        xvec = self.get_xvec(x)
        p_candidates = []
        for idx_a in range(self.n_arms):
            theta_ta = inv(self.A[idx_a]) @ self.b[idx_a]
            p_ta = theta_ta.T @ xvec + self.alpha * np.sqrt(
                xvec.T @ inv(self.A[idx_a]) @ xvec
            )
            p_candidates.append(p_ta)
        # "break ties uniformly"
        # best_choice_idx = np.random.choice(
        #     np.where(p_candidates == np.max(p_candidates))[0]
        # )
        best_choice_idx = np.argmax(p_candidates)
        return DOSE_CHOICES[best_choice_idx]
        #######################################################
        #########

    def update(self, x, a, r):
        """
        See Algorithm 1 from paper:
                "A Contextual-Bandit Approach to Personalized News Article Recommendation"

        Args:
                x: Dictionary containing the possible patient features.
                a: string, indicating the action your algorithem chose ('low', 'medium', 'high')
                r: the reward you recieved for that action
        Returns:
                Nothing

        TODO:
        Please implement the update step for Disjoint Linear Upper Confidence Bound Bandit algorithm.

        Hint: Which parameters should you update?
        """
        #######################################################
        #########   YOUR CODE HERE - ~4 lines.   #############
        xvec = self.get_xvec(x)
        a_idx = np.where(np.array(DOSE_CHOICES) == a)[0][0]
        self.A[a_idx] = np.add(self.A[a_idx], np.outer(xvec, xvec))
        self.b[a_idx] = np.add(self.b[a_idx], r * xvec)
        #######################################################
        #########          END YOUR CODE.          ############


# eGreedy Linear bandit
class eGreedyLinB(LinUCB):
    def __init__(self, n_arms, features, alpha=1.0):
        super(eGreedyLinB, self).__init__(n_arms, features, alpha=1.0)
        self.time = 0

    def choose(self, x):
        """
        Args:
                x: Dictionary containing the possible patient features.
        Returns:
                output: string containing one of ('low', 'medium', 'high')

        TODO:
        Instead of using the Upper Confidence Bound to find which action to take,
        compute the probability of each action using a simple dot product between Theta & the input features.
        Then use an epsilion greedy algorithm to choose the action.
        Use the value of epsilon provided
        """

        self.time += 1
        epsilon = float(1.0 / self.time) * self.alpha
        #######################################################
        #########   YOUR CODE HERE - ~7 lines.   #############
        xvec = self.get_xvec(x)
        p_candidates = []
        for idx_a in range(self.n_arms):
            theta_ta = inv(self.A[idx_a]) @ self.b[idx_a]
            p_ta = theta_ta.T @ xvec
            p_candidates.append(p_ta)
        # "break ties uniformly"
        best_choice_idx = np.random.choice(
            np.where(p_candidates == np.max(p_candidates))[0]
        )
        random_choice_idx = np.random.choice(
            list(range(self.n_arms))
        )
        u_draw = np.random.uniform()
        if u_draw < epsilon:
            return DOSE_CHOICES[random_choice_idx]
        return DOSE_CHOICES[best_choice_idx]
        #######################################################
        #########


# Thompson Sampling
class ThomSampB(BanditPolicy):
    def __init__(self, n_arms, features, alpha=1.0):
        """
        See Algorithm 1 and section 2.2 from paper:
                "Thompson Sampling for Contextual Bandits with Linear Payoffs"

        Args:
                n_arms: int, the number of different arms/ actions the algorithm can take
                features: list of strings, contains the patient features to use
                alpha: float, hyperparameter for step size.

        TODO:
        Please initialize the following internal variables for the Disjoint Thompson Sampling Bandit algorithm.
        Please refer to the paper to understadard what they are.
        Please feel free to add additional internal variables if you need them, but they are not necessary.

        Hints:
                - Keep track of a seperate B, mu, f for each action (this is what the Disjoint in the algorithm name means)
                - Unlike in section 2.2 in the paper where they sample a single mu_tilde, we'll sample a mu_tilde for each arm
                        based on our saved B, f, and mu values for each arm. Also, when we update, we only update the B, f, and mu
                        values for the arm that we selected
                - What the paper refers to as b in our case is the medical features vector
                - The paper uses a summation (from time =0, .., t-1) to compute the model paramters at time step (t),
                        however if you can't access prior data how might one store the result from the prior time steps.

        """

        #######################################################
        #########   YOUR CODE HERE - ~6 lines.   #############
        self.n_arms = n_arms
        self.features = features
        self.d = len(features)
        # Simply use alpha for the v mentioned in the paper
        self.v2 = alpha
        self.B = [np.eye(self.d) for _ in range(self.n_arms)]

        # Variable used to keep track of data needed to compute mu
        self.f = [np.zeros(self.d) for _ in range(self.n_arms)]

        # You can actually compute mu from B and f at each time step. So you don't have to use this.
        self.mu = [np.zeros(self.d) for _ in range(self.n_arms)]
        #######################################################
        #########          END YOUR CODE.          ############

    def get_xvec(self, x):
        """
        Converts x as a dictionary into x as a feature vector.

        Args:
                x: Dictionary containing the possible patient features.
        Returns:
                output: np.array of x as a feature vector
        """
        return np.array(
            [
                # [1] + [
                x[feat]
                for feat in self.features
            ]
        )

    def choose(self, x):
        """
        See Algorithm 1 and section 2.2 from paper:
                "Thompson Sampling for Contextual Bandits with Linear Payoffs"

        Args:
                x: Dictionary containing the possible patient features.
        Returns:
                output: string containing one of ('low', 'medium', 'high')

        TODO:
        Please implement the "forward pass" for Disjoint Thompson Sampling Bandit algorithm.
        Please use the gaussian distribution like they do in the paper
        """

        #######################################################
        #########   YOUR CODE HERE - ~8 lines.   #############
        xvec = self.get_xvec(x)
        candidate_draws = []
        for idx_a in range(self.n_arms):
            # unpack prior params
            mu_hat = self.mu[idx_a]
            B = self.B[idx_a]
            # take draw
            mu_tilde = multivariate_normal(mu_hat, self.v2 * inv(B)).rvs()
            # apply inner prod
            candidate_draws.append(xvec.T @ mu_tilde)
        # "break ties uniformly"
        best_choice_idx = np.argmax(candidate_draws)
        return DOSE_CHOICES[best_choice_idx]
        #######################################################
        #########          END YOUR CODE.          ############

    def update(self, x, a, r):
        """
        See Algorithm 1 and section 2.2 from paper:
                "Thompson Sampling for Contextual Bandits with Linear Payoffs"

        Args:
                x: Dictionary containing the possible patient features.
                a: string, indicating the action your algorithem chose ('low', 'medium', 'high')
                r: the reward you recieved for that action
        Returns:
                Nothing

        TODO:
        Please implement the update step for Disjoint Thompson Sampling Bandit algorithm.
        Please use the gaussian distribution like they do in the paper

        Hint: Which parameters should you update?
        """
        rew, act, xvec = r, a, self.get_xvec(x)
        #######################################################
        #########   YOUR CODE HERE - ~6 lines.   #############
        xvec = self.get_xvec(x)
        a_idx = np.where(np.array(DOSE_CHOICES) == a)[0][0]  # extract action index
        # update the appropriate terms
        self.B[a_idx] = np.add(self.B[a_idx], np.outer(xvec, xvec))
        self.f[a_idx] = np.add(self.f[a_idx], rew * xvec)
        self.mu[a_idx] = inv(self.B[a_idx]) @ self.f[a_idx]
        #######################################################
        #########          END YOUR CODE.          ############


def run(data, learner, large_error_penalty=False):
    # Shuffle
    data = data.sample(frac=1)
    T = len(data)
    n_egregious = 0
    correct = np.zeros(T, dtype=bool)
    for t in range(T):
        x = dict(data.iloc[t])
        label = x.pop(LABEL_KEY)
        action = learner.choose(x)
        correct[t] = action == dose_class(label)
        reward = int(correct[t]) - 1
        if (action == "low" and dose_class(label) == "high") or (
            action == "high" and dose_class(label) == "low"
        ):
            n_egregious += 1
            reward = large_error_penalty
        learner.update(x, action, reward)

    return {
        "total_fraction_correct": np.mean(correct),
        "average_fraction_incorrect": np.mean(
            [np.mean(~correct[:t]) for t in range(1, T)]
        ),
        "fraction_incorrect_per_time": [np.mean(~correct[:t]) for t in range(1, T)],
        "fraction_egregious": float(n_egregious) / T,
    }


def main(args):
    data = load_data()

    frac_incorrect = []
    features = [
        "Age in decades",
        "Height (cm)",
        "Weight (kg)",
        "Male",
        "Female",
        "Asian",
        "Black",
        "White",
        "Unknown race",
        "Carbamazepine (Tegretol)",
        "Phenytoin (Dilantin)",
        "Rifampin or Rifampicin",
        "Amiodarone (Cordarone)",
    ]

    extra_features = [
        "VKORC1AG",
        "VKORC1AA",
        "VKORC1UN",
        "CYP2C912",
        "CYP2C913",
        "CYP2C922",
        "CYP2C923",
        "CYP2C933",
        "CYP2C9UN",
    ]

    features = features + extra_features

    if args.run_fixed:
        avg = []
        for i in range(args.runs):
            print("Running fixed")
            results = run(data, FixedDosePolicy())
            avg.append(results["fraction_incorrect_per_time"])
            print(
                [(x, results[x]) for x in results if x != "fraction_incorrect_per_time"]
            )
        frac_incorrect.append(("Fixed", np.mean(np.asarray(avg), 0)))

    if args.run_clinical:
        avg = []
        for i in range(args.runs):
            print("Runnining clinical")
            results = run(data, ClinicalDosingPolicy())
            avg.append(results["fraction_incorrect_per_time"])
            print(
                [(x, results[x]) for x in results if x != "fraction_incorrect_per_time"]
            )
        frac_incorrect.append(("Clinical", np.mean(np.asarray(avg), 0)))

    if args.run_linucb:
        avg = []
        for i in range(args.runs):
            print("Running LinUCB bandit")
            results = run(
                data,
                LinUCB(3, features, alpha=args.alpha),
                large_error_penalty=args.large_error_penalty,
            )
            avg.append(results["fraction_incorrect_per_time"])
            print(
                [(x, results[x]) for x in results if x != "fraction_incorrect_per_time"]
            )
        frac_incorrect.append(("LinUCB", np.mean(np.asarray(avg), 0)))

    if args.run_egreedy:
        avg = []
        for i in range(args.runs):
            print("Running eGreedy bandit")
            results = run(
                data,
                eGreedyLinB(3, features, alpha=args.ep),
                large_error_penalty=args.large_error_penalty,
            )
            avg.append(results["fraction_incorrect_per_time"])
            print(
                [(x, results[x]) for x in results if x != "fraction_incorrect_per_time"]
            )
        frac_incorrect.append(("eGreedy", np.mean(np.asarray(avg), 0)))

    if args.run_thompson:
        avg = []
        for i in range(args.runs):
            print("Running Thompson Sampling bandit")
            results = run(
                data,
                ThomSampB(3, features, alpha=args.v2),
                large_error_penalty=args.large_error_penalty,
            )
            avg.append(results["fraction_incorrect_per_time"])
            print(
                [(x, results[x]) for x in results if x != "fraction_incorrect_per_time"]
            )
        frac_incorrect.append(("Thompson", np.mean(np.asarray(avg), 0)))

    os.makedirs("results", exist_ok=True)
    if frac_incorrect != []:
        for algorithm, results in frac_incorrect:
            with open(f"results/{algorithm}.csv", "w") as f:
                csv.writer(f).writerows(results.reshape(-1, 1).tolist())
    frac_incorrect = []
    for filename in os.listdir("results"):
        if filename.endswith(".csv"):
            algorithm = filename.split(".")[0]
            with open(os.path.join("results", filename), "r") as f:
                frac_incorrect.append(
                    (
                        algorithm,
                        np.array(list(csv.reader(f))).astype("float64").squeeze(),
                    )
                )
    plt.xlabel("examples seen")
    plt.ylabel("fraction_incorrect")
    legend = []
    for name, values in frac_incorrect:
        legend.append(name)
        plt.plot(values[10:])
    plt.ylim(0.0, 1.0)
    plt.legend(legend)
    plt.savefig(os.path.join("results", "fraction_incorrect.png"))


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--run-fixed", action="store_true")
    parser.add_argument("--run-clinical", action="store_true")
    parser.add_argument("--run-linucb", action="store_true")
    parser.add_argument("--run-egreedy", action="store_true")
    parser.add_argument("--run-thompson", action="store_true")
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--ep", type=float, default=1)
    parser.add_argument("--v2", type=float, default=0.001)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--large-error-penalty", type=float, default=-1)
    args = parser.parse_args()
    main(args)
