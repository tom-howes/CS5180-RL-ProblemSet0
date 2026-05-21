from env import ApartmentEnv
from policies import RandomPolicy, ThresholdPolicy, OptimalPolicy
import numpy as np
import matplotlib.pyplot as plt


class Empirical():

    def __init__(self, T, K, N):
        self.N = N
        self.apt = ApartmentEnv(T, K)
        self.rewards = []
        self.rejected = 0

    def run(self, policy):
        for episode in range(self.N):
            obs, _ = self.apt.reset()
            while True:
                action = policy.act(obs)
                obs, reward, terminated, truncated, info = self.apt.step(action)
                if terminated:
                    if obs[0] == 5:
                        self.rejected += 1
                    self.rewards.append(obs[1])
                    break
        return np.array(self.rewards), self.rejected / self.N


def stats(returns, rejected_frac, label):
    mean = np.mean(returns)
    se = np.std(returns) / np.sqrt(len(returns))
    print(f"{label}: mean={mean:.3f} +/- {se:.3f}, rejected all={rejected_frac:.3f}")
    return mean, se


if __name__ == "__main__":

    # random
    e = Empirical(4, 4, 10000)
    random_returns, frac = e.run(RandomPolicy(4, 4))
    stats(random_returns, frac, "Random")

    # threshold sweep
    best_umin, best_mean, threshold_returns = 1, -np.inf, {}
    for umin in [1, 2, 3, 4]:
        e = Empirical(4, 4, 10000)
        returns, frac = e.run(ThresholdPolicy(umin))
        threshold_returns[umin] = returns
        mean, _ = stats(returns, frac, f"Threshold umin={umin}")
        if mean > best_mean:
            best_mean, best_umin = mean, umin
    print(f"best threshold: umin={best_umin}")

    # optimal
    e = Empirical(4, 4, 10000)
    optimal_returns, frac = e.run(OptimalPolicy(4, 4))
    stats(optimal_returns, frac, "Optimal")

    # plot
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
    bins = np.arange(-0.5, 5.5, 1)
    for ax, (returns, label) in zip(axes, [
        (random_returns, "Random"),
        (threshold_returns[best_umin], f"Threshold (umin={best_umin})"),
        (optimal_returns, "Optimal"),
    ]):
        ax.hist(returns, bins=bins, edgecolor="white")
        ax.axvline(np.mean(returns), color="red", linestyle="--")
        ax.set_title(label)
        ax.set_xlabel("Return")

    axes[0].set_ylabel("Count")
    plt.tight_layout()
    plt.savefig("returns_histogram.png", dpi=150)
    plt.show()