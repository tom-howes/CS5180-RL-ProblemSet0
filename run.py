from env import ApartmentEnv
from policies import RandomPolicy, ThresholdPolicy, OptimalPolicy
import numpy as np
import matplotlib.pyplot as plt


class Empirical():

    def __init__(self, T, K, N, noisy=False, std=1.0):
        self.N = N
        self.apt = ApartmentEnv(T, K, noisy=noisy, std=std)
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
                    self.rewards.append(reward)
                    break
        return np.array(self.rewards), self.rejected / self.N


def stats(returns, rejected_frac, label):
    mean = np.mean(returns)
    se = np.std(returns) / np.sqrt(len(returns))
    print(f"{label}: mean={mean:.3f} +/- {se:.3f}, rejected all={rejected_frac:.3f}")
    return mean, se


def run_part_c(N, best_umin):
    print("\n=== Part (c): three-policy comparison, no noise ===")

    policies = [
        ("Random",                      RandomPolicy(4, 4)),
        (f"Threshold(umin={best_umin})", ThresholdPolicy(best_umin)),
        ("Optimal",                     OptimalPolicy(4, 4)),
    ]

    all_returns = []
    for name, policy in policies:
        e = Empirical(4, 4, N)
        returns, frac = e.run(policy)
        stats(returns, frac, name)
        all_returns.append((returns, name))

    # threshold sweep
    print("\n--- Threshold sweep ---")
    best_mean = -np.inf
    for umin in [1, 2, 3, 4]:
        e = Empirical(4, 4, N)
        returns, frac = e.run(ThresholdPolicy(umin))
        mean, _ = stats(returns, frac, f"Threshold(umin={umin})")
        if mean > best_mean:
            best_mean, best_umin = mean, umin
    print(f"best threshold: umin={best_umin}")

    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
    bins = np.arange(-0.5, 5.5, 1)
    for ax, (returns, label) in zip(axes, all_returns):
        ax.hist(returns, bins=bins, edgecolor="white")
        ax.axvline(np.mean(returns), color="red", linestyle="--")
        ax.set_title(label)
        ax.set_xlabel("Return")
    axes[0].set_ylabel("Count")
    plt.suptitle("Part (c) — Return distributions")
    plt.tight_layout()
    plt.savefig("part_c.png", dpi=150)
    plt.show()


def run_part_d(N, best_umin):
    print("\n=== Part (d): robustness to noise ===")

    sigmas = [0, 0.5, 1.0, 2.0]
    policy_names = ["Random", f"Threshold(umin={best_umin})", "Optimal"]
    results = {name: [] for name in policy_names}

    for sigma in sigmas:
        print(f"\n--- sigma={sigma} ---")
        for name, make_policy in zip(policy_names, [
            lambda: RandomPolicy(4, 4),
            lambda: ThresholdPolicy(best_umin),
            lambda: OptimalPolicy(4, 4),
        ]):
            e = Empirical(4, 4, N, noisy=sigma > 0, std=sigma)
            returns, frac = e.run(make_policy())
            mean, _ = stats(returns, frac, name)
            results[name].append(mean)

    fig, ax = plt.subplots(figsize=(7, 4))
    for name, means in results.items():
        ax.plot(sigmas, means, marker="o", label=name)
    ax.set_xlabel("Noise (sigma)")
    ax.set_ylabel("Mean utility")
    ax.set_title("Part (d) — Robustness to observation noise")
    ax.legend()
    plt.tight_layout()
    plt.savefig("part_d.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    N = 10000
    best_umin = 3  # from part (c)

    run_part_c(N, best_umin)
    run_part_d(N, best_umin)