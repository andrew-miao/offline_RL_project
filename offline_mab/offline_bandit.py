import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
import random
import argparse

class Bandit:
    def __init__(self, expected_rewards, sample_reward_distributon, std=0.1):
        self.expected_rewards = expected_rewards
        self.std = std
        if sample_reward_distributon == 'Bernoulli':
            self.sample_reward = self.Bernoulli_rewards
        elif sample_reward_distributon == 'Normal_like':
            self.sample_reward = self.Normal_like_rewards
        elif sample_reward_distributon == 'Normal':
            self.sample_reward = self.Normal

    def Bernoulli_rewards(self, value, std=None):
        if np.random.rand() < value:
            return 1.0
        return 0.0

    def Normal_like_rewards(self, value):
        reward = np.random.normal(value, self.std)
        reward = min(1, reward)
        reward = max(0, reward)
        return reward
    
    def Normal(self, value):
        return np.random.normal(value, self.std)

    def getReward(self, action):
        reward = self.sample_reward(self.expected_rewards[action])
        return reward

    def generateData(self, data_sizes=[100, 10, 1], shuffle=False):
        data = []
        for action in range(len(data_sizes)):
            if data_sizes[action] > 0:
                for _ in range(data_sizes[action]):
                    reward = self.getReward(action)
                    data.append((action, reward))
        if shuffle:
            np.random.shuffle(data)
            return data
        return data

class DROmab:
    def __init__(self, 
                 action_space_size=3, 
                 f_divergence='tv', 
                 w_max=None, 
                 f_conjugate=None, 
                 f_conjugate_grad=None,
                 n_iterations=100,
                 lr_u=0.05,
                 lr_v=0.05,
                 epsilon=1e-16,):
        self.action_space_size = action_space_size
        self.w_max = w_max
        self.n_iterations = n_iterations
        self.lr_u = lr_u
        self.lr_v = lr_v
        self.epsilon = epsilon
        self.rewards = [0] * self.action_space_size
        self.objective_values = [0] * self.action_space_size
        if f_divergence == 'tv':
            self.f_conjugate = self.total_varation_conjugate
            self.f_conjugate_grad = self.total_varation_conjugate_grad
        elif f_divergence == 'chi_square':
            self.f_conjugate = self.chi_square_conjugate
            self.f_conjugate_grad = self.chi_square_conjugate_grad
        elif f_divergence == 'kl':
            self.f_conjugate = self.kl_conjugate
            self.f_conjugate_grad = self.kl_conjugate_grad
        else:
            if not f_conjugate or not f_conjugate_grad:
                raise Exception("not implemented")
            else:
                self.f_conjugate = f_conjugate
                self.f_conjugate_grad = f_conjugate_grad

    def reset(self):
        self.rewards = [0] * self.action_space_size
        self.objective_values = [0] * self.action_space_size
    
    def total_varation_conjugate(self, y):
        if y <= -0.5:
            return -0.5
        if y > -0.5 and y <= 0.5:
            return y
        if self.w_max:
            return (y - 0.5) * self.w_max + 0.5
        return (y - 0.5) * 100 + 0.5
    
    def total_varation_conjugate_grad(self, y):
        if y <= -0.5:
            return 0
        if y > -0.5 and y <= 0.5:
            return 1
        if self.w_max:
            return self.w_max
        return 100
    
    def chi_square_conjugate(self, y):
        return (y ** 2) / 4 + y
    
    def chi_square_conjugate_grad(self, y):
        return y/2 + 1

    def kl_conjugate(self, y):
        return np.exp(y - 1)
    
    def kl_conjugate_grad(self, y):
        return np.exp(y - 1)
    
    def max_uv(self, action, batch, delta, n_action):
        u, v = 0.5, 0.5
        u_grad, v_grad = 1, 1
        i = 0
        while (u_grad > self.epsilon or v_grad > self.epsilon) and i < self.n_iterations:
            i += 1
            u_grad, v_grad = 0, 0
            objective_value = 0
            for (a, r) in batch:
                if action == a:
                    y = -(r + v)/(u + self.epsilon)
                    conjugate_value = self.f_conjugate(y)
                    conjugate_grad = self.f_conjugate_grad(y)
                    objective_value += -u * conjugate_value - u * delta - v
                    u_grad += (-conjugate_value + y * conjugate_grad - delta)
                    v_grad += (conjugate_grad - 1)
            u_grad /= n_action
            v_grad /= n_action
            objective_value /= n_action
            u = max(0, u + self.lr_u * u_grad)
            v += self.lr_v * v_grad
        return objective_value

    def learn(self, batch):
        n_actions = [0] * self.action_space_size
        for (a, r) in batch:
            n_actions[a] += 1
            self.rewards[a] += r
        self.rewards = np.array(self.rewards)
        for action in range(self.action_space_size):
            if n_actions[action] == 0:
                self.objective_values[action] = -np.inf
            else:
                delta = self.action_space_size/n_actions[action]
                self.rewards[action] /= n_actions[action]
                self.objective_values[action] = self.max_uv(action, batch, delta, n_actions[action])
        self.objective_values = np.array(self.objective_values)

    def selectArm(self):
        return np.argmax(self.objective_values)

class LCB:
    def __init__(self, action_space_size):
        self.action_space_size = action_space_size
        self.rewards = [0] * self.action_space_size
        self.penalty = [0] * self.action_space_size

    def reset(self):
        self.rewards = [0] * self.action_space_size
        self.penalty = [0] * self.action_space_size

    def learn(self, batch):
        n_actions = [0] * self.action_space_size
        delta = 1 / len(batch)
        for (a, r) in batch:
            n_actions[a] += 1
            self.rewards[a] += r
        for action in range(self.action_space_size):
            if n_actions[action] == 0:
                self.penalty[action] = 1
            else:
                self.rewards[action] /= n_actions[action]
                self.penalty[action] = np.sqrt(np.log(2 * self.action_space_size / delta) / (2 * n_actions[action]))
    
    def selectArm(self):
        return np.argmax(np.array(self.rewards) - np.array(self.penalty))

def testBandit(bandit, action, n_eval=100):
    avg_reward = 0
    for _ in range(n_eval):
        avg_reward += bandit.getReward(action)
    return avg_reward/n_eval

def plot_curves(data, colors, labels, x=None):
    for i, y in enumerate(data):
        if x is None:
            x = np.arange(y.shape[0])
        mean = np.mean(y, axis=1)
        std = np.std(y, axis=1)
        plt.plot(x, np.mean(y, axis=1), color=colors[i], label=labels[i])
        plt.fill_between(x, mean-std, mean+std, color=colors[i], alpha=0.3)
    plt.legend(loc="best")
    plt.xlabel("Optimal Arm Samples")
    plt.ylabel("Avg Rewards")
    plt.title("Results of Different Bandit Algorithms")
    plt.grid()
    plt.savefig('./offline_bern_bandit.png')
    plt.show()

def create_data_sizes(action_space_size, optimal_action, sample_size=1000, optimal_sample=1):
    data_sizes = [0] * action_space_size
    data_sizes[optimal_action] = optimal_sample
    rest_samples = sample_size - data_sizes[optimal_action]
    fill_action_list = [False] * action_space_size
    fill_action_list[optimal_action] = True
    for action in range(action_space_size):
        if action != optimal_action:
            fill_action_list[action] = True
            if not (False in fill_action_list):
                data_sizes[action] = rest_samples
            else:
                data_sizes[action] = np.random.randint(0, rest_samples+1)
                rest_samples -= data_sizes[action]
    return data_sizes

def set_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--reward_d', type=str, default='Bernoulli', help='reward distribution, defalult: Bernoulli distribution')
    parser.add_argument('--n_iterations', type=int, default=200)
    parser.add_argument('--n_eval', type=int, default=500)
    parser.add_argument('--sample_size', type=int, default=1000)
    parser.add_argument('--lr_u', type=float, default=0.05)
    parser.add_argument('--lr_v', type=float, default=0.05)

    args = parser.parse_args()
    return args

def main(expected_rewards=[0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60],
         optimal_samples=np.arange(0, 501, 25)):
    args = set_config()
    n_actions = len(expected_rewards)
    bandit = Bandit(expected_rewards, args.reward_d)

    # algo_list = ['DRO TV', 'DRO chi-square', 'DRO KL', 'LCB']
    algo_list = ['DRO TV', 'DRO chi-square', 'LCB']
    dro_tv = DROmab(n_actions, f_divergence='tv', n_iterations=args.n_iterations, lr_u= args.lr_u, lr_v=args.lr_v)
    dro_chi_square = DROmab(n_actions, f_divergence='chi_square', n_iterations=args.n_iterations, lr_u=args.lr_u, lr_v=args.lr_v)
    # dro_kl = DROmab(n_actions, f_divergence='kl', n_iterations=n_iterations)
    lcb = LCB(n_actions)
    seeds = [0, 1, 2, 3, 4]
    plot_data = np.zeros((len(algo_list), len(optimal_samples), len(seeds)))

    for i, optimal_sample in tqdm(enumerate(optimal_samples)):
        for j, seed in enumerate(seeds):
            np.random.seed(seed)
            random.seed(seed)
            data_sizes = create_data_sizes(n_actions, np.argmax(expected_rewards), args.sample_size, optimal_sample)
            dataset = bandit.generateData(data_sizes, shuffle=True)
            dro_tv.reset()
            dro_chi_square.reset()
            # dro_kl.reset()
            lcb.reset()
            dro_tv.learn(dataset)
            dro_chi_square.learn(dataset)
            # dro_kl.learn(dataset)
            lcb.learn(dataset)
            plot_data[0][i][j] = testBandit(bandit, action=dro_tv.selectArm(), n_eval=args.n_eval)
            plot_data[1][i][j] = testBandit(bandit, action=dro_chi_square.selectArm(), n_eval=args.n_eval)
            # plot_data[2][i][j] = testBandit(bandit, action=dro_kl.selectArm(), n_eval=args.n_eval)
            plot_data[2][i][j] = testBandit(bandit, action=lcb.selectArm(), n_eval=args.n_eval)

    # labels = [r"DRO TV", r"DRO $\chi^2$", r"DRO KL", r"LCB"]
    # colors = ["coral", "gold", "limegreen", "skyblue"]
    labels = [r"DRO TV", r"DRO $\chi^2$", r"LCB"]
    colors = ["peru", "gold", "skyblue"]
    plt.axhline(y=np.max(expected_rewards), color = 'red', linestyle = '--')
    plot_curves(plot_data, colors, labels, optimal_samples)


if __name__ == "__main__":
    main()