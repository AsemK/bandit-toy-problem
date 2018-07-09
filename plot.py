import pickle
import numpy as np
import matplotlib.pyplot as plt


def load_results(dim, acq, const, init=1):
    f = open('res-' + str(dim)*init + '-' + acq + '-' + str(const) + '.pkl', 'rb')
    results, points, init_n, iter_n, acq, const = pickle.load(f)
    f.close()
    return results, points, init_n, iter_n, acq, const


def extract_data(results, points, init_n, iter_n):
    max_mean = []
    max_std = []
    avrg_mean = []
    avrg_std = []
    dist_mean = []
    dist_std = []
    for i in range(init_n, init_n + iter_n + 1):
        max_mean.append(np.mean(np.max(results[:, 0:i], 1)))
        max_std.append(np.std(np.max(results[:, 0:i], 1)))
        avrg_mean.append(np.mean(np.mean(results[:, 0:i], 1)))
        avrg_std.append(np.std(np.mean(results[:, 0:i], 1)))
        dist_mean.append(np.mean([np.sqrt(min(point[0:i-1].dot(point[i-1]))) for point in points]))
        dist_std.append(np.std([np.sqrt(min(point[0:i-1].dot(point[i-1]))) for point in points]))
    return max_mean, max_std, avrg_mean, avrg_std, dist_mean[1:], dist_std[1:]


def plot_param_study(dim, acq, param, consts, init=1):
    plt.figure(figsize=(6, 8))
    f1 = plt.subplot(3, 1, 1)
    f1.set_ylabel('Maximum Reward')
    f2 = plt.subplot(3, 1, 2)
    f2.set_ylabel('Average Reward')
    f3 = plt.subplot(3, 1, 3)
    f3.set_ylabel('Euclidean Distance')
    f3.set_xlabel('Evaluations of Unknown Function')
    legends = []
    init_n = iter_n = 0
    for const in consts:
        try:
            results, points, init_n, iter_n, acq, const = load_results(dim, acq, const, init)
        except FileNotFoundError:
            continue
        max_mean, max_std, avrg_mean, avrg_std, dist_mean, dist_std = extract_data(results, points, init_n, iter_n)
        f1.plot(list(range(init_n, init_n + iter_n + 1)), max_mean)
        f2.plot(list(range(init_n, init_n + iter_n + 1)), avrg_mean)
        f3.plot(list(range(init_n + 1, init_n + iter_n + 1)), dist_mean)
        legends.append(param + ' = ' + str(const))
        # plt.errorbar(list(range(init_n, init_n+iter_n)), max_found, yerr=error, errorevery=5)

    f1.set_xlim(init_n, init_n + iter_n)
    f2.set_xlim(init_n, init_n + iter_n)
    f3.set_xlim(init_n, init_n + iter_n)
    plt.figlegend(legends, loc='upper center', ncol=5, frameon=False, mode='expand')
    # plt.ylim(0.7, 1.05
    plt.gcf().subplots_adjust(top=0.93)
    plt.show()

def plot_dim(dims, acq, const, init=1):
    plt.figure(figsize=(6, 8))
    f1 = plt.subplot(3, 1, 1)
    f1.set_ylabel('Maximum Reward')
    f2 = plt.subplot(3, 1, 2)
    f2.set_ylabel('Average Reward')
    f3 = plt.subplot(3, 1, 3)
    f3.set_ylabel('Euclidean Distance')
    f3.set_xlabel('Evaluations of Unknown Function')
    legends = []
    init_n = iter_n = 0
    for dim in dims:
        try:
            results, points, init_n, iter_n, acq, const = load_results(dim, acq, const, init)
        except FileNotFoundError:
            continue
        max_mean, max_std, avrg_mean, avrg_std, dist_mean, dist_std = extract_data(results, points, init_n, iter_n)
        f1.plot(list(range(init_n, init_n + iter_n + 1)), max_mean)
        f2.plot(list(range(init_n, init_n + iter_n + 1)), avrg_mean)
        f3.plot(list(range(init_n + 1, init_n + iter_n + 1)), dist_mean)
        legends.append(str(dim)+'D')
        # plt.errorbar(list(range(init_n, init_n+iter_n)), max_found, yerr=error, errorevery=5)

    f1.set_xlim(init_n, init_n + iter_n)
    f2.set_xlim(init_n, init_n + iter_n)
    f3.set_xlim(init_n, init_n + iter_n)
    plt.figlegend(legends, loc='upper center', ncol=5, frameon=False, mode='expand')
    # plt.ylim(0.7, 1.05
    plt.gcf().subplots_adjust(top=0.93)
    plt.show()
    pass


# consts = [2.5, 2, 1.5, 1, 0.5, 0.1, 0.05, 0.01, 0.005, 0.001]
# plot_param_study(2, 'poi', 'xi', consts)
# plot_param_study(2, 'ei', 'xi', consts)
# plot_param_study(2, 'ucb', 'k', consts)

dims = [2, 3, 4, 5, 6]
plot_dim(dims, 'poi', 0.05)
plot_dim(dims, 'poi', 0.005)
plot_dim(dims, 'poi', 0.001)
plot_dim(dims, 'ucb', 1)
plot_dim(dims, 'poi', 0.005, 2)
plot_dim(dims, 'poi', 0.001, 2)
plot_dim(dims, 'ucb', 1, 2)
