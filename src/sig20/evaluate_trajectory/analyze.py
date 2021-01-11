import matplotlib.pyplot as plt
import numpy as np
import scipy.stats

if __name__ == "__main__":
    data = [item.strip().split(",")[:-1] for item in open(r"D:\\SIG21_Local\\2_2_error_map\\asia18.txt").readlines()]
    data = np.asarray(data).astype(np.float32)
    print(scipy.stats.spearmanr(data[0], data[1]))
    print(scipy.stats.spearmanr(data[2], data[3]))

    a = data[0]
    b = data[1]
    mask = data[1] < 0.05
    print(scipy.stats.spearmanr(a[mask], b[mask]))
    

    # plt.scatter(data[0], data[1])
    plt.scatter(data[2], data[3])
    plt.ylim(0, 0.053)
    # plt.xlim(1, 300)
    # plt.scatter(data[0], data[2])
    plt.show()

    pass
    
