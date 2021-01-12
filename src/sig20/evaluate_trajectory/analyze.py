import matplotlib.pyplot as plt
import numpy as np
import scipy.stats

if __name__ == "__main__":
    data = [item.strip().split(",")[:-1] for item in open(
        r"D:\\SIG21_Local\\2_2_error_map\\asia20.txt").readlines()]
    data = np.asarray(data).astype(np.float32)
    print(scipy.stats.spearmanr(data[0], data[1]))
    print(scipy.stats.spearmanr(data[2], data[3]))

    a = data[0]
    b = data[1]
    c = data[2]
    d = data[3]

    accuracy_x=[]
    accuracy_y=[]
    completeness_x=[]
    completeness_y=[]
    
    for i in range(0,20):
        max_value=(i+1)*5
        min_value=i*5
        mask1 = np.logical_and(min_value < a, a < max_value)
        mask2 = np.logical_and(min_value < c, c < max_value)
        accuracy_x.append(i*5)
        accuracy_y.append(scipy.stats.spearmanr(a[mask1], b[mask1])[0])
        completeness_x.append(i*5)
        completeness_y.append(scipy.stats.spearmanr(c[mask2], d[mask2])[0])

    # for i in range(0,20):
    #     max_value=(i+1)*0.05
    #     min_value=i*0.05
    #     mask1 = np.logical_and(min_value < b, b < max_value)
    #     mask2 = np.logical_and(min_value < d, d < max_value)
    #     accuracy_x.append(i*0.05)
    #     accuracy_y.append(scipy.stats.spearmanr(a[mask1], b[mask1])[0])
    #     completeness_x.append(i*0.05)
    #     completeness_y.append(scipy.stats.spearmanr(c[mask2], d[mask2])[0])

    # plt.scatter(data[0], data[1])
    plt.plot(accuracy_x, accuracy_y, label='Accuracy')
    plt.plot(completeness_x, completeness_y, label='Completeness')
    plt.legend(loc="lower right")
    # plt.xlabel('Mean Error')
    plt.xlabel('Reconstructability')
    plt.ylabel('Spearman Correlation')
    # plt.ylim(0, 0.053)
    # plt.xlim(1, 300)
    # plt.scatter(data[0], data[2])
    plt.show()

    pass
    
