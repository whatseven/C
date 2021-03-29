import matplotlib.pyplot as plt
import numpy as np
import scipy.stats

if __name__ == "__main__":
    asia20_data = [item.strip().split(",")[:-1] for item in open(
        r"D:\\BACKUP\\SIG21_Local\\2_2_error_map\\asia20.txt").readlines()]
    asia20_data = np.asarray(asia20_data).astype(np.float32)

    asia18_data = [item.strip().split(",")[:-1] for item in open(
        r"D:\\BACKUP\\SIG21_Local\\2_2_error_map\\asia18.txt").readlines()]
    asia18_data = np.asarray(asia18_data).astype(np.float32)

    ours_data = [item.strip().split(",")[:-1] for item in open(
        r"D:\\BACKUP\\SIG21_Local\\2_2_error_map\\ours.txt").readlines()]
    ours_data = np.asarray(ours_data).astype(np.float32)

    asia18_reconstructability_recon=asia18_data[0]
    asia18_accuracy=asia18_data[1]
    asia18_reconstructability_gt=asia18_data[2]
    asia18_completeness = asia18_data[3]
    
    asia20_reconstructability_recon=asia20_data[0]
    asia20_accuracy=asia20_data[1]
    asia20_reconstructability_gt=asia20_data[2]
    asia20_completeness = asia20_data[3]
    
    ours_reconstructability_recon=ours_data[0]
    ours_accuracy=ours_data[1]
    ours_reconstructability_gt=ours_data[2]
    ours_completeness = ours_data[3]
    

    # print(scipy.stats.spearmanr(data[0], data[1]))
    # print(scipy.stats.spearmanr(data[2], data[3]))

    fig, ax = plt.subplots(2, 2)

    ax[0,0].hist(asia18_reconstructability_recon, density=True,bins=10,range=[0,100],cumulative=True)
    ax[0,0].set_ylabel('Accumulate Density')
    ax[0,0].set_xlabel('Reconstructability [Smith et al. 2018]')

    accuracy_x=[]
    accuracy_y=[]
    completeness_x=[]
    completeness_y=[]
    for i in range(0,20):
        max_value=(i+1)*5
        min_value=i*5
        mask1 = np.logical_and(min_value < asia18_reconstructability_recon, asia18_reconstructability_recon < max_value)
        mask2 = np.logical_and(min_value < asia18_reconstructability_gt, asia18_reconstructability_gt < max_value)
        accuracy_x.append(i*5)
        accuracy_y.append(scipy.stats.spearmanr(asia18_reconstructability_recon[mask1], asia18_accuracy[mask1])[0])
        completeness_x.append(i*5)
        completeness_y.append(scipy.stats.spearmanr(asia18_reconstructability_gt[mask2], asia18_completeness[mask2])[0])
    
    ax[0,1].plot(accuracy_x, accuracy_y, label='Accuracy')
    ax[0,1].plot(completeness_x, completeness_y, label='Completeness')
    ax[0,1].legend(loc="lower right")
    ax[0,1].set_xlabel('Reconstructability [Smith et al. 2018]')
    ax[0,1].set_ylabel('Spearman Correlation')

    ax[1,0] = plt.subplot(2, 2, 3)
    ax[1,0].hist(asia20_reconstructability_recon, density=True,bins=10,range=[0,100],cumulative=True)
    ax[1,0].set_ylabel('Accumulate Density')
    ax[1,0].set_xlabel('Reconstructability [Zhou et al. 2020]')

    accuracy_x=[]
    accuracy_y=[]
    completeness_x=[]
    completeness_y=[]
    for i in range(0,20):
        max_value=(i+1)*5
        min_value=i*5
        mask1 = np.logical_and(min_value < asia20_reconstructability_recon, asia20_reconstructability_recon < max_value)
        mask2 = np.logical_and(min_value < asia20_reconstructability_gt, asia20_reconstructability_gt < max_value)
        accuracy_x.append(i*5)
        accuracy_y.append(scipy.stats.spearmanr(asia20_reconstructability_recon[mask1], asia20_accuracy[mask1])[0])
        completeness_x.append(i*5)
        completeness_y.append(scipy.stats.spearmanr(asia20_reconstructability_gt[mask2], asia20_completeness[mask2])[0])
    
    ax[1,1].plot(accuracy_x, accuracy_y, label='Accuracy')
    ax[1,1].plot(completeness_x, completeness_y, label='Completeness')
    ax[1,1].legend(loc="lower right")
    ax[1,1].set_xlabel('Reconstructability [Smith et al. 2018]')
    ax[1,1].set_ylabel('Spearman Correlation')

    fig.tight_layout()
    plt.show()

    # a = data[0]
    # b = data[1]
    # c = data[2]
    # d = data[3]
    
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
    # plt.plot(accuracy_x, accuracy_y, label='Accuracy')
    # plt.plot(completeness_x, completeness_y, label='Completeness')
    # plt.legend(loc="lower right")
    # plt.xlabel('Mean Error')
    # plt.xlabel('Reconstructability')
    # plt.ylabel('Spearman Correlation')
    # plt.ylim(0, 0.053)
    # plt.xlim(1, 300)
    # plt.scatter(data[0], data[2])
    # plt.show()

    pass
    
