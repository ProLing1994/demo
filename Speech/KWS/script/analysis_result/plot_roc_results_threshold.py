import numpy as np
import matplotlib.pyplot as plt
import sys

def plot_roc(fpr, tpr, color, linestyle, label):
    plt.plot(fpr, tpr, color = color,  linewidth=1.0, linestyle=linestyle, marker = 'o', label = label)
    plt.legend(loc=4)
    plt.xlim([-0.01, 0.3])
    plt.ylim([90.0, 100.1])
    plt.xlabel('Flase Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')


def main():
    # init 
    fpr_list = [[0.06, 0.07, 0.07, 0.12, 0.14, 0.17, 0.21], [0, 0, 0, 0, 0.04, 0.10, 0.15]]
    tpr_list = [[92.6, 92.6, 95.77, 95.77, 95.77, 98.59, 100.0], [95.07, 95.07, 95.07, 95.07, 97.18, 97.18, 98.59]]
    color_list =  ["r", "g"]
    linestyle_list =  ["-", "-"]
    name_list = ["5.0", "6.0"]

    plt.figure()

    for idx in range(len(name_list)):
        
        fpr = fpr_list[idx]
        tpr = tpr_list[idx]
        plot_roc(fpr, tpr, color_list[idx], linestyle_list[idx], name_list[idx])

    plt.show()

if __name__ == "__main__":
    main()