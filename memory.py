import matplotlib.pyplot as plt
import numpy as np
import math

def main():
    requests = np.loadtxt('requests.csv', delimiter=',', dtype=np.int, skiprows=1)
    requests = requests[np.argsort(requests[:, 0])]
    print(requests.shape)
    plt.axes()
    colorMap = ["red", "green", "blue", "black"]
    for i in range(requests.shape[0]):
        row = requests[i]
        allocType = row[4]
        size = row[2]
        rect = plt.Rectangle((row[0], row[3]), (row[1] - row[0]), size, color=colorMap[allocType])
        plt.gca().add_patch(rect)
    plt.axis('scaled')
    plt.show()


if __name__ == "__main__":
    main()
