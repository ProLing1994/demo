import numpy as np

def softmax(x):
    """ softmax """
    return np.exp(x) / np.sum(np.exp(x))

def main():
    x = np.array([0.1, 0.2, 0.4, 0.8])
    # x = np.array([10, 20, 40, 80])
    print(softmax(x))


if __name__ == "__main__":
    main()