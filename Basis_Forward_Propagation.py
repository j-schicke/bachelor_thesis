import numpy as np 

def newton_euler(w:array, ct, cq, d):
    m = np.array([ct, ct, ct, ct], [0, d*ct, 0, -d*ct], [-d*ct, 0, d*ct, 0], [cq, cq, cq, cq])
    u = m * w**2
    return u



if __name__ == '__main__':
    newton_euler()
