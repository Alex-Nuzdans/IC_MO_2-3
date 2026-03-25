import numpy as np
def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:

    n_classes = max(np.unique(y_pred))+1
    matrix = np.zeros((n_classes, n_classes),dtype=int)
    for i,j in zip(y_true, y_pred):
        matrix[i,j]+=1
    return matrix
