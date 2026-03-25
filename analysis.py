import numpy as np
import matplotlib.pyplot as plt

def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
'''
Эта же функция с теми же переменными существует в
sklearn.metrics, так что если есть желание подключите её.
'''
    n_classes = max(np.unique(y_pred))+1
    matrix = np.zeros((n_classes, n_classes),dtype=int)
    for i,j in zip(y_true, y_pred):
        matrix[i,j]+=1
    return matrix
    
def graf(models,accuracy):
    plt.figure(figsize=(10,10))
    plt.bar(models,accuracy)
    plt.title("Диограмма точности")
    plt.xlabel("Модель")
    plt.ylabel("Точность")
    plt.show()
