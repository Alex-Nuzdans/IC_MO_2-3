import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

class analysis:

    def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray,model_name) -> np.ndarray:
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Не отток (0)', 'Отток (1)'],
                    yticklabels=['Не отток (0)', 'Отток (1)'])
        plt.title(f'Confusion Matrix - {model_name}')
        plt.ylabel('Истинные значения')
        plt.xlabel('Предсказанные значения')
        plt.show()
    
    def graf(models,accuracy):
        plt.figure(figsize=(10,20))
        plt.bar(models,accuracy)
        plt.title("Диаграмма точности")
        plt.xlabel("Модели")
        plt.ylabel("Точность")
        plt.show()
