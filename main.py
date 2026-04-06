from models import Models
from analysis import analysis
from code import code
from sklearn.model_selection import train_test_split
from model_optimization import ModelOptimizer
import os

if __name__ == '__main__':
  df=code.load_data("telecom_churn.csv")
  columns_to_drop = ['phone number']#Удаляем колонку с уникальными значениями
  df = df.drop(columns=columns_to_drop)#создаёт проблемму OneHotEncoder (создаётся 3333 новых столбцов)

  df['area code'] = df['area code'].astype('category')

  df['churn'] = df['churn'].astype(int)

  target = df['churn']
  preprocessor=code.transform(df,'churn')

  target_column = 'churn'
  X = df.drop(target_column, axis=1)
  y = df[target_column]

  X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42)

  X_train_transformed = preprocessor.fit_transform(X_train)
  X_test_transformed = preprocessor.transform(X_test)

  models = Models(X_train_transformed, X_test_transformed, y_train, y_test)

  accuracy_knn_test = models.knn_model()
  accuracy_dt_test = models.tree_model()
  accuracy_lr_test = models.logisticRegression_model()

  models_names = ['KNN', 'Decision Tree', 'Logistic Regression']
  models_accuracy = [accuracy_knn_test,accuracy_dt_test, accuracy_lr_test]
  analysis.graf(models_names, models_accuracy)

  '''
  Хотелось бы сделать автоматический поис лучшего результата, но пока работаем руками
  Я потом займусь этим для гарантии
  '''

  optimizer = ModelOptimizer(df)

  # Шаг 1: Диагностика
  baseline_recall, baseline_precision, baseline_f1 = optimizer.diagnose_baseline()

  # Шаг 2-3: Оптимизация
  grid_search = optimizer.optimize()

  # Шаг 4: Оценка
  optimized_metrics = optimizer.evaluate(grid_search)

  # Шаг 5: Сравнение
  optimizer.compare(baseline_recall, baseline_precision, baseline_f1, optimized_metrics)

