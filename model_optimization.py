import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (classification_report, confusion_matrix,
                             accuracy_score, precision_score, recall_score,
                             f1_score, roc_curve, roc_auc_score)
from sklearn.tree import DecisionTreeClassifier

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

from code import code


class ModelOptimizer:
    def __init__(self, df, target_column='churn'):
        """Инициализация: подготовка данных и разделение на train/test"""
        self.target = target_column
        self.X = df.drop(target_column, axis=1)
        self.y = df[target_column]

        # Разделение с сохранением пропорции классов
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.25, random_state=42, stratify=self.y
        )

        # Категориальные колонки
        self.cat_cols = ['state', 'area code', 'international plan', 'voice mail plan']
        self.num_cols = [col for col in self.X.columns if col not in self.cat_cols]

        print(f"\n=== Данные ===")
        print(f"Числовые ({len(self.num_cols)}): {self.num_cols[:5]}...")
        print(f"Категориальные ({len(self.cat_cols)}): {self.cat_cols}")

        # Создаем предобработчик
        self.preprocessor = ColumnTransformer([
            ('num', StandardScaler(), self.num_cols),
            ('cat', OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False), self.cat_cols)
        ])

        # Результаты
        self.baseline_model = None
        self.baseline_pred = None
        self.optimized_model = None
        self.optimized_pred = None

    def _encode_for_baseline(self):
        """Простое кодирование для базовой модели"""
        X_train = self.X_train.copy()
        X_test = self.X_test.copy()

        # Преобразуем yes/no в 1/0
        for col in ['international plan', 'voice mail plan']:
            X_train[col] = X_train[col].map({'yes': 1, 'no': 0})
            X_test[col] = X_test[col].map({'yes': 1, 'no': 0})

        # Кодируем остальные категории
        for col in ['state', 'area code']:
            le = LabelEncoder()
            all_vals = pd.concat([X_train[col], X_test[col]]).astype(str).unique()
            le.fit(all_vals)
            X_train[col] = le.transform(X_train[col].astype(str))
            X_test[col] = le.transform(X_test[col].astype(str))

        return X_train, X_test

    def diagnose_baseline(self):
        """Шаг 1: Диагностика текущей модели"""
        print("\n" + "=" * 60)
        print("ШАГ 1: ДИАГНОСТИКА БАЗОВОЙ МОДЕЛИ")
        print("=" * 60)

        # Подготовка и обучение
        X_train_enc, X_test_enc = self._encode_for_baseline()
        self.baseline_model = DecisionTreeClassifier(
            max_depth=7, min_samples_split=15, min_samples_leaf=8, random_state=42
        )
        self.baseline_model.fit(X_train_enc, self.y_train)
        self.baseline_pred = self.baseline_model.predict(X_test_enc)

        # Метрики
        print(classification_report(self.y_test, self.baseline_pred,
                                    target_names=['Не отток', 'Отток']))

        # Confusion Matrix
        cm = confusion_matrix(self.y_test, self.baseline_pred)
        plt.figure(figsize=(7, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Не отток', 'Отток'],
                    yticklabels=['Не отток', 'Отток'])
        plt.title('Confusion Matrix - Базовая модель')
        plt.ylabel('Истинные')
        plt.xlabel('Предсказанные')
        plt.tight_layout()
        plt.show()

        # Анализ дисбаланса
        dist = self.y_train.value_counts()
        total = len(self.y_train)
        print(f"\n--- Дисбаланс классов ---")
        print(f"Класс 0 (Не отток): {dist[0]} ({dist[0] / total * 100:.1f}%)")
        print(f"Класс 1 (Отток):   {dist[1]} ({dist[1] / total * 100:.1f}%)")

        # Анализ ошибок
        tn, fp, fn, tp = cm.ravel()
        print(f"\n--- Ошибки модели ---")
        print(f"False Negatives (пропущен отток): {fn} ← КРИТИЧНО!")
        print(f"False Positives (ложная тревога): {fp}")

        # Целевые метрики
        recall = recall_score(self.y_test, self.baseline_pred)
        precision = precision_score(self.y_test, self.baseline_pred)
        f1 = f1_score(self.y_test, self.baseline_pred)

        print(f"\n--- Метрики для класса 'Отток' ---")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f} ← нужно повысить до 0.70")
        print(f"F1-score:  {f1:.4f}")

        return recall, precision, f1

    def optimize(self):
        """Шаг 2-3: Создание конвейера и поиск лучших параметров"""
        print("\n" + "=" * 60)
        print("ШАГ 2-3: ОПТИМИЗАЦИЯ МОДЕЛИ")
        print("=" * 60)

        # Конвейер с SMOTE для борьбы с дисбалансом
        pipeline = ImbPipeline([
            ('preprocessor', self.preprocessor),
            ('smote', SMOTE(random_state=42)),
            ('classifier', DecisionTreeClassifier(random_state=42))
        ])

        # Параметры для поиска
        param_grid = {
            'classifier__max_depth': [5, 7, 10, 15],
            'classifier__min_samples_split': [5, 10, 15],
            'classifier__min_samples_leaf': [2, 4, 8],
            'classifier__criterion': ['gini', 'entropy']
        }

        print("\nПоиск лучших параметров (может занять минуту)...")
        grid_search = GridSearchCV(
            pipeline, param_grid, cv=5, scoring='recall', n_jobs=-1, verbose=0
        )
        grid_search.fit(self.X_train, self.y_train)

        print(f"\nЛучшие параметры: {grid_search.best_params_}")
        print(f"Recall на кросс-валидации: {grid_search.best_score_:.4f}")

        return grid_search

    def evaluate(self, grid_search):
        """Шаг 4: Финальная оценка модели"""
        print("\n" + "=" * 60)
        print("ШАГ 4: ФИНАЛЬНАЯ ОЦЕНКА")
        print("=" * 60)

        self.optimized_model = grid_search.best_estimator_
        self.optimized_pred = self.optimized_model.predict(self.X_test)
        proba = self.optimized_model.predict_proba(self.X_test)[:, 1]

        # Метрики
        metrics = {
            'accuracy': accuracy_score(self.y_test, self.optimized_pred),
            'precision': precision_score(self.y_test, self.optimized_pred),
            'recall': recall_score(self.y_test, self.optimized_pred),
            'f1': f1_score(self.y_test, self.optimized_pred),
            'auc_roc': roc_auc_score(self.y_test, proba)
        }

        print(f"\nAccuracy:  {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall:    {metrics['recall']:.4f}")
        print(f"F1-score:  {metrics['f1']:.4f}")
        print(f"AUC-ROC:   {metrics['auc_roc']:.4f}")

        print("\n--- Classification Report ---")
        print(classification_report(self.y_test, self.optimized_pred,
                                    target_names=['Не отток', 'Отток']))

        # Confusion Matrix
        cm = confusion_matrix(self.y_test, self.optimized_pred)
        plt.figure(figsize=(7, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Greens',
                    xticklabels=['Не отток', 'Отток'],
                    yticklabels=['Не отток', 'Отток'])
        plt.title('Confusion Matrix - Оптимизированная модель')
        plt.ylabel('Истинные')
        plt.xlabel('Предсказанные')
        plt.tight_layout()
        plt.show()

        # ROC-кривая
        fpr, tpr, _ = roc_curve(self.y_test, proba)
        plt.figure(figsize=(7, 5))
        plt.plot(fpr, tpr, linewidth=2, label=f'AUC = {metrics["auc_roc"]:.4f}')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC-кривая')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

        # Важность признаков
        self._plot_feature_importance()

        return metrics

    def _plot_feature_importance(self):
        """График важности признаков"""
        print("\n--- Важность признаков ---")

        # Получаем имена признаков
        preprocessor = self.optimized_model.named_steps['preprocessor']
        cat_features = preprocessor.named_transformers_['cat'].get_feature_names_out(self.cat_cols)
        all_features = list(self.num_cols) + list(cat_features)

        # Важность
        importance = self.optimized_model.named_steps['classifier'].feature_importances_
        importance_df = pd.DataFrame({'feature': all_features, 'importance': importance}) \
            .sort_values('importance', ascending=False)

        # Топ-10
        print("\nТоп-10 важных признаков:")
        for i, row in importance_df.head(10).iterrows():
            print(f"  {row['feature']}: {row['importance']:.4f}")

        # График
        plt.figure(figsize=(10, 7))
        top = importance_df.head(15)
        plt.barh(range(len(top)), top['importance'])
        plt.yticks(range(len(top)), top['feature'], fontsize=9)
        plt.xlabel('Важность')
        plt.title('Топ-15 важных признаков')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.show()

    def compare(self, baseline_recall, baseline_precision, baseline_f1, optimized_metrics):
        """Шаг 5: Сравнение моделей"""
        print("\n" + "=" * 60)
        print("ШАГ 5: СРАВНЕНИЕ МОДЕЛЕЙ")
        print("=" * 60)

        baseline_acc = accuracy_score(self.y_test, self.baseline_pred)

        comparison = pd.DataFrame({
            'Метрика': ['Accuracy', 'Precision', 'Recall', 'F1-score', 'AUC-ROC'],
            'Базовая': [
                f"{baseline_acc:.4f}", f"{baseline_precision:.4f}",
                f"{baseline_recall:.4f}", f"{baseline_f1:.4f}", "—"
            ],
            'Оптимизированная': [
                f"{optimized_metrics['accuracy']:.4f}",
                f"{optimized_metrics['precision']:.4f}",
                f"{optimized_metrics['recall']:.4f}",
                f"{optimized_metrics['f1']:.4f}",
                f"{optimized_metrics['auc_roc']:.4f}"
            ],
            'Изменение': [
                f"{(optimized_metrics['accuracy'] - baseline_acc) * 100:+.1f}%",
                f"{(optimized_metrics['precision'] - baseline_precision) * 100:+.1f}%",
                f"{(optimized_metrics['recall'] - baseline_recall) * 100:+.1f}%",
                f"{(optimized_metrics['f1'] - baseline_f1) * 100:+.1f}%",
                "—"
            ]
        })

        print(comparison.to_string(index=False))

        # Итог
        print(f"\n--- ИТОГ ---")
        if optimized_metrics['recall'] >= 0.70:
            print(f"✓ Цель достигнута! Recall = {optimized_metrics['recall']:.4f} (>= 0.70)")
        else:
            print(f"✗ Recall = {optimized_metrics['recall']:.4f} (< 0.70)")
            print(f"  Рекомендация: попробуйте RandomForestClassifier")

        return comparison


def main():
    print("=" * 60)
    print("ОПТИМИЗАЦИЯ ML-МОДЕЛИ ДЛЯ ПРОГНОЗИРОВАНИЯ ОТТОКА")
    print("=" * 60)

    # Загрузка данных
    df = code.load_data("telecom_churn.csv")
    df = df.drop(columns=['phone number'])
    df['churn'] = df['churn'].astype(int)

    # Преобразуем категории в строки
    for col in ['state', 'area code', 'international plan', 'voice mail plan']:
        df[col] = df[col].astype(str)

    print(f"\nДатасет: {df.shape[0]} строк, {df.shape[1]} колонок")

    # Запуск оптимизации
    optimizer = ModelOptimizer(df)

    # Шаг 1: Диагностика
    baseline_recall, baseline_precision, baseline_f1 = optimizer.diagnose_baseline()

    # Шаг 2-3: Оптимизация
    grid_search = optimizer.optimize()

    # Шаг 4: Оценка
    optimized_metrics = optimizer.evaluate(grid_search)

    # Шаг 5: Сравнение
    optimizer.compare(baseline_recall, baseline_precision, baseline_f1, optimized_metrics)

    print("\n" + "=" * 60)
    print("ГОТОВО!")
    print("=" * 60)


if __name__ == '__main__':
    main()