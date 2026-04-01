import pandas as pd
import numpy as np



def main():
    # 1. Загрузка датасета
    df = pd.read_csv('telecom_churn.csv')
    print("=" * 50)
    print("РАЗБОР ДАННЫХ")
    print("=" * 50)

    # 2. Изучение данных
    print(f"\n1. Размер датасета: {df.shape[0]} строк, {df.shape[1]} столбцов")

    print("\n2. Типы столбцов:")
    print("-" * 30)

    # Определяем типы столбцов
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'string', 'bool']).columns.tolist()

    print(f"Числовые столбцы ({len(numeric_cols)}): {numeric_cols}")
    print(f"Категориальные столбцы ({len(categorical_cols)}): {categorical_cols}")

    print("\n3. Информация о данных:")
    print("-" * 30)
    print(df.info())

    print("\n4. Статистика пропусков:")
    print("-" * 30)
    missing_data = df.isnull().sum()
    missing_percent = (df.isnull().sum() / len(df)) * 100
    missing_table = pd.DataFrame({
        'Пропуски': missing_data,
        'Процент': missing_percent
    })
    missing_table = missing_table[missing_table['Пропуски'] > 0]
    if len(missing_table) > 0:
        print(missing_table)
    else:
        print("Пропусков в данных нет")

    # 5. Анализ целевой переменной (ИСПРАВЛЕНО для bool типа)
    print("\n5. Анализ целевой переменной (churn):")
    print("-" * 30)

    # Проверяем наличие колонки churn
    if 'churn' in df.columns:
        # Получаем value_counts для bool типа
        churn_counts = df['churn'].value_counts()

        # Преобразуем индексы для вывода (True/False -> 1/0)
        print(f"Клиенты, которые остались (0): {churn_counts[False] if False in churn_counts else 0} "
              f"({churn_counts[False] / len(df) * 100 if False in churn_counts else 0:.2f}%)")
        print(f"Клиенты, которые ушли (1): {churn_counts[True] if True in churn_counts else 0} "
              f"({churn_counts[True] / len(df) * 100 if True in churn_counts else 0:.2f}%)")

        # Вычисляем долю оттока (преобразуем bool в int для среднего)
        churn_numeric = df['churn'].astype(int)
        print(f"\nДоля оттока: {churn_numeric.mean():.2%}")

        # Дополнительная статистика
        print(f"\nРаспределение подробно:")
        print(f"  - Всего клиентов: {len(df)}")
        print(f"  - Остались (False/0): {churn_counts.get(False, 0)}")
        print(f"  - Ушли (True/1): {churn_counts.get(True, 0)}")
    else:
        print("Внимание: колонка 'churn' не найдена!")
        print(f"Доступные колонки: {df.columns.tolist()}")

    # 6. Первые строки данных
    print("\n6. Первые 5 строк данных:")
    print("-" * 30)
    print(df.head())

    # 7. Основная статистика числовых признаков
    print("\n7. Основная статистика числовых признаков:")
    print("-" * 30)
    if numeric_cols:
        print(df[numeric_cols].describe().round(2))

    # 8. Анализ категориальных признаков
    print("\n8. Анализ категориальных признаков:")
    print("-" * 30)
    categorical_for_analysis = [col for col in categorical_cols if col != 'churn']
    for col in categorical_for_analysis:
        print(f"\n{col}:")
        print(df[col].value_counts().head(3))

    # 9. Сохраняем базовую информацию
    print("\n" + "=" * 50)
    print("Анализ данных завершен")
    print("=" * 50)

    # Сохраняем информацию о датасете
    dataset_info = {
        'shape': df.shape,
        'numeric_cols': numeric_cols,
        'categorical_cols': categorical_cols,
        'churn_distribution': churn_counts.to_dict() if 'churn' in df.columns else {},
        'churn_rate': churn_numeric.mean() if 'churn' in df.columns else 0
    }

    # Сохраняем в файл
    import json
    with open('dataset_info.json', 'w', encoding='utf-8') as f:
        # Преобразуем numpy типы в стандартные Python типы
        info_serializable = {
            'shape': list(dataset_info['shape']),
            'numeric_cols': dataset_info['numeric_cols'],
            'categorical_cols': dataset_info['categorical_cols'],
            'churn_distribution': {str(k): int(v) for k, v in dataset_info['churn_distribution'].items()},
            'churn_rate': float(dataset_info['churn_rate'])
        }
        json.dump(info_serializable, f, indent=2, ensure_ascii=False)
    print("\nИнформация о датасете сохранена в 'dataset_info.json'")

    return df


if __name__ == "__main__":
    df = main()