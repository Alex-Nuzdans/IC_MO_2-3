from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression

'''
Код для разделения данных на обучающую и тестовую выборки. Использовать в предобработке.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)
'''
class Models:
    X_train = None
    X_test = None
    y_train = None
    y_test = None

    def __init__(self, X_train, X_test, y_train, y_test):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.y_knn_pred=None
        self.y_dt_pred=None
        self.y_lr_pred=None

    def knn_model(self):
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(self.X_train)
        X_test_scaled = scaler.transform(self.X_test)

        print("KNeighborsClassifier модель")

        knn_classifier = KNeighborsClassifier()

        # Обучаем модель на масштабированных данных
        knn_classifier.fit(X_train_scaled, self.y_train)

        # Делаем предсказание
        y_train_pred_knn = knn_classifier.predict(X_train_scaled)
        self.y_knn_pred = knn_classifier.predict(X_test_scaled)

        # Оцениваем точность модели
        accuracy_knn_train = accuracy_score(self.y_train, y_train_pred_knn)

        print("\nОБУЧАЮЩАЯ ВЫБОРКА:")
        print(f"  Accuracy:  {accuracy_knn_train:.4f}")

        # Тестовая выборка
        accuracy_knn_test = accuracy_score(self.y_test, self.y_knn_pred)

        print("\nТЕСТОВАЯ ВЫБОРКА:")
        print(f"  Accuracy:  {accuracy_knn_test:.4f}")
        return accuracy_knn_test

    def tree_model(self):
        print("DecisionTreeClassifier модель")
        dt_classifier = DecisionTreeClassifier( max_depth=7,min_samples_split=15,
                                                min_samples_leaf=8,random_state=42)

        # Обучение модели на тренировочных данных
        dt_classifier.fit(self.X_train, self.y_train)

        # Получение предсказаний на тестовых данных
        y_train_pred_dt = dt_classifier.predict(self.X_train)
        self.y_dt_pred = dt_classifier.predict(self.X_test)

        # Оценка точности модели
        accuracy_dt_train = accuracy_score(self.y_train, y_train_pred_dt)

        print("\nОБУЧАЮЩАЯ ВЫБОРКА:")
        print(f"  Accuracy:  {accuracy_dt_train:.4f}")
        # Тестовая выборка
        accuracy_dt_test = accuracy_score(self.y_test, self.y_dt_pred)
        print("\nТЕСТОВАЯ ВЫБОРКА:")
        print(f"  Accuracy:  {accuracy_dt_test:.4f}")
        return accuracy_dt_test

    def logisticRegression_model(self):
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(self.X_train)
        X_test_scaled = scaler.transform(self.X_test)

        print("logisticRegression модель")
        model = LogisticRegression(
        )
        model.fit(X_train_scaled, self.y_train)
        y_train_pred_lr = model.predict(X_train_scaled)
        self.y_lr_pred = model.predict(X_test_scaled)
        accuracy_lr_train = accuracy_score(self.y_train, y_train_pred_lr)

        print("ОБУЧАЮЩАЯ ВЫБОРКА:")
        print(f"  Accuracy:  {accuracy_lr_train:.4f}")
        accuracy_lr_test = accuracy_score(self.y_test, self.y_lr_pred)
        print("\nТЕСТОВАЯ ВЫБОРКА:")
        print(f"  Accuracy:  {accuracy_lr_test:.4f}")
        return accuracy_lr_test
