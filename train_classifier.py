import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

# Загрузка данных из файла с использованием pickle
data_dict = pickle.load(open('./data.pickle', 'rb'))

# Преобразование данных и меток в numpy массивы
data = np.asarray(data_dict['data'])
labels = np.asarray(data_dict['labels'])

# Разделение данных на обучающую и тестовую выборки
# test_size=0.2 означает, что 20% данных будут использованы для тестирования
# stratify=labels гарантирует, что распределение классов в обучающей и тестовой выборках будет одинаковым
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

# Создание модели случайного леса (RandomForestClassifier)
model = RandomForestClassifier()

# Обучение модели на обучающих данных
model.fit(x_train, y_train)
# добавить графики
# Предсказание меток для тестовой выборки
y_predict = model.predict(x_test)

# Вычисление точности предсказаний модели
score = accuracy_score(y_predict, y_test)

# Вывод точности предсказаний модели в процентах
print('{}% of samples were classified correctly !'.format(score * 100))

# Сохранение обученной модели в файл с использованием pickle
with open('model.p', 'wb') as f:
    pickle.dump({'model': model}, f)
