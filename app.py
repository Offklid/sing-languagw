from flask import Flask, render_template, Response, redirect, url_for
import cv2
import mediapipe as mp
import numpy as np
import pickle

# Создание экземпляра приложения Flask
app = Flask(__name__)

# Инициализация MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Загрузка модели для распознавания жестов из файла
try:
    model_dict = pickle.load(open('./model.p', 'rb'))
    model = model_dict['model']
    print("Модель успешно загружена!")
except Exception as e:
    print(f"Ошибка загрузки модели: {e}")
    model = None

# Создаем объект для распознавания рук с заданными параметрами
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# Словарь для отображения предсказанных цифр на соответствующие метки
labels_dict = {0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9'}

# Переменные для управления видеопотоком
cap = None  # объект для захвата видео
streaming = False  # флаг, указывающий на состояние стриминга

# Функция для генерации кадров видеопотока
def generate_frames():
    global cap
    while streaming:
        success, frame = cap.read()  # считывание кадра из видеопотока
        if not success:
            break

        H, W, _ = frame.shape  # получение размеров кадра

        # Преобразование кадра из BGR в RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Обработка кадра для обнаружения рук
        results = hands.process(frame_rgb)
        if results.multi_hand_landmarks and model is not None:
            # Обработка каждой найденной руки
            for hand_landmarks in results.multi_hand_landmarks:
                data_aux = []  # вспомогательный список для данных
                x_ = []  # список координат x для ключевых точек руки
                y_ = []  # список координат y для ключевых точек руки

                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y

                    x_.append(x)
                    y_.append(y)

                # Нормализация координат ключевых точек
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    data_aux.append(x - min(x_))
                    data_aux.append(y - min(y_))

                # Определение координат для отрисовки прямоугольника вокруг руки
                x1 = int(min(x_) * W) - 10
                y1 = int(min(y_) * H) - 10

                x2 = int(max(x_) * W) - 10
                y2 = int(max(y_) * H) - 10

                try:
                    # Предсказание цифры с помощью модели
                    prediction = model.predict([np.asarray(data_aux)])

                    # Получение предсказанной цифры из словаря меток
                    predicted_character = labels_dict[int(prediction[0])]

                    # Отрисовка прямоугольника и предсказанной цифры на кадре
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
                    cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3,
                                cv2.LINE_AA)
                except Exception as e:
                    print(f"Ошибка предсказания: {e}")

        # Кодирование кадра в формат JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        # Генерация кадра для видеопотока
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# Маршрут для главной страницы
@app.route('/')
def index():
    return render_template('index.html')

# Маршрут для запуска видеопотока
@app.route('/start')
def start():
    global cap, streaming
    if not streaming:
        cap = cv2.VideoCapture(0)  # Открытие видеопотока с камеры (0 - первая камера)
        streaming = True
        print("Камера запущена")
    return redirect(url_for('index'))

# Маршрут для остановки видеопотока
@app.route('/stop')
def stop():
    global cap, streaming
    if streaming:
        streaming = False
        cap.release()  # Закрытие видеопотока
        cap = None
        print("Камера остановлена")
    return redirect(url_for('index'))

# Маршрут для получения видеопотока
@app.route('/video_feed')
def video_feed():
    if streaming:
        return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')
    else:
        return Response(b'')

if __name__ == '__main__':
    app.run(debug=True)