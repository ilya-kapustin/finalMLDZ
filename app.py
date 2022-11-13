from flask import Flask, render_template, request, redirect
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

model = tf.keras.models.load_model('saved_model/1')
(train_x, train_y), (test_x, test_y) = tf.keras.datasets.mnist.load_data()
test_x = test_x.reshape(-1, 28, 28, 1).astype(np.float32) / 255.


def predict_digit(sample):
    prediction = model(sample[None, ...])[0]
    ans = np.argmax(prediction)

    fig = plt.figure(figsize=(12, 4))

    # Визуализация входного изображения
    ax = fig.add_subplot(1, 2, 1)
    ax.imshow(sample[:, :, 0], cmap='gray')
    plt.xticks([]), plt.yticks([])

    # Визуализация распределения вероятностей по классам
    ax = fig.add_subplot(1, 2, 2)
    bar_list = ax.bar(np.arange(10), prediction, align='center')
    bar_list[ans].set_color('g')
    ax.set_xticks(np.arange(10))
    ax.set_xlim([-1, 10])
    ax.grid(True)
    plt.show()
    print('Predicted number: {}'.format(ans))


app = Flask(__name__)

hed_menu = {
    'Главная': '/',
}


@app.route('/index/<int:idx>/')
def index(idx):
    sample = test_x[idx, ...]
    print(idx)
    predict_digit(sample)
    return render_template('index.html', menu=hed_menu, title='Главная')


if __name__ == '__main__':
    app.run(debug=True)
