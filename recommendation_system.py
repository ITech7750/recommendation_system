
import tensorflow as tf
from tensorflow.keras.layers import Input, Embedding, Concatenate, Dense
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt

class RecommenderModel(tf.keras.Model):
    def __init__(self, user_features, movie_features, genre_features):
        super(RecommenderModel, self).__init__()

        self.user_embedding = Embedding(user_features, 50, input_length=1)
        self.movie_embedding = Embedding(movie_features, 50, input_length=1)
        self.genre_embedding = Embedding(genre_features, 50, input_length=1)

        self.dense1 = Dense(64, activation='relu')
        self.dense2 = Dense(1, activation='sigmoid')

    def call(self, inputs):
        user_id, user_age, user_genre, movie_id, movie_rating, movie_genre, movie_year = inputs

        user = self.user_embedding(user_id)
        user_genre = self.genre_embedding(user_genre)
        movie = self.movie_embedding(movie_id)
        movie_genre = self.genre_embedding(movie_genre)

        x = Concatenate()([user, user_genre, movie, movie_genre])
        x = self.dense1(x)
        output = self.dense2(x)

        return output

user_input_data = np.array([
    [1, 25, 1],  # Пользователь с id=1, возраст=25, жанр=1
    [2, 30, 3],  # Пользователь с id=2, возраст=30, жанр=3
])

movie_input_data = np.array([
    [101, 4, 1, 2010],  # Фильм с id=101, рейтинг=4, жанр=1, год=2010
    [102, 3, 2, 2015],  # Фильм с id=102, рейтинг=3, жанр=2, год=2015
])

target = np.array([1, 0])  # Целевые значения для пользователей и фильмов

user_train, user_val, target_train, target_val = train_test_split(user_input_data, target, test_size=0.2)
movie_train, movie_val, _, _ = train_test_split(movie_input_data, target, test_size=0.2)

model = RecommenderModel(user_features=1000, movie_features=2000, genre_features=5)

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

history = model.fit([user_train[:, 0], user_train[:, 1], user_train[:, 2], movie_train[:, 0], movie_train[:, 1], movie_train[:, 2], movie_train[:, 3]], target_train,
          epochs=10, batch_size=32,
          validation_data=([user_val[:, 0], user_val[:, 1], user_val[:, 2], movie_val[:, 0], movie_val[:, 1], movie_val[:, 2], movie_val[:, 3]], target_val))

# Получаем точность на обучающей и тестовой выборках
train_accuracy = history.history['accuracy']
val_accuracy = history.history['val_accuracy']

# Визуализация точности
epochs = range(len(train_accuracy))

plt.figure()
plt.plot(epochs, train_accuracy, 'b', label='Training accuracy')
plt.plot(epochs, val_accuracy, 'r', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()


'''
model.fit([user_train, movie_train], target_train,
          epochs=10, batch_size=32,
          validation_data=([user_val, movie_val], target_val))

user_vector = np.array([1])  # Идентификатор пользователя для предсказания
movie_input_vector = np.array([101])  # Идентификатор фильма для предсказания
prediction = model.predict([user_vector, movie_input_vector])
print("Predicted rating:", prediction[0][0])
'''

