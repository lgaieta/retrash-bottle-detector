import tensorflow as tf
import tensorflow_hub as tfhub


class BottleDetectorModel:
    def create_model(self):
        url = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4"
        mobilenetv2 = tfhub.KerasLayer(url, input_shape=(224, 224, 3))
        mobilenetv2.trainable = False

        model = tf.keras.Sequential([
            mobilenetv2,
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])

        model.summary()
        return model

    def compile_model(self, model):
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )

    def train_model(self, model, data, epochs=10, batch_size=32):
        model.fit(data, epochs=epochs, batch_size=batch_size)

    def save_weights(self, model, path):
        model.save_weights(path)

    def load_weights(self, model, path):
        model.load_weights(path)

    def test_model(self, model, data):
        x_train, y_train = data
        return model.evaluate(x_train, y_train, batch_size=32)
