import tensorflow as tf
print("TensorFlow version:", tf.__version__)

def work():
    mnist = tf.keras.datasets.mnist

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10)
    ])

    predictions = model(x_train[:1]).numpy() # type: ignore
    print(f"og predictions: {predictions}")

    softmax_predictions = tf.nn.softmax(predictions).numpy() # type: ignore
    print(f"softmax predictions: {softmax_predictions}")

    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    loss_fn(y_train[:1], softmax_predictions).numpy() # type: ignore

    model.compile(optimizer='adam',
              loss=loss_fn,
              metrics=['accuracy'])
    print(f"model.fit w/ softmax preds loss_fn")
    model.fit(x_train, y_train, epochs=5)

    model.evaluate(x_test,  y_test, verbose=2) # type: ignore (verbose default is a str...but accepts ints)

    probability_model = tf.keras.Sequential([
    model,
    tf.keras.layers.Softmax()
    ])

    probability_model(x_test[:5])
    print(probability_model)

if __name__ == "__main__":
    work()