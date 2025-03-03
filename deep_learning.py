import numpy as np
import pandas as pd
import tensorflow as tf
from keras.utils import plot_model


# Checking the version of tensorflow
print(tf.__version__)
# Are we running with Eager execution?
print(tf.executing_eagerly())

# Importing our data
train = pd.read_csv('deep_learning/train.csv')
print(train)
test = pd.read_csv('deep_learning/test.csv')
print(test)

# Splitting train and test sets
X_train = train.drop('label',axis=1).values.astype('float32')
y_train = train['label'].values
X_test = test.values.astype('float32')

print(np.max(X_train))

# Now we need to divide them all by 255
X_train = X_train/255.0
X_test = X_test/255.0

# Next we need to reshape our data for the convolutional network
X_train = X_train.reshape(X_train.shape[0], 28, 28,1)
X_test = X_test.reshape(X_test.shape[0], 28, 28,1)

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu', kernel_initializer='he_normal', input_shape=(28,28,1)))
model.add(tf.keras.layers.MaxPool2D((2, 2)))
model.add(tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal'))
model.add(tf.keras.layers.MaxPool2D((2, 2)))
model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal'))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Dense(10, activation='softmax'))

# Compiling model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Training model
model.fit(X_train, y_train, epochs=5)

#plot_model(model, to_file='model.png', show_shapes=True,
#    show_dtype=True,
#    show_layer_names=True,
#    rankdir="TB",
#    dpi=300,
#    show_layer_activations=True,
#    show_trainable=True)

prediction = model.predict(X_test)
print(prediction)
prediction = np.round(prediction, 5)
np.savetxt("predction.csv", prediction, delimiter=",", fmt='%10.5f')