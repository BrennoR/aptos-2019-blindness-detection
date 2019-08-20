from data_manipulation import train_gen, valid_gen, test_gen
from visualization import plot_acc_and_loss
import pandas as pd
import numpy as np
from math import ceil
import matplotlib.pyplot as plt
from keras.applications.inception_v3 import InceptionV3
from keras.layers import Dense, GlobalAveragePooling2D, Dropout
from keras.models import Sequential
import keras.optimizers as optimizers


n_epochs = 20

STEP_SIZE_TRAIN = train_gen.n // train_gen.batch_size
STEP_SIZE_VALID = valid_gen.n // valid_gen.batch_size
STEP_SIZE_TEST = ceil(test_gen.n / test_gen.batch_size)

inceptionv3 = InceptionV3(weights='imagenet', include_top=False, input_shape=(299, 299, 3))

model = Sequential()

model.add(inceptionv3)
model.add(GlobalAveragePooling2D())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(5, activation='softmax'))

opt = optimizers.RMSprop(lr=0.001)

model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit_generator(generator=train_gen,
                              steps_per_epoch=STEP_SIZE_TRAIN,
                              validation_data=valid_gen,
                              validation_steps=STEP_SIZE_VALID,
                              epochs=n_epochs)

model.evaluate_generator(valid_gen, steps=STEP_SIZE_VALID)

test_gen.reset()
pred = model.predict_generator(test_gen,
                               steps=STEP_SIZE_TEST,
                               verbose=1)

predicted_class_indices = np.argmax(pred, axis=1)

labels = train_gen.class_indices
labels = dict((v, k) for k, v in labels.items())
predictions = [labels[k] for k in predicted_class_indices]

filenames = test_gen.filenames
results = pd.DataFrame({"id_code": filenames,
                        "diagnosis": predictions})

results.to_csv("model_1_results.csv", index=False)

model.save('model_1.h5')

plot_acc_and_loss(history)
