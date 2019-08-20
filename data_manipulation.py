import pandas as pd
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.inception_v3 import preprocess_input

train_file_path = './data/training'
test_file_path = './data/test'

train_df = pd.read_csv('./data/train.csv')
train_df.columns = ['filename', 'class']
train_df['filename'] = train_df['filename'].apply(lambda x: x + '.png')
train_df['class'] = train_df['class'].astype(str)

test_df = pd.read_csv('./data/test.csv')
test_df.columns = ['filename']
test_df['class'] = 'N/A'
test_df['filename'] = test_df['filename'].apply(lambda x: x + '.png')


train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input,
                                   horizontal_flip=True,
                                   vertical_flip=True,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   validation_split=0.2)

test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

train_gen = train_datagen.flow_from_dataframe(dataframe=train_df, directory=train_file_path, subset='training',
                                              class_mode='categorical', target_size=(299, 299))
valid_gen = train_datagen.flow_from_dataframe(dataframe=train_df, directory=train_file_path, subset='validation',
                                              class_mode='categorical', target_size=(299, 299))
test_gen = test_datagen.flow_from_dataframe(dataframe=test_df, directory=test_file_path,
                                            class_mode='categorical', target_size=(299, 299))

