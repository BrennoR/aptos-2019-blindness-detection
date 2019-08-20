from data_manipulation import train_df
import matplotlib.pyplot as plt

train_file_path = './data/training/'


def plot_data_grid(df=train_df):
    plt.figure(figsize=(18, 13))

    for i in range(5):
        class_lbl = i
        i *= 5
        j = 1
        class_imgs = df[df['class'] == '{}'.format(class_lbl)]
        img_paths = train_file_path + class_imgs['filename'].iloc[:5]
        for img_path in img_paths:
            plt.subplot(5, 5, i + j)
            plt.title('Class {}'.format(class_lbl))
            plt.imshow(plt.imread(img_path))
            plt.axis('off')
            j += 1

    plt.show()


def plot_acc_and_loss(history):
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(['train', 'validation'])
    plt.title('Model Accuracy')
    plt.show()

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(['train', 'validation'])
    plt.title('Model Loss')
    plt.show()


if __name__ == '__main__':
    plot_data_grid(train_df)
