
from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D, Flatten
from keras.optimizers import SGD, Adadelta

from keras.applications.imagenet_utils import  preprocess_input

from keras.utils import np_utils

import numpy as np
from os.path import join
from skimage.transform import resize
from scipy.misc import imread
from os import listdir
from keras.models import load_model

from keras.callbacks import ModelCheckpoint

def read_csv(filename):
    res = {}
    with open(filename) as fhandle:
        next(fhandle)
        for line in fhandle:
            parts = line.rstrip('\n').split(',')
            coords = np.array([float(x) for x in parts[1:]], dtype='float64')
            res[parts[0]] = coords
    return res

def train_classifier(train_gt, train_img_dir, fast_train=False):
    train_shape = 250

    filenames = listdir(train_img_dir)
    num_files = len(filenames)
    X_train = np.zeros((num_files,train_shape,train_shape,3), dtype = "float32")
    y_train = np.zeros((num_files, 1), dtype = "float32")


    for i, filename in enumerate(filenames):

        img = imread(join(train_img_dir, filename), mode = "RGB") #grayscale might work

        resized = resize(img, (train_shape, train_shape), preserve_range = True)
        X_train[i] = resized
        y_train[i] = train_gt[filename]

    if fast_train:
        num_epochs = 1
        num_layers = 100
    else:
        num_epochs = 15 #30 works fine as well
        num_layers = 20
    batch_size = 16
    fc_dense = 1024

    X_train = preprocess_input(X_train)

    Y_train = np_utils.to_categorical(y_train)
    num_classes = Y_train.shape[1]
    num_train = num_files

    X_val = X_train[::10]
    Y_val = Y_train[::10]

    Y_train = np.delete(Y_train, list(range(Y_train.shape[0])[::10]), axis = 0)
    X_train = np.delete(X_train, list(range(X_train.shape[0])[::10]), axis = 0)


    train_datagen = image.ImageDataGenerator(
        rotation_range=30,
        #width_shift_range=0.1,
        #height_shift_range=0.1,
        #shear_range=0.1,
        #zoom_range=0.1,
        horizontal_flip=True,
        fill_mode='nearest')


    train_generator = train_datagen.flow(X_train, Y_train,
        batch_size=batch_size)


    base_model = ResNet50(weights='imagenet', include_top=False, input_shape = (train_shape,train_shape,3))

    x = base_model.output
    flat = Flatten()(x)
    out = Dense(num_classes, activation="softmax")(flat)

    model = Model(inputs=base_model.input, outputs=out)

########################################################################
    #
    # for layer in base_model.layers:
    #      layer.trainable = False
    #
    # model.compile(optimizer=Adadelta(), loss='categorical_crossentropy',metrics=['accuracy']) #
    #
    # model.fit_generator(
    #     train_generator,
    #     samples_per_epoch=num_files,
    #     epochs=num_epochs, verbose = 1,
    #     validation_data=(X_val, Y_val))
###############################################################################################


    for layer in model.layers[:num_layers]:
         layer.trainable = False
    for layer in model.layers[num_layers:]:
        layer.trainable = True


    model.compile(optimizer=Adadelta(lr=0.1), loss='categorical_crossentropy',metrics=['accuracy']) #
    if not fast_train:
        checkpoint = ModelCheckpoint("model.hdf5", save_best_only=True, monitor='val_acc', mode='max')

        model.fit_generator(
            train_generator,
            samples_per_epoch=num_files,
            epochs=num_epochs, verbose = 1,
            callbacks=[checkpoint],
            validation_data=(X_val, Y_val))
        model.load_weights("model.hdf5")
    else:
        model.fit_generator(
            train_generator,
            samples_per_epoch=num_files,
            epochs=num_epochs, verbose = 1,
            validation_data=(X_val, Y_val))

    print("num_layers: ", num_layers, "batch: ", batch_size)
    return model

def classify(model, test_img_dir):
    test_shape = 250

    filenames = listdir(test_img_dir)
    X = np.zeros((len(filenames),test_shape,test_shape,3), dtype = "float32")

    for i, filename in enumerate(filenames):
        img = imread(join(test_img_dir, filename), mode = "RGB") #grayscale might work

        X[i] = resize(img, (test_shape, test_shape), preserve_range = True)
        del img

    X = preprocess_input(X)
    prediction = model.predict(X)

    result = {}
    for i, filename in enumerate(filenames):
        result[filename] = (np.argmax(prediction[i]))
    return result





if __name__=="__main__":
    train_gt = read_csv("birds-train/gt.csv")
    #model = train_classifier(train_gt, "birds-train/images")

    #model.save('birds_model.hdf5')
    model = load_model('birds_model.hdf5')
    a = classify(model, "img_test")
    plot_model(model, to_file='model.png')
    n = 0
    for i in a.keys():
        if a[i]== int(train_gt[i][0]):
            n+=1

    print(float(n)/len(a.keys()))






1
