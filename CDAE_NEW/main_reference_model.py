#!/usr/bin/env python
import sys
import pickle
import numpy as np
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_auc_score
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, MaxPool1D, Activation, Dense, BatchNormalization, GlobalAveragePooling1D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
np.set_printoptions(threshold=np.inf)

def main(train):
    sample_rate = 250  # Hz
    time_frame = 4  # seconds
    input_length = sample_rate * time_frame

    # Load dataset
    x_all = np.concatenate((deserialization('noisy_nsr_ecg_250hz_4sec_1000size'),deserialization('noisy_af_ecg_250hz_4sec_1000size')), axis=0)
    y_all = np.concatenate((deserialization('nsr_label_ecg_250hz_4sec_1000size'),deserialization('af_label_ecg_250hz_4sec_1000size')), axis=0)
    print(y_all)

    pyplot.plot(x_all[1])
    pyplot.show()
    pyplot.close()

    RATIO = 0.3
    shuffle_index = np.random.permutation(len(x_all))
    test_len = int(RATIO * len(shuffle_index))

    test_index = shuffle_index[:test_len]
    train_index = shuffle_index[test_len:]

    x_valid,y_valid = x_all[test_index],y_all[test_index]
    x_train,y_train = x_all[train_index],y_all[train_index]

    print(x_train.shape)
    print(y_train.shape)

    if train:
        # Fit model
        model = get_model(input_dims=x_all[0].shape)
        history = model.fit(
            x_train,
            y_train,
            validation_data=(x_valid, y_valid),
            batch_size=32,
            epochs=50,
            callbacks=[
                EarlyStopping(monitor='val_loss', mode='min', patience=20),
                ModelCheckpoint('saved-reference-model.h5', monitor='val_accuracy', mode='max', save_best_only=True)
            ],
            verbose=2
        )

        model.summary()

        # Plot training history
        pyplot.plot(history.history['val_loss'], label='validation')
        pyplot.plot(history.history['loss'], label='training')
        pyplot.legend()
        pyplot.xlabel("# epochs")
        pyplot.ylabel("loss")
        pyplot.savefig("saved-reference-model_{}sec_training-history.svg".format(time_frame))


    # Evaluate model
    saved_model = load_model('saved-reference-model.h5')
    
    y_train_probs = saved_model.predict(x_train, verbose=2)
    y_valid_probs = saved_model.predict(x_valid, verbose=2)

    thresh = 0.5  # use prevalence as threshold
    print('Training set performance:')
    make_report(y_train, y_train_probs, thresh)
    print('Validation set performance:')
    make_report(y_valid, y_valid_probs, thresh)

def get_model(input_dims):
    input = Input(input_dims)
    x = BatchNormalization()(input)

    # 1st layer
    x = Conv1D(
        filters=6,
        kernel_size=10,
        strides=3,
        use_bias=False,
    )(x)
    x = BatchNormalization(
        scale=False
    )(x)
    x = Activation('relu')(x)
    x = MaxPool1D(
        pool_size=8,
        strides=6,
    )(x)

    # 2nd layer
    x = Conv1D(
        filters=3,
        kernel_size=1,
        use_bias=False,
    )(x)
    x = BatchNormalization(
        scale=False,
    )(x)
    x = Activation('relu')(x)

    # 3rd layer
    x = Conv1D(
        filters=4,
        kernel_size=6,
        use_bias=False,        
    )(x)
    x = BatchNormalization(
        scale=False,
    )(x)
    x = Activation('relu')(x)

    # 4th layer
    x = Conv1D(
        filters=3,
        kernel_size=3,
        use_bias=False,
    )(x)
    x = BatchNormalization(
        scale=False,
    )(x)
    x = Activation('relu')(x)
    x = MaxPool1D(
        pool_size=2,
        strides=2,
    )(x)

    # 5th layer
    x = Conv1D(
        filters=2,
        kernel_size=2,
        use_bias=False,
    )(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # averaging over complete feature maps
    x = GlobalAveragePooling1D()(x)

    # classifier
    x = Dense(
        units=1,
        use_bias=False,
    )(x)
    x = BatchNormalization()(x)
    output = Activation('sigmoid')(x)

    model = Model(input, output, name="ReferenceModel")

    model.compile(
        loss='binary_crossentropy',
        optimizer=Adam(
            learning_rate=0.015,
        ),
        metrics=['accuracy']
    )

    return model

def load_dataset(file_path):
    try:
        return pickle.load(open(file_path, 'rb'))
    except (pickle.UnpicklingError, FileNotFoundError):
        print("Dataset file not available!")
        exit(-1)

def make_report(y_actual, y_pred, thresh):
    tn, fp, fn, tp = confusion_matrix(y_actual, (y_pred > thresh)).ravel()
    
    accuracy = (tp + tn) / (tn + fp + fn + tp)
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    precision = tp/(tp+fp)
    auc = roc_auc_score(y_actual, y_pred)

    print('- accuracy: %.4f' % accuracy)
    print('- sensitivity: %.4f' % sensitivity)
    print('- specificity: %.4f' % specificity)
    print('- precision: %.4f' % precision)
    print('- AUC: %.4f' % auc)

    return tn, fp, fn, tp

def serialization(name, content):
    f = open(r'pickle\s_' + name, 'wb')
    pickle.dump(content, f)
    f.close()
def deserialization(name):
    f = open(r'pickle\s_' + name, 'rb')
    x = pickle.load(f)
    f.close()
    return x

if __name__ == '__main__':
    main(train=True)

