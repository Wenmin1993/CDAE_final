import numpy as np
from matplotlib import pyplot
import matplotlib.pyplot as plt
import pickle
import tensorflow as tf
from tensorflow.keras import metrics
from tensorflow.keras import backend as k
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model,Model
from tensorflow.keras.layers import Input,Conv1D,BatchNormalization,Activation,MaxPooling1D,Conv1DTranspose,UpSampling1D,GlobalAveragePooling1D
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
from wfdb.processing import normalize_bound

def rmse(y_true, y_pred):
        return k.sqrt(k.mean(k.square(y_pred - y_true)))

def prd(y_true, y_pred):
        squared_differnece = k.sum(k.square(y_true- y_pred))
        squared_d = k.sum(k.square(y_true))
        prd = k.sqrt(squared_differnece / squared_d)
        return prd * 100

def prdn(y_true, y_pred):
        y_true_mean = k.mean(y_true)
        squared_differnece = k.sum(k.square(y_true - y_pred))
        squared_d_m = k.sum(k.square(y_true - y_true_mean))
        prdn = k.sqrt(squared_differnece / squared_d_m)
        return prdn * 100

def snr(y_true,y_pred):
        y_true_mean = k.mean(y_true)
        divi_squared_differnece = k.sum(k.square(y_true - y_true_mean))/k.sum(k.square(y_true-y_pred))
        snr = 10.0*k.log(divi_squared_differnece)/k.log(10.0)
        return snr

def qs(y_true, y_pred):
    qs = cr(y_true, y_pred) / prd(y_true, y_pred)
    return qs

def serialization(name, content):
    f = open(r'pickle\s_' + name, 'wb')
    pickle.dump(content, f)
    f.close()

def deserialization(name):
    f = open(r'pickle\s_' + name, 'rb')
    x = pickle.load(f)
    f.close()
    return x

def load_dataset():
    try:

        #修改mode
        x_noisy_nsr = deserialization('noisy_nsr_ecg_250hz_4sec_1000size')
        x_noisy_af = deserialization('noisy_af_ecg_250hz_4sec_1000size')
        x_original_nsr = deserialization('original_nsr_ecg_250hz_4sec_1000size')
        x_original_af = deserialization('original_af_ecg_250hz_4sec_1000size')

        X_noisy = np.concatenate((x_noisy_nsr, x_noisy_af), axis=0)
        X_original = np.concatenate((x_original_nsr, x_original_af), axis=0)

        # build train & test dataset
        RATIO = 0.3
        shuffle_index = np.random.permutation(len(X_noisy))
        test_len = int(RATIO * len(shuffle_index))

        test_index = shuffle_index[:test_len]
        train_index = shuffle_index[test_len:]

        X_noisy_test = X_noisy[test_index]
        X_noisy_train = X_noisy[train_index]
        X_original_test = X_original[test_index]
        X_original_train = X_original[train_index]
        print(' X_noisy_test shape:', X_noisy_test.shape)
        
        serialization('X_noisy_test',X_noisy_test)
        serialization('X_noisy_train',X_noisy_train)
        serialization('X_original_test',X_original_test)
        serialization('X_original_train',X_original_train)

        X_noisy_test = deserialization('X_noisy_test')
        X_noisy_train = deserialization('X_noisy_train')
        X_original_test = deserialization('X_original_test')
        X_original_train = deserialization('X_original_train')


        id = np.random.randint(0, len(X_noisy_test))
        plt.subplot(411)
        plt.plot(X_noisy_train[id])
        plt.title('Noisy ECG in train set_' + str(id))
        plt.subplot(412)
        plt.plot(X_original_train[id])
        plt.title('Original ECG in train set_' + str(id))
        plt.subplot(413)
        plt.plot(X_noisy_test[id])
        plt.title('Noisy ECG in test set_' + str(id))
        plt.subplot(414)
        plt.plot(X_original_test[id])
        plt.title('Original ECG in test set_' + str(id))
        # plt.savefig(name+'.png')
        plt.show()

        return X_noisy_test,X_noisy_train,X_original_test,X_original_train
    except (pickle.UnpicklingError, FileNotFoundError):
        print('Dataset file not available!')
        exit(-1)

#7-layer cnn
def get_model_3(input_dims):
    input = Input(input_dims)
    x = BatchNormalization()(input)

    #encoder
    #1st layer
    x = Conv1D(filters=8, kernel_size=3,padding='same',use_bias=False)(x)
    x = Activation('tanh')(x)
    x = MaxPooling1D(pool_size = 2, strides=2)(x)

    #2nd layer
    x = Conv1D(filters=16, kernel_size=3, padding='same',use_bias=False)(x)
    x = Activation('tanh')(x)
    x = MaxPooling1D(pool_size = 2, strides=2)(x)

    #3rd layer
    x = Conv1D(filters=24, kernel_size=3, padding='same',use_bias=False)(x)
    x = Activation('tanh')(x)
    x = MaxPooling1D(pool_size = 2, strides=2)(x)

    #4th layer
    x = Conv1D(filters=24, kernel_size=3, padding='same', use_bias=False)(x)
    encoded = Activation('tanh')(x)

    #decoder
    #5th layer
    x = UpSampling1D(size=2)(encoded)
    x = Conv1D(filters=16, kernel_size=3, padding='same',use_bias=False)(x)
    x = Activation('tanh')(x)

    #6th layer
    x = UpSampling1D(size=2)(x)
    x = Conv1D(filters=8, kernel_size=3, padding='same',use_bias=False)(x)
    x = Activation('tanh')(x)

    #7th layer
    x = UpSampling1D(size=2)(x)
    x = Conv1D(filters=1, kernel_size=3, padding='same',use_bias=False)(x)
    decoded = Activation('tanh')(x)

    model = Model(input,decoded,name='CDAEModel_3')
    encoder = Model(inputs=input, outputs=encoded)
    model.compile(loss='mse',
                  optimizer=Adam(learning_rate=0.0005, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False,name='Adam'),
                  metrics=[metrics.RootMeanSquaredError(),prd,prdn,snr]
                  )
    return model,encoder

#5layer CNN used in thesis
def get_model_2(input_dims):
    input = Input(input_dims)
    x = BatchNormalization()(input)

    #encoder
    #1st layer
    x = Conv1D(filters=8, kernel_size=3,padding='same',use_bias=False)(x)
    x = Activation('tanh')(x)
    x = MaxPooling1D(pool_size = 2, strides=2)(x)

    #2nd layer
    x = Conv1D(filters=16, kernel_size=3, padding='same',use_bias=False)(x)
    x = Activation('tanh')(x)
    x = MaxPooling1D(pool_size = 2, strides=2)(x)

    #3rd layer
    x = Conv1D(filters=16, kernel_size=3, padding='same',use_bias=False)(x)
    encoded = Activation('tanh')(x)

    #decoder
    #4th layer
    x = UpSampling1D(size=2)(encoded)
    x = Conv1D(filters=8, kernel_size=3, padding='same',use_bias=False)(x)
    x = Activation('tanh')(x)

    #5th layer
    x = UpSampling1D(size=2)(x)
    x = Conv1D(filters=1, kernel_size=3, padding='same',use_bias=False)(x)
    decoded = Activation('tanh')(x)

    model = Model(input,decoded,name='CDAEModel_2')
    encoder = Model(inputs=input, outputs=encoded)
    model.compile(loss='mse',
                  optimizer=Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False,name='Adam'),
                  metrics=[metrics.RootMeanSquaredError(),prd,prdn,snr]
                  )
    return model,encoder

#9-layer CNN
def get_model(input_dims):
    input = Input(input_dims)
    x = BatchNormalization()(input)

    # encoder
    # 1st layer
    x = Conv1D(filters=8, kernel_size=3, padding='same', use_bias=False)(x)
    x = Activation('tanh')(x)
    x = MaxPooling1D(pool_size=2, strides=2)(x)

    # 2nd layer
    x = Conv1D(filters=16, kernel_size=3, padding='same', use_bias=False)(x)
    x = Activation('tanh')(x)
    x = MaxPooling1D(pool_size=2, strides=2)(x)

    # 3rd layer
    x = Conv1D(filters=24, kernel_size=3, padding='same', use_bias=False)(x)
    x = Activation('tanh')(x)
    x = MaxPooling1D(pool_size=2, strides=2)(x)

    # 4rd layer
    x = Conv1D(filters=32, kernel_size=3, padding='same', use_bias=False)(x)
    x = Activation('tanh')(x)
    x = MaxPooling1D(pool_size=2, strides=2)(x)

    # 5th layer
    x = Conv1D(filters=32, kernel_size=3, padding='same', use_bias=False)(x)
    encoded = Activation('tanh')(x)

    # decoder
    # 6th layer
    x = UpSampling1D(size=2)(encoded)
    x = Conv1D(filters=16, kernel_size=3, padding='same', use_bias=False)(x)
    x = Activation('tanh')(x)

    # 7th layer
    x = UpSampling1D(size=2)(x)
    x = Conv1D(filters=8, kernel_size=3, padding='same', use_bias=False)(x)
    x = Activation('tanh')(x)

    # 8th layer
    x = UpSampling1D(size=2)(x)
    x = Conv1D(filters=8, kernel_size=3, padding='same', use_bias=False)(x)
    x = Activation('tanh')(x)

    # 9th layer
    x = UpSampling1D(size=2)(x)
    x = Conv1D(filters=1, kernel_size=3, padding='same', use_bias=False)(x)
    decoded = Activation('sigmoid')(x)



    model = Model(input, decoded, name='CDAEModel_3')
    encoder = Model(inputs=input, outputs=encoded)
    model.compile(loss='mse',
                  optimizer=Adam(learning_rate=0.0005, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False,
                                 name='Adam'),
                  metrics=[metrics.RootMeanSquaredError(),prd, prdn, snr]
                  )

    return model,encoder

def main(train):
        cr =[]
        sample_rate = 250  # Hz
        time_frame = 4  # seconds
        #input_length = sample_rate * time_frame

        #load dataset
        X_noisy_test,X_noisy_train,X_original_test,X_original_train = load_dataset()


        if train :
            #Fit model
            model,encoder = get_model_2(input_dims = X_noisy_train[0].shape)
            history = model.fit(X_noisy_train,X_original_train,epochs=100,batch_size=70,shuffle=True,verbose=1,
                                validation_data=(X_noisy_test, X_original_test),
                                callbacks=[
                                     EarlyStopping(monitor='val_loss', mode='min', patience=20),
                                     ModelCheckpoint('CDAE-model_2.h5', monitor='val_loss', mode='max',
                                                     save_best_only=True)]
                                )
            model.summary()

            #Plot trainning history
            #print(history.list)
            plt.subplot(321)
            plt.plot(history.history['val_loss'], label='validation')
            plt.plot(history.history['loss'], label='training')
            pyplot.legend()
            pyplot.xlabel('# epochs')
            pyplot.ylabel('loss')

            plt.subplot(322)
            plt.plot(history.history['val_prd'], label='validation')
            plt.plot(history.history['prd'], label='training')
            pyplot.legend()
            pyplot.xlabel('# epochs')
            pyplot.ylabel('prd')

            plt.subplot(323)
            plt.plot(history.history['val_prdn'], label='validation')
            plt.plot(history.history['prdn'], label='training')
            pyplot.legend()
            pyplot.xlabel('# epochs')
            pyplot.ylabel('prdn')

            plt.subplot(324)
            plt.plot(history.history['val_root_mean_squared_error'], label='validation')
            plt.plot(history.history['root_mean_squared_error'], label='training')
            pyplot.legend()
            pyplot.xlabel('# epochs')
            pyplot.ylabel('root_mean_squared_error')

            plt.subplot(325)
            plt.plot(history.history['val_snr'], label='validation')
            plt.plot(history.history['snr'], label='training')
            pyplot.legend()
            pyplot.xlabel('# epochs')
            pyplot.ylabel('snr')

            pyplot.savefig('CDAE-model_2_{}sec_training-history_5figur.svg'.format(time_frame))

            plt.show()

        #Evaluate model
        #saved_model = load_model('CDAE-model.h5')

        encoded_ecg = encoder.predict(X_noisy_test, verbose=2)
        #decoded_ecg = saved_model.predict(X_noisy_test,verbose=2)
        decoded_ecg = model.predict(X_noisy_test, verbose=2)

        #decoded_ecg = model.predict(decoded_ecg, verbose=2)
        for i in range(len(decoded_ecg)):
            decoded_ecg[i] = normalize_bound(decoded_ecg[i],lb=-1, ub=1)

        sr= tf.size(X_noisy_test,out_type=tf.dtypes.int32)
        sc= tf.size(encoded_ecg,out_type=tf.dtypes.int32)
        print('X_noisy_test tf size:',sr)
        print('encoded ecg tf size:', sc)
        print(sr/sc)

        serialization('reconstructed_ecg',decoded_ecg)

        decoded_ecg = deserialization('reconstructed_ecg')
        print(type(X_noisy_test))
        print(type(decoded_ecg))

        #show result
        #show

        id = np.random.randint(0, len(X_noisy_test))
        id=16
        print('id',id)
        plt.subplot(411)
        plt.plot(X_noisy_train[id])
        plt.title('Noisy ECG in train set_'+str(id))
        plt.subplot(412)
        plt.plot(X_original_train[id])
        plt.title('Original ECG in train set_'+str(id))
        plt.subplot(413)
        plt.plot(X_noisy_test[id])
        plt.title('Noisy ECG in test set_'+str(id))
        plt.subplot(414)
        plt.plot(X_original_test[id])
        plt.title('Original ECG in test set_'+str(id))
        manager = plt.get_current_fig_manager()
        manager.window.showMaximized()
        plt.savefig('CDAE-model_2_{}sec_data_{}.svg'.format(time_frame,id))
        plt.show()
        


        #cr[id] = X_noisy_test[id].itemsize() / encoded_ecg[id].itemsize()
        #print(cr[id])
        for i in range(len(X_noisy_test)):
            plt.subplot(311)
            plt.plot(X_noisy_test[i])
            plt.title('test noisy ecg_'+str(i))
            plt.subplot(312)
            plt.plot(decoded_ecg[i])
            plt.title('reconstruction ecg_'+ str(i))
            plt.subplot(313)
            plt.plot(X_original_test[i])
            plt.title('original ecg_'+str(i))
            manager = plt.get_current_fig_manager()
            manager.window.showMaximized()
            plt.show()
            #pyplot.savefig('CDAE-model_{}sec_result_figur.svg'.format(time_frame,i))
            pyplot.close()

if __name__ == '__main__':
    main(train=True)

