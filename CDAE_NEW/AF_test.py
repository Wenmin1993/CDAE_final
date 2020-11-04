import matplotlib
import wfdb
from wfdb import rdsamp, rdann
import numpy as np
from wfdb.processing import resample_singlechan
import pickle

def loadData():
    numberSet = ['100', '101', '103', '105', '106', '107', '108', '109', '111', '112', '113', '114', '115', '116', '117', '119', '121', '122', '123', '124', '200', '201', '202', '203', '205', '208', '210', '212', '213', '214', '215', '217', '219', '220', '221', '222', '223', '228', '230', '231', '232', '233', '234']
    dataSet = []
    lableSet = []
    for n in numberSet:
        getDataSet(n, dataSet, lableSet)

    # 转numpy数组,打乱顺序
    dataSet = np.array(dataSet).reshape(-1, 300)
    lableSet = np.array(lableSet).reshape(-1, 1)
    train_ds = np.hstack((dataSet, lableSet))
    np.random.shuffle(train_ds)

    # 数据集及其标签集
    X = train_ds[:, :300].reshape(-1, 300, 1)
    Y = train_ds[:, 300]

    # 测试集及其标签集
    shuffle_index = np.random.permutation(len(X))
    test_length = int(RATIO * len(shuffle_index)) # RATIO = 0.3
    test_index = shuffle_index[:test_length]
    train_index = shuffle_index[test_length:]
    X_test, Y_test = X[test_index], Y[test_index]
    X_train, Y_train = X[train_index], Y[train_index]
    return X_train, Y_train, X_test, Y_test

def getDataSet(number, X_data, Y_data):
    ecgClassSet = ['N', 'L', 'R', 'B', 'A',
                   'a', 'e', 'J', 'V', 'r',
                    'F', 'S', 'j', 'n', 'E',
                    '/', 'Q', 'f', '?', '(AFIB',
                    '(N', '(AFL', '+']

    # 读取心电数据记录
    print("正在读取 " + number + " 号心电数据...")
    record = wfdb.rdrecord('ecg_data/' + number, channel_names=['MLII'])
    data = record.p_signal.flatten()

    # 获取心电数据记录中R波的位置和对应的标签
    annotation = wfdb.rdann('ecg_data/' + number, 'atr')
    Rlocation = annotation.sample
    Rclass = annotation.symbol

    # 去掉前后的不稳定数据
    start = 10
    end = 5
    i = start
    j = len(annotation.symbol) - end

    # 因为只选择NAVLR五种心电类型,所以要选出该条记录中所需要的那些带有特定标签的数据,舍弃其余标签的点
    # X_data在R波前后截取长度为300的数据点
    # Y_data将NAVLR按顺序转换为01234
    while i < j:
        try:
            lable = ecgClassSet.index(Rclass[i])
            x_train = rdata[Rlocation[i] - 99:Rlocation[i] + 201]
            X_data.append(x_train)
            Y_data.append(lable)
            i += 1
        except ValueError:
            i += 1
    return


def data_from_af(db, channel):
    """
    Extract ECG, beat locations and beat types from Physionet database.
    Takes a list of record names, ECG channel index and name of the
    PhysioNet data base. Tested only with db == 'mitdb'.
    Parameters
    ----------
    channel : int
        ECG channel that is wanted from each record
    db : string
        Name of the PhysioNet ECG database
    Returns
    -------
    signals : list
        list of single channel ECG records stored as numpy arrays
    beat_locations : list
        list of numpy arrays where each array stores beat locations as
        samples from the beg of one resampled single channel
        ECG recording
    beat_types : list
        list of numpy arrays where each array stores the information of
        the beat types for the corresponding array in beat_locations
    """
    signals = []
    beat_locations = []
    beat_types = []
    useless_afrecord = ['00735', '03665', '04043', '08405', '08434']

    record_files = wfdb.get_record_list(db)
    print('record_files:', record_files)

    for record in record_files:
        if record in useless_afrecord:
            continue
        else:
            print('processing record: ', record)
            signal = (rdsamp(record, channels=None, pn_dir=db))
            signal_fs = signal[1]['fs']
            print('signal frequence:', signal_fs)
            # annotation= rdann(record, 'atr', pn_dir=db)
            annotation = rdann(record, 'atr', pn_dir=db)
            print(annotation.aux_note)

            signal, annotation = resample_singlechan(
                signal[0][:, channel],
                annotation,
                fs=signal_fs,
                fs_target=128)

            beat_annotations = ['N', 'L', 'R', 'B', 'A',
                                'a', 'e', 'J', 'V', 'r',
                                'F', 'S', 'j', 'n', 'E',
                                '/', 'Q', 'f', '?', '(AFIB',
                                '(N', '(AFL', '+']

            indices = np.isin(annotation.aux_note, beat_annotations)
            beat_type = np.asarray(annotation.aux_note)[indices]
            beat_loc = annotation.sample[indices]

            signals.append(signal)
            beat_locations.append(beat_loc)
            beat_types.append(beat_type)
    print('________data processed!_____')

    serialization(db + '_signal_test', signals)
    serialization(db + '_beat_loc_test', beat_locations)
    serialization(db + '_beat_types_test', beat_types)
    return signals, beat_locations, beat_types

'''
def data_from_af(db,channel):
    """
    Extract ECG, beat locations and beat types from Physionet database.
    Takes a list of record names, ECG channel index and name of the
    PhysioNet data base. Tested only with db == 'mitdb'.
    Parameters
    ----------
    channel : int
        ECG channel that is wanted from each record
    db : string
        Name of the PhysioNet ECG database
    Returns
    -------
    signals : list
        list of single channel ECG records stored as numpy arrays
    beat_locations : list
        list of numpy arrays where each array stores beat locations as
        samples from the beg of one resampled single channel
        ECG recording
    beat_types : list
        list of numpy arrays where each array stores the information of
        the beat types for the corresponding array in beat_locations
    """
    signals = []
    beat_locations = []
    beat_types = []
    useless_afrecord = ['00735', '03665', '04043', '08405', '08434']

    record_files = wfdb.get_record_list(db)
    print('record_files:', record_files)

    for record in record_files:
        if record in useless_afrecord:
            continue
        else:
            print('processing record: ', record)
            signal = (rdsamp(record, channels=None, pn_dir=db))
            signal_fs = signal[1]['fs']
            print('signal frequence:',signal_fs)
            #annotation= rdann(record, 'atr', pn_dir=db)
            annotation = rdann(record, 'atr', pn_dir=db)

            signal, annotation = resample_singlechan(
                signal[0][:, channel],
                annotation,
                fs=signal_fs,
                fs_target=128)

            beat_annotations = ['N', 'L', 'R', 'B', 'A',
                                'a', 'e', 'J', 'V', 'r',
                                'F', 'S', 'j', 'n', 'E',
                                '/', 'Q', 'f', '?', '(AFIB',
                                '(N', '(AFL', '+']

            indices = np.isin(annotation.auxnote, beat_annotations)
            beat_type = np.asarray(annotation.auxnote)[indices]
            beat_loc = annotation.sample[indices]

            signals.append(signal)
            beat_locations.append(beat_loc)
            beat_types.append(beat_type)
            print(annotation.aux_note)
    serialization(db + '_signal_test', signals)
    serialization(db + '_beat_loc_test', beat_locations)
    serialization(db + '_beat_types_test', beat_types)
    return signals, beat_locations, beat_types
'''
def get_fs():
    record_files = wfdb.get_record_list('afdb')
    print('record_files:', record_files)
    signal = rdsamp('04015', channels=None, pn_dir='afdb')
    signal_fs = signal[1]['fs']
    print(signal_fs)
    return signal_fs

def serialization(name, content):
    f = open(r'pickle\s_' + name, 'wb')
    pickle.dump(content, f)
    f.close()

def deserialization(name):
    f = open(r'pickle\s_' + name, 'rb')
    x = pickle.load(f)
    f.close()
    return x

#data_from_af('afdb',0)
#get_fs()
print(matplotlib.get_backend())