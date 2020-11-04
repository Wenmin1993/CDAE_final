import wfdb
import pickle
from wfdb import  rdsamp,rdann
from wfdb.processing import resample_singlechan
from scipy.signal import resample
import numpy as np

def loadData():
    numberSet = wfdb.get_record_list('afdb')
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
    RATIO = 0.3
    # 测试集及其标签集
    shuffle_index = np.random.permutation(len(X))
    test_length = int(RATIO * len(shuffle_index))  # RATIO = 0.3
    test_index = shuffle_index[:test_length]
    train_index = shuffle_index[test_length:]
    X_test, Y_test = X[test_index], Y[test_index]
    X_train, Y_train = X[train_index], Y[train_index]
    return X_train, Y_train, X_test, Y_test

def getDataSet(number, X_data, Y_data):
    ecgClassSet = ['(AFIB','(N', '(AFL',]

    # 读取心电数据记录
    print("正在读取 " + number + " 号心电数据...")
    record = wfdb.rdrecord('ecg_data/' + number, channel_names=['MLII'])
    data = record.p_signal.flatten()
    print(data)

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
            x_train = data[Rlocation[i] - 99:Rlocation[i] + 201]
            X_data.append(x_train)
            Y_data.append(lable)
            i += 1
        except ValueError:
            i += 1
    return

def serialization(name, content):
    f = open(r'pickle\s_' + name, 'wb')
    pickle.dump(content, f)
    f.close()
    return

signals = []
beat_locations = []
beat_types = []

record_files = wfdb.get_record_list('nsrdb')
print('record_files:', record_files)

for record in record_files:
    print('processing record: ', record)
    signal,field = (rdsamp(record, channels=None,pn_dir='nsrdb'))
    print(signal)
    print(signal.shape)
    length = len(signal)/2
    print('signal frequence:',field['fs'])
    annotation = rdann(record, 'atr', pn_dir='nsrdb',summarize_labels=True)
    print(type(annotation))
    #signal = resample(signal,len(signal)*250/120)
    signal, annotation = resample_singlechan(signal[:,0],annotation,fs=128,fs_target=250)

    beat_annotations = ['N', 'L', 'R', 'B', 'A',
                        'a', 'e', 'J', 'V', 'r',
                        'F', 'S', 'j', 'n', 'E',
                        '/', 'Q', 'f', '?', '(AFIB',
                        '(N', '(AFL', '+']
    indices = np.isin(annotation.symbol, beat_annotations)
    beat_type = np.asarray(annotation.symbol)[indices]
    beat_loc = annotation.sample[indices]

    signals.append(signal)
    beat_locations.append(beat_loc)
    beat_types.append(beat_type)
    # print('beat_types',beat_types)

serialization('data_nsr_250HZ_signal', signals)
serialization('data_nsr_250HZ_beat_loc', beat_locations)
serialization('data_nsr_250HZ_beat_types', beat_types)
