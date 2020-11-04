import wfdb
import numpy
import pickle
from scipy.signal import resample
from wfdb.processing import resample_singlechan
import sys
import matplotlib


def download_arrhythmia_db():
    db_name = 'afdb'
    wfdb.dl_database(db_name, dl_dir=db_name)

def extract_annotated_heartbeats(db_folder="afdb"):
    """
    :param db_folder: path to the folder the db was downloaded to
    :return: a pair (x, y) where x is the list of heart beats centered at each annotation y.
        The window size is calculated to be about 277 samples. Since the sample rate of
        the arrhythmia db is 360 hz, this corresponds to a window of ca. 0.79 seconds
        Each heart beat is stored as a numpy array.
    """
    useless_afrecord = ['00735', '03665', '04043', '08405', '08434']

    record_files = wfdb.get_record_list(db_folder)
    print(record_files)
    #data = wfdb.rdsamp(record_files[2],sampto= 250*60*60, pn_dir=db_folder)
    '''
    records = []
    annotations = []
    for record in record_files:
        if record in useless_afrecord:
            record_files.remove(record)
            print(record_files)
            continue
        else:
            print('processing-{}'.format(record))
            x = (wfdb.rdsamp(record,sampfrom=0,sampto=250*60*60,pn_dir=db_folder))
            print(type(x))
            annotation = wfdb.rdann(record, 'atr',pn_dir='afdb')
            print(type(annotation))
            signal, annotation = resample_singlechan(
                x[0][:, 0],
                annotation,
                fs=250,
                fs_target=128)
            records.append(signal)
            annotations.append(annotation)

    #data = wfdb.rdsamp(record_files[0], pn_dir=db_folder)
    #data = wfdb.rdsamp('{}/{}'.format(db_folder, record_files[0]))
    #records = [wfdb.rdsamp('{}/{}'.format(db_folder, record)) for record in record_files]

    print("number of records: ", len(record_files))
    print(numpy.shape(data[0]))
    #annotations = [wfdb.rdann('{}/{}'.format(db_folder, record), extension='atr') for record in record_files]

    serialization('record_11111',records)
    serialization('annotation_1111',annotations)
    '''
    records= deserialization('record_11111')
    annotations= deserialization('annotation_1111')
    print('annotation:',annotations[0].aux_note)
    print('record length',records[0][0])
    #print(type(records_1),type(annotations))
    #annotations = [wfdb.rdann('{}/{}'.format(db_folder, record), extension='atr') for record in record_files]

    '''
    #--------resample----------
    for i in range(len(annotations)):
        signal, label = resample_singlechan(records_1[i][0], annotations_1[i], fs=250, fs_target=128)
        records.append(signal)
        annotations.append(label)
    '''
    '''
    for record,annotation in zip(records_1,annotations_1):
        print(record[0].shape,type(record[0]))
        print(annotation,type(annotation))
        signal,label =resample_singlechan(record[0],annotation,fs =250,fs_target=128)
        records.append(signal)
        annotations.append(label)
    '''
    print('annotation:',len(annotations))

    #records,annotations = resample_multichan(records,annotations,fs =250,fs_target=128,)
    '''''
    sample_distances = [y - x
                        for annotation in annotations
                        for x, y in zip(annotation.sample[0:-2], annotation.sample[1:-1])]
    neighbourhood_size_in_samples = int(numpy.mean(sample_distances) / 2)
    print('ns:',neighbourhood_size_in_samples)
    '''''
    neighbourhood_size_in_samples = 640
    #print("neighbourhood size ", neighbourhood_size_in_samples)

    samples_per_record = 250 * 60 * 60


    def get_section_from_record(record, sample, neighbourhood_size):
        return record[sample - neighbourhood_size:sample + neighbourhood_size, :]

    def neighbourhood_is_sufficient(sample):
        return neighbourhood_size_in_samples < sample < samples_per_record - neighbourhood_size_in_samples

    def annotated_samples_with_sufficient_neighbourhood(annotation):
        yield from filter(lambda annotated_sample: neighbourhood_is_sufficient(annotated_sample[1]),
                          zip(annotation.aux_note, annotation.sample))

    print("sectioning samples")
    sectioned_labeled_samples = [(symbol, get_section_from_record(record, sample, neighbourhood_size_in_samples))
                                 for record, annotation in zip(records, annotations)
                                 for symbol, sample in annotated_samples_with_sufficient_neighbourhood(annotation)]

    #relevant_labels = ['N', 'L', 'R', 'A', 'V', '/',]
    relevant_labels = ['N', 'L', 'R', 'B', 'A',
                        'a', 'e', 'J', 'V', 'r',
                        'F', 'S', 'j', 'n', 'E',
                        '/', 'Q', 'f', '?', '(AFIB',
                        '(N', '(AFL', '+']
    
    relevant_labeled_samples = list(filter(lambda labeled_sample: labeled_sample[0] in relevant_labels,
                                      sectioned_labeled_samples))
    
    samples = [sample for _, sample in relevant_labeled_samples]
    labels = [symbol for symbol, _ in relevant_labeled_samples]

    print(labels)
    return samples, labels

def serialization(name, content):
    f = open(r'pickle\s_' + name, 'wb')
    pickle.dump(content, f)
    f.close()

def deserialization(name):
    f = open(r'pickle\s_' + name, 'rb')
    x = pickle.load(f)
    f.close()
    return x

if __name__ == "__main__":
    #download_arrhythmia_db()
    heartbeats = extract_annotated_heartbeats()
    print('signal shape:',heartbeats[0][0].shape)
    print(len(heartbeats[0]))
    print(heartbeats[1])
    serialization('new_afdb',heartbeats)
    '''
        with open(sys.argv[1], 'wb') as f:
        pickle.dump(heartbeats, f)
    '''

