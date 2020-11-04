import pickle
import numpy as np
from scipy.signal import resample
import wfdb
import matplotlib.pyplot as plt
from wfdb import rdann,rdsamp
from wfdb.processing import resample_singlechan, find_local_peaks, correct_peaks, normalize_bound, resample_sig
from random import uniform
np.set_printoptions(threshold=np.inf)

def serialization(name, content):
    f = open(r'pickle\s_' + name, 'wb')
    pickle.dump(content, f)
    f.close()
    return
def deserialization(name):
    f = open(r'pickle\s_' + name, 'rb')
    x = pickle.load(f)
    f.close()
    return x

def get_beats(annotation,db):
    """
    Extract beat indices and types of the beats.
    Beat indices indicate location of the beat as samples from the
    beg of the signal. Beat types are standard character
    annotations used by the PhysioNet.
    Parameters
    ----------
    annotation : wfdbdb.io.annotation.Annotation
        wfdbdb annotation object
    Returns
    -------
    beats : array
        beat locations (samples from the beg of signal)
    symbols : array
        beat symbols (types of beats)
    """
    # All beat annotations
    beat_annotations = ['N', 'L', 'R', 'B', 'A',
                        'a', 'e', 'J', 'V', 'r',
                        'F', 'S', 'j', 'n', 'E',
                        '/', 'Q', 'f', '?', '(AFIB',
                        '(N','(AFL','+']
    if db == 'afdb':
        indices = np.isin(annotation.auxnote, beat_annotations)
        symbols = np.asarray(annotation.auxnote)[indices]
        beats = annotation.sample[indices]
    else:
        # Get indices and symbols of the beat annotations
        indices = np.isin(annotation.symbol, beat_annotations)
        symbols = np.asarray(annotation.symbol)[indices]
        beats = annotation.sample[indices]
    return beats, symbols

def data_from_nsr(db,channel):
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

    record_files = wfdb.get_record_list(db)
    print('record_files:', record_files)

    for record in record_files:
        print('processing record: ', record)
        signal = rdsamp(record, channels=None, pn_dir=db)
        signal_fs = signal[1]['fs']
        print('signal frequence:',signal_fs)
        annotation = rdann(record, 'atr', pn_dir=db,summarize_labels=True)
        print(type(annotation))
        '''
        signal, annotation = resample_singlechan(
                                signal[0][:, channel],
                                annotation,
                                fs=signal_fs,
                                fs_target=250)
        #resample(signal,)
        '''
        '''
            if db == 'afdb':
                beat_loc, beat_type = get_beats(annotation,db)
            else:
                beat_loc, beat_type = get_beats(annotation,db)
            '''
        signal, annotation = resample_singlechan(signal[0][:, channel], annotation, fs=signal_fs, fs_target=250)

        beat_annotations = ['N', 'L', 'R', 'B', 'A',
                                'a', 'e', 'J', 'V', 'r',
                                'F', 'S', 'j', 'n', 'E',
                                '/', 'Q', 'f', '?', '(AFIB',
                                '(N','(AFL','+']

        indices = np.isin(annotation.symbol, beat_annotations)
        beat_type = np.asarray(annotation.symbol)[indices]
        beat_loc = annotation.sample[indices]

        signals.append(signal)
        beat_locations.append(beat_loc)
        beat_types.append(beat_type)
        #print('beat_types',beat_types)
    return signals, beat_locations, beat_types

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
            signal = rdsamp(record, channels=None, pn_dir=db)
            signal_fs = signal[1]['fs']
            print('signal frequence:', signal_fs)
            # annotation= rdann(record, 'atr', pn_dir=db)
            annotation = rdann(record, 'atr', pn_dir=db)

            signal, annotation = resample_singlechan(signal[0][:, channel],annotation,fs=signal_fs,fs_target=250)

            beat_annotations = ['N', 'L', 'R', 'B', 'A',
                                'a', 'e', 'J', 'V', 'r',
                                'F', 'S', 'j', 'n', 'E',
                                '/', 'Q', 'f', '?', '(AFIB',
                                '(N','(AFL','+']

            indices = np.isin(annotation.aux_note, beat_annotations)
            #print('indices:',indices)
            beat_type = np.asarray(annotation.aux_note)[indices]
            #print('beat_type',beat_type)
            beat_loc = annotation.sample[indices]
            #print('beat_loc',beat_loc)

            signals.append(signal)
            beat_locations.append(beat_loc)
            beat_types.append(beat_type)

    serialization(db + '_signal_128hz', signals)
    serialization(db + '_beat_loc_128hz', beat_locations)
    serialization(db + '_beat_types_128hz', beat_types)
    return signals, beat_locations, beat_types

def load_data(database):
    print('-----loading----- ')
    signals = deserialization(database + '_signal_128hz')
    beat_locations = deserialization(database + '_beat_loc_128hz')
    beat_types = deserialization(database + '_beat_types_128hz')

    print('singnals.shape:', np.asarray(signals).shape)
    print('-------ecg record from ' + database + 'loaded!------')
    return signals,beat_locations,beat_types

def fix_labels(signals, beats, labels):
    """
    Change labeling of the normal beats.
    Beat index of some normal beats doesn't occur at the local maxima
    of the ECG signal in MIT-BIH Arrhytmia database. Function checks if
    beat index occurs within 5 samples from the local maxima. If this is
    not true, beat labeling is changed to -1.
    Parameters
    ----------
    signals : list
        List of ECG signals as numpy arrays
    beats : list
        List of numpy arrays that store beat locations
    labels : list
        List of numpy arrays that store beat types
    Returns
    -------
    fixed_labels : list
        List of numpy arrays where -1 has been added for beats that are
        not located in local maxima
    """
    print('----fix label start----')
    fixed_labels = []
    for s, b, l in zip(signals, beats, labels):

        # Find local maximas
        localmax = find_local_peaks(sig=s, radius=5)
        localmax = correct_peaks(sig=s,
                                 peak_inds=localmax,
                                 search_radius=5,
                                 smooth_window_size=20,
                                 peak_dir='up')

        # Make sure that beat is also in local maxima
        fixed_p = correct_peaks(sig=s,
                                peak_inds=b,
                                search_radius=5,
                                smooth_window_size=20,
                                peak_dir='up')

        # Check what beats are in local maximas
        beat_is_local_peak = np.isin(fixed_p, localmax)
        '''
        if l == 'N':
            fixed_l = 1
        else:
            fixed_l = l
        '''

        fixed_l = l
        # Add -1 if beat is not in local max
        fixed_l[~beat_is_local_peak] = -1
        fixed_labels.append(fixed_l)
    print('----fix label finished----')
    return fixed_labels

def get_white_Gaussian_Noise(noise_type, signal, snr):
    snr = 10 ** (snr / 10.0)
    print(len(signal))
    power_signal = np.sum(signal ** 2) / len(signal)
    power_noise = power_signal / snr
    wgn = np.random.randn(len(signal)) * np.sqrt(power_noise)
    serialization(noise_type + '_wgn_noise', wgn)
    return wgn

def load_wgn_noise(noise_type):
    wgn = deserialization(noise_type + '_wgn_noise')
    print(noise_type + '_wgn.shape:', wgn.shape)
    return wgn

def create_sine(sampling_frequency, time_s, sine_frequency):
    """
    Create sine wave.
    Function creates sine wave of wanted frequency and duration on a
    given sampling frequency.
    Parameters
    ----------
    sampling_frequency : float
        Sampling frequency used to sample the sine wave
    time_s : float
        Lenght of sine wave in seconds
    sine_frequency : float
        Frequency of sine wave
    Returns
    -------
    sine : array
        Sine wave
    """
    samples = np.arange(time_s * sampling_frequency) / sampling_frequency
    sine = np.sin(2 * np.pi * sine_frequency * samples)

    return sine

def creat_noise(ma, bw, wgn , win_size):
    """
    Create noise that is typical in ambulatory ECG recordings.
    Creates win_size of noise by using muscle artifact, baseline
    wander, and mains interefence (60 Hz sine wave) noise. Windows from
    both ma and bw are randomly selected to
    maximize different noise combinations. Selected noise windows from
    all of the sources are multiplied by different random numbers to
    give variation to noise strengths. Mains interefence is always added
    to signal, while addition of other two noise sources varies.
    Parameters
    ----------
    ma : array
        Muscle artifact signal
    bw : array
        Baseline wander signal
    wgn: array
        White gaussian noise
    win_size : int
        Wanted noise length
    Returns
    -------
    noise : array
        Noise signal of given window size
    """
    # Get the slice of data
    beg = np.random.randint(ma.shape[0] - win_size)
    end = beg + win_size
    beg2 = np.random.randint(bw.shape[0] - win_size)
    end2 = beg2 + win_size
    beg3 = np.random.randint(wgn.shape[0] - win_size)
    end3 = beg3 + win_size

    # Get mains_frequency US 50 Hz (alter strength by multiplying)
    mains = create_sine(250, int(win_size/250), 50)*uniform(0, 0.5)

    # Choose what noise to add
    mode = np.random.randint(7)

    # Add noise with different strengths
    ma_multip = uniform(0, 1)#1
    bw_multip = uniform(0, 3)#3
    wgn_multip = uniform(0, 0.5)

    # Add noise
    if mode == 0:
        noise = ma[beg:end] * ma_multip
    elif mode == 1:
        noise = bw[beg2:end2] * bw_multip
    elif mode == 2:
        noise = wgn[beg3:end3] * wgn_multip
    elif mode == 3:
        noise = ma[beg:end] * ma_multip + wgn[beg3:end3] * wgn_multip
    elif mode == 4:
        noise = bw[beg2:end2] * bw_multip + wgn[beg3:end3] * wgn_multip
    elif mode == 5:
        noise = ma[beg:end] * ma_multip + bw[beg2:end2] * bw_multip
    else:
        noise = ma[beg:end] * ma_multip + bw[beg2:end2] * bw_multip + wgn[beg3:end3] * wgn_multip

    print('noise.shape',noise.shape)

    return noise + mains
    #return noise

def get_noise_records(noise_type,database):
    print('processing noise:', noise_type)
    s, f = wfdb.rdsamp(noise_type, channels=[0], pn_dir=database)
    print(s)
    print(f)
    signal, _ = resample_sig(s[:, 0], fs=360, fs_target=250)
    print(signal)
    signal_size = f['sig_len']
    print('-----noise processed!-----')

    # serialization the data
    serialization(database + '_' + noise_type + '_signal_128hz', signal)
    serialization(database + '_' + noise_type + '_field_128hz', f)
    serialization(database + '_' + noise_type + '_size_128hz', signal_size)

    #return signal
    return signal, f, signal_size

def load_noise_signal(database, noise_type):
    # deserialization the data
    signal = deserialization(database + '_' + noise_type + '_signal_128hz')
    field = deserialization(database + '_' + noise_type + '_field_128hz')
    signal_size = deserialization(database + '_' + noise_type + '_size_128hz')
    print(noise_type + ' singnals.shape:', np.asarray(signal).shape)
    print('-------' + noise_type + ' noise signal loaded!------')
    #return signal, field, signal_size
    return signal,field,signal_size

'''
def ecg_generator(db,ma,bw,wgn):
    """
    :param db_folder: path to the folder the db was downloaded to
    :return: a pair (x, y) where x is the list of heart beats centered at each annotation y.
        The window size is calculated to be about 277 samples. Since the sample rate of
        the arrhythmia db is 360 hz, this corresponds to a window of ca. 0.79 seconds
        Each heart beat is stored as a numpy array.
    """
    record_files = wfdb.get_record_list(db)
    data = wfdb.rdsamp('{}/{}'.format(db, record_files[0]))
    records = [wfdb.rdsamp('{}/{}'.format(db, record)) for record in record_files]

    print("number of records: ", len(record_files))
    print(np.shape(data[0]))

    annotations = [wfdb.rdann('{}/{}'.format(db, record), extension='atr') for record in record_files]

    sample_distances = [y - x
                        for annotation in annotations
                        for x, y in zip(annotation.sample[0:-2], annotation.sample[1:-1])]
    print(sample_distances)
    neighbourhood_size_in_samples = int(np.mean(sample_distances) / 2)

    print("neighbourhood size ", neighbourhood_size_in_samples)

    samples_per_record = 360 * 60 * 30

    def get_section_from_record(record, sample, neighbourhood_size):
        return record[0][sample - neighbourhood_size:sample + neighbourhood_size, :]

    def neighbourhood_is_sufficient(sample):
        return neighbourhood_size_in_samples < sample < samples_per_record - neighbourhood_size_in_samples

    def annotated_samples_with_sufficient_neighbourhood(annotation):
        yield from filter(lambda annotated_sample: neighbourhood_is_sufficient(annotated_sample[1]),
                          zip(annotation.symbol, annotation.sample))

    print("sectioning samples")
    sectioned_labeled_samples = [(symbol, get_section_from_record(record, sample, neighbourhood_size_in_samples))
                                 for record, annotation in zip(records, annotations)
                                 for symbol, sample in annotated_samples_with_sufficient_neighbourhood(annotation)]

    relevant_labels = ['N', 'L', 'R', 'A', 'V', '/','AFIB','AFL','J']

    relevant_labeled_samples = filter(lambda labeled_sample: labeled_sample[0] in relevant_labels,
                                      sectioned_labeled_samples)
    samples = [sample for _, sample in relevant_labeled_samples]
    labels = [symbol for symbol, _ in relevant_labeled_samples]
    return samples, labels
'''
'''
def ecg_generator(name, signals, wgn, ma, bw, win_size, batch_size):
    print('processing')
    while True:
        x = []
        section = []
        noise_list = []
        while len(x) < batch_size:
            random_sig_idx = np.random.randint(0, len(signals))
            random_sig = signals[random_sig_idx]
            #print(random_sig_idx)

            # Select one window
            beg = np.random.randint(random_sig.shape[0] - win_size)
            end = beg + win_size
            #section = normalize_bound(section, lb=-1, ub=1)

            # Select data for window and normalize it (-1, 1)
            data_win = normalize_bound(random_sig[beg:end],lb=-1, ub=1)
            section.append(random_sig[beg:end])
            # Add noise into data window and normalize it again
            added_noise = creat_noise(wgn,ma,bw,win_size)
            added_noise = normalize_bound(added_noise,lb=-1, ub=1)
            noise_list.append(added_noise)
            data_win = data_win + added_noise
            data_win = normalize_bound(data_win, lb=-1, ub=1)
            x.append(data_win)

        section = np.asarray(section)
        section = section.reshape(section.shape[0],section.shape[1],1)
        #print(section, section.shape, type(section))
        x = np.asarray(x)
        print(x.shape)
        x = x.reshape(x.shape[0], x.shape[1],1)
        #print(x)

        id = np.random.randint(0,len(x))
        plt.subplot(311)
        plt.plot(section[id])
        plt.title('original '+name+' ecg')
        plt.subplot(312)
        plt.plot(x[id])
        plt.title('noised '+name+' ecg')
        plt.subplot(313)
        plt.title('added noise')
        plt.plot(noise_list[id])
        #plt.savefig(name+'.png')
        plt.show()

        print('shape:',x.shape)
        serialization(name+'_original_ecg',section)
        serialization(name+'_noised_ecg',x)
        return x,section
'''
def ecg_generator(name, signals, peaks, labels, wgn,ma, bw, win_size, batch_size):
    """
    Generate ECG data with R-peak labels.
    Data generator that yields training data as batches. Every instance
    of training batch is composed as follows:
    1. Randomly select one ECG signal from given list of ECG signals
    2. Randomly select one window of given win_size from selected signal
    3. Check that window has at least one beat and that all beats are
       labled as normal
    4. Create label window corresponding the selected window
        -beats and four samples next to beats are labeled as 1 while
         rest of the samples are labeled as 0
    5. Normalize selected signal window from -1 to 1
    6. Add noise into signal window and normalize it again to (-1, 1)
    7. Add noisy signal and its labels to trainig batch
    8. Transform training batches to arrays of needed shape and yield
       training batch with corresponding labels when needed
    Parameters
    ----------
    signals : list
        List of ECG signals
    peaks : list
        List of peaks locations for the ECG signals
    labels : list
        List of labels (peak types) for the peaks
    ma : array
        Muscle artifact signal
    bw : array
        Baseline wander signal
    win_size : int
        Number of time steps in the training window
    batch_size : int
        Number of training examples in the batch
    Yields
    ------
    (X, y) : tuple
        Contains training samples with corresponding labels
    """
    while True:
        print('generating!')
        X = []
        Original = []
        Noise = []
        y = []
        print(signals[0].shape)
        while len(X) < batch_size:
            random_sig_idx = np.random.randint(0, len(signals))
            random_sig = signals[random_sig_idx]
            p4sig = peaks[random_sig_idx]
            plabels = labels[random_sig_idx]

            # Select one window
            beg = np.random.randint(random_sig.shape[0]-win_size)
            end = beg + win_size

            # Select peaks that fall into selected window.
            # Buffer of 3 to the window edge is needed as labels are
            # inserted also next to point)
            p_in_win = p4sig[(p4sig >= beg+3) & (p4sig <= end-3)]-beg
            #print('p in win',p_in_win)
            # Check that there is at least one peak in the window
            if p_in_win.shape[0] >= 1:

                # Select labels that fall into selected window
                lab_in_win = plabels[(p4sig >= beg + 3) & (p4sig <= end - 3)]
                # print(type(lab_in_win),lab_in_win)
                b = 0
                l = 0
                if ['(AFIB'] in lab_in_win: b = 1
                if ['(AFL'] in lab_in_win: l = 1

                if b == 1 or l == 1:
                    window_label = 1
                else:
                    window_label = 0
                '''
                # Select labels that fall into selected window
                lab_in_win = plabels[(p4sig >= beg+3) & (p4sig <= end-3)]
                #print('lab',lab_in_win)

                try:
                    # Create labels for data window
                    window_labels = np.zeros(win_size,object)
                    #window_labels.astype(str)
                    #print('1', window_labels)
                    np.put(window_labels, p_in_win, lab_in_win)
                    #print('2', window_labels[p_in_win])

                    # Put labels also next to peak
                    np.put(window_labels, p_in_win+1, lab_in_win)
                    #print('2', window_labels[p_in_win+1])
                    np.put(window_labels, p_in_win+2, lab_in_win)
                    np.put(window_labels, p_in_win-1, lab_in_win)
                    np.put(window_labels, p_in_win-2, lab_in_win)
                except ValueError as e:
                    print('error',e)
                '''

                #print('3',window_labels)
                #print('4',window_labels[p_in_win])

                # Select data for window and normalize it (-1, 1)
                data_win = normalize_bound(random_sig[beg:end],lb=-1, ub=1)
                print('1',data_win.shape)
                Original.append(data_win)
                n = creat_noise(ma, bw, wgn, win_size)
                print('3',n.shape)
                Noise.append(n)
                # Add noise into data window and normalize it again
                data_win = data_win + n
                print('2',data_win.shape)
                data_win = normalize_bound(data_win, lb=-1, ub=1)

                X.append(data_win)

                y.append(window_label)

        X = np.asarray(X)
        y = np.asarray(y)
        Original = np.asarray(Original)
        X = X.reshape(X.shape[0],X.shape[1],1)
        Original = Original.reshape(Original.shape[0], Original.shape[1], 1)
        y = y.reshape(y.shape[0]).astype(int)

        #Output the graph of original-,noised- ecg signal and added noise
        id = np.random.randint(0, len(X))
        plt.subplot(311)
        plt.plot(Original[id])
        plt.title('original ' + name + ' ecg')
        plt.subplot(312)
        plt.plot(X[id])
        plt.title('noised ' + name + ' ecg')
        plt.subplot(313)
        plt.title('added noise')
        plt.plot(Noise[id])
        # plt.savefig(name+'.png')
        plt.show()
        #print(y)
        print(X.shape)
        print(Original.shape)
        #print(y.shape)
        #yield (X, y)
        return X,y,Original


def ecg_clean_generator(name, signals, peaks, labels,win_size, batch_size):
    """
    Generate ECG data with R-peak labels.
    Data generator that yields training data as batches. Every instance
    of training batch is composed as follows:
    1. Randomly select one ECG signal from given list of ECG signals
    2. Randomly select one window of given win_size from selected signal
    3. Normalize selected signal window from -1 to 1
    4. Transform training batches to arrays of needed shape and yield
       training batch with corresponding labels when needed
    Parameters
    ----------
    signals : list
        List of ECG signals
    peaks : list
        List of peaks locations for the ECG signals
    labels : list
        List of labels (peak types) for the peaks
    ma : array
        Muscle artifact signal
    bw : array
        Baseline wander signal
    win_size : int
        Number of time steps in the training window
    batch_size : int
        Number of training examples in the batch
    Yields
    ------
    (X, y) : tuple
        Contains training samples with corresponding labels
    """
    while True:
        X = []
        y = []
        while len(X) < batch_size:
            random_sig_idx = np.random.randint(0, len(signals))
            random_sig = signals[random_sig_idx]
            p4sig = peaks[random_sig_idx]
            #print('p4sig:',p4sig)
            plabels = labels[random_sig_idx]

            # Select one window
            beg = np.random.randint(random_sig.shape[0]-win_size)
            end = beg + win_size

            # Select peaks that fall into selected window.
            # Buffer of 3 to the window edge is needed as labels are
            # inserted also next to point)
            p_in_win = p4sig[(p4sig >= beg+3) & (p4sig <= end-3)]-beg
            #print('p in win',p_in_win)
            # Check that there is at least one peak in the window
            if p_in_win.shape[0] >= 1:

                # Select labels that fall into selected window
                lab_in_win = plabels[(p4sig >= beg+3) & (p4sig <= end-3)]
                #print(type(lab_in_win),lab_in_win)
                b = 0
                l = 0
                if ['(AFIB'] in lab_in_win: b=1
                if ['(AFL'] in lab_in_win: l=1

                if b==1 or l==1 :
                    window_label = 1
                else:
                    window_label = 0

                '''
                if ['(AFIB'] in lab_in_win : lab_in_win = [1]
                if ['(AFL'] in lab_in_win: lab_in_win = [1]
                if ['(N'] in lab_in_win : lab_in_win = [2]

                #print('lab',lab_in_win)
                try:
                    # Create labels for data window
                    #window_labels = np.zeros(win_size, object)
                    window_labels = np.zeros(win_size)
                    #print('1', window_labels)
                    np.put(window_labels, p_in_win, lab_in_win)
                    print('2', window_labels[p_in_win])

                    # Put labels also next to peak
                    np.put(window_labels, p_in_win+1, lab_in_win)
                    #print('2', window_labels[p_in_win+1])
                    np.put(window_labels, p_in_win+2, lab_in_win)
                    np.put(window_labels, p_in_win-1, lab_in_win)
                    np.put(window_labels, p_in_win-2, lab_in_win)
                except ValueError as e:
                    print('error',e)
                '''
                #print('3',window_labels)
                #print('4',window_labels[p_in_win])

                # Select data for window and normalize it (-1, 1)
                #data_win =random_sig[beg:end]
                data_win = normalize_bound(random_sig[beg:end],lb=-1, ub=1)
                X.append(data_win)
                y.append(window_label)

        X = np.asarray(X)
        y = np.asarray(y)
        print(type(y))
        X = X.reshape(X.shape[0],X.shape[1],1)
        y = y.reshape(y.shape[0]).astype(int)
        '''
        #Output the graph of original-,noised- ecg signal and added noise
        id = np.random.randint(0, len(X))
        plt.plot(X[id])
        plt.title(name + ' ecg')
        plt.show()
        print(X.shape)
        print(y.shape)
        '''
        #yield (X, y)
        return X,y

def ecg_noisy_generator(name, signals, peaks,ma, bw, wgn, labels,win_size, batch_size):

    while True:
        X = []
        y = []
        while len(X) < batch_size:
            random_sig_idx = np.random.randint(0, len(signals))
            random_sig = signals[random_sig_idx]
            p4sig = peaks[random_sig_idx]
            #print('p4sig:',p4sig)
            plabels = labels[random_sig_idx]

            # Select one window
            beg = np.random.randint(random_sig.shape[0]-win_size)
            end = beg + win_size

            # Select peaks that fall into selected window.
            # Buffer of 3 to the window edge is needed as labels are
            # inserted also next to point)
            p_in_win = p4sig[(p4sig >= beg+3) & (p4sig <= end-3)]-beg
            #print('p in win',p_in_win)
            # Check that there is at least one peak in the window
            if p_in_win.shape[0] >= 1:

                # Select labels that fall into selected window
                lab_in_win = plabels[(p4sig >= beg+3) & (p4sig <= end-3)]
                #print(type(lab_in_win),lab_in_win)
                b = 0
                l = 0
                if ['(AFIB'] in lab_in_win: b=1
                if ['(AFL'] in lab_in_win: l=1

                if b==1 or l==1 :
                    window_label = 1
                else:
                    window_label = 0

                '''
                if ['(AFIB'] in lab_in_win : lab_in_win = [1]
                if ['(AFL'] in lab_in_win: lab_in_win = [1]
                if ['(N'] in lab_in_win : lab_in_win = [2]

                #print('lab',lab_in_win)
                try:
                    # Create labels for data window
                    #window_labels = np.zeros(win_size, object)
                    window_labels = np.zeros(win_size)
                    #print('1', window_labels)
                    np.put(window_labels, p_in_win, lab_in_win)
                    print('2', window_labels[p_in_win])

                    # Put labels also next to peak
                    np.put(window_labels, p_in_win+1, lab_in_win)
                    #print('2', window_labels[p_in_win+1])
                    np.put(window_labels, p_in_win+2, lab_in_win)
                    np.put(window_labels, p_in_win-1, lab_in_win)
                    np.put(window_labels, p_in_win-2, lab_in_win)
                except ValueError as e:
                    print('error',e)
                '''
                #print('3',window_labels)
                #print('4',window_labels[p_in_win])

                # Select data for window and normalize it (-1, 1)
                #data_win =random_sig[beg:end]
                data_win = normalize_bound(random_sig[beg:end],lb=-1, ub=1)
                n = creat_noise(ma, bw, wgn, win_size)
                print('2',n.shape)

                # Add noise into data window and normalize it again
                data_win = data_win + n
                print(data_win.shape)
                data_win = normalize_bound(data_win, lb=-1, ub=1)

                X.append(data_win)
                y.append(window_label)

        X = np.asarray(X)
        y = np.asarray(y)
        print(type(y))
        X = X.reshape(X.shape[0],X.shape[1],1)
        y = y.reshape(y.shape[0]).astype(int)
        '''
        #Output the graph of original-,noised- ecg signal and added noise
        id = np.random.randint(0, len(X))
        plt.plot(X[id])
        plt.title(name + ' ecg')
        plt.show()
        print(X.shape)
        print(y.shape)
        '''
        #yield (X, y)
        return X,y

'''
#serialization the records and ma,bw,wgn noise
nsr,nsr_bls,nsr_labels =data_from_nsr('nsrdb', 0)#---已存
serialization('nsr_signal_250hz', nsr)
serialization('nsr_beat_loc_250hz', nsr_bls)
serialization('nsr_beat_types_250hz', nsr_labels)
'''
#af,af_bls,af_labels = data_from_af('afdb',0)#---已存


#deserialization the records and noise
nsr,nsr_bls,nsr_labels =load_data('nsr')
af,af_bls,af_labels = load_data('afdb')
#print(nsr_labels)
#print(af_labels)

'''
ma,ma_field,ma_size = get_noise_records('ma','nstdb')#---已存
bw,bw_field,bw_size = get_noise_records('bw','nstdb')#---已存
id_nsr = np.random.randint(len(nsr))
id_af = np.random.randint(len(af))
wgn_nsr = get_white_Gaussian_Noise('nsr',nsr[id_nsr],0.5)#---已存
wgn_af = get_white_Gaussian_Noise('af',af[id_af],0.5)#---已存
'''
'''
#fix label and serialization
nsr_labels = fix_labels(nsr, nsr_bls, nsr_labels)
af_labels = fix_labels(af, af_bls, af_labels)
serialization('nsrdb' + '_beat_types_fixed', nsr_labels)
serialization('afdb' + '_beat_types_fixed', af_labels)

#af_labels = fix_labels(af, af_bls, af_labels)
#deserialization label and noise
#nsr_labels = deserialization('nsrdb_beat_types_fixed')
#af_labels = deserialization('afdb_beat_types_fixed')
'''

ma,ma_field,ma_size = load_noise_signal('nstdb','ma')
bw,bw_field,bw_size = load_noise_signal('nstdb','bw')
wgn_nsr = load_wgn_noise('nsr')
wgn_af = load_wgn_noise('af')

#print(nsr_labels)
#creat data for reference_model
#generate clean ecg signals and serialization
'''
afc, afc_label = ecg_clean_generator('af',af,af_bls,af_labels,win_size=3000,batch_size=10000)
nsrc, nsrc_label = ecg_clean_generator('nsr',nsr,nsr_bls,nsr_labels,win_size=3000,batch_size=10000)
serialization('afdb_clean_ecg_250hz',afc)
serialization('afdb_clean_label_250hz',afc_label)
serialization('nsrdb_clean_ecg_250hz',nsrc)
serialization('nsrdb_clean_label_250hz',afc_label)
#print('clean ecg generated!')
'''

#generate noise ecg signals and serialization
nsr,nsr_label,nsr_original = ecg_generator('nsr',nsr,nsr_bls,nsr_labels,wgn_nsr,ma,bw,win_size=1000,batch_size=1000)
af,af_label,af_original = ecg_generator('af',af,af_bls,af_labels,wgn_af,ma,bw,win_size=1000,batch_size=1000)#---已存
serialization('noisy_nsr_ecg_250hz_4sec_1000size',nsr)
serialization('nsr_label_ecg_250hz_4sec_1000size',nsr_label)
serialization('original_nsr_ecg_250hz_4sec_1000size',nsr_original)

serialization('noisy_af_ecg_250hz_4sec_1000size',af) #---已存
serialization('af_label_ecg_250hz_4sec_1000size',af_label)#---已存
serialization('original_af_ecg_250hz_4sec_1000size',af_original)#---已存
print('noisy ecg generated!')

#generate noisy af ecg
'''''
#add single type noise mode =0
nsr,nsr_label,nsr_original = ecg_generator('nsr',nsr,nsr_bls,nsr_labels,wgn_nsr,ma,bw,win_size=1280,batch_size=256)
af,af_label,af_original = ecg_generator('af',af,af_bls,af_labels,wgn_af,ma,bw,win_size=1280,batch_size=256)
serialization('noisy_nsr_ecg_mode0',nsr)
serialization('nsr_label_ecg_mode0',nsr_label)
serialization('original_nsr_ecg_mode0',nsr_original)
serialization('noisy_af_ecg_mode0',af)
serialization('af_label_ecg_mode0',af_label)
serialization('original_af_ecg_mode0',af_original)
'''''
'''
afn,afn_label = ecg_noisy_generator('af',af,af_bls,af_labels,wgn_af,ma,bw,win_size=1280*30,batch_size=10000)
serialization('noisy_afn_ecg_128hz_30_30sec',afn)
serialization('noisy_afn_label_128hz_30sec',afn_label)
'''