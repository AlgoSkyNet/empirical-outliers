import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import shuffle

def _load_orig(ext):
    """Loads the raw data."""

    with open("../data/pendigits/pendigits-orig.{}".format(ext), 'r') as f:
        data_lines = f.readlines()

    data = []
    data_labels = []
    digit = None

    for line in data_lines:
        if line == "\n":
            continue

        if line[0] == ".":
            if "SEGMENT DIGIT" in line[1:]:
                if digit is not None:
                    data.append(np.array(digit))
                    data_labels.append(digit_label)

                digit = []
                digit_label = int(line.split('"')[1])
            else:
                continue

        else:
            x, y = map(float, line.split())
            digit.append([x, y])

    return data, data_labels

def normalise(row):
    path = row.reshape(-1, 2)
    path = MinMaxScaler().fit_transform(path)

    return path

def load_sampled(corpus_digits, outlier_digits):
    """Loads the sample pendigits dataset, where 8 points from
    each digits are selected."""

    data = np.loadtxt("../data/pendigits/pendigits.tes", delimiter=",")
    data = shuffle(data)

    corpus = data[np.isin(data[:, -1], corpus_digits)][:, :-1]
    corpus = np.array([normalise(row) for row in corpus])

    outliers = data[np.isin(data[:, -1], outlier_digits)][:, :-1]
    outliers = np.array([normalise(row) for row in outliers])

    return corpus, outliers



def load(corpus_digits, outlier_digits):
    """Loads the full, original dataset."""

    data1, data_labels1 = _load_orig("tra")
    data2, data_labels2 = _load_orig("tes")
    data = data1 + data2
    data_labels = data_labels1 + data_labels2

    data = [MinMaxScaler().fit_transform(path) for path in data]

    corpus = [p for p, label in zip(data, data_labels) if label in corpus_digits]
    outliers = [p for p, label in zip(data, data_labels) if label in outlier_digits]

    return corpus, outliers
