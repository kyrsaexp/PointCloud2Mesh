from keras.layers import Dense, Input
from keras.models import Model
from keras.optimizers import Adam
import numpy as np


def get_model():
    pts = Input(2)
    x = Dense(2, activation='linear')(pts)
    x = Dense(2, activation='sigmoid')(x)
    out = Dense(3, activation='sigmoid')(x)
    net = Model(pts, out)
    
    return net


def load_data_from_file(f_name, val_split=0.2):
    with open(f_name) as f:
        data_paths = f.read().splitlines()

        val_data_num = int(len(data_paths) * val_split)
        train_data_paths = data_paths[val_data_num:]
        val_data_paths = data_paths[:val_data_num]

        train_data = []
        val_data = []

        print 'DATA NUM:', len(data_paths)
        print 'TRAIN DATA:'

        for p in train_data_paths:
            print p
            train_data.extend(np.load(p))

        print 'VALIDATION DATA:'

        for p in val_data_paths:
            print p
            val_data.extend(np.load(p))

        train_data = np.array(train_data)
        val_data = np.array(val_data)

        return train_data, val_data


def train(src_f_name, tar_f_name, weights_path, lr = 0.001, loss='mse', b_size=10, epochs=100, val_split=0.2):
    """

    :param src_f_name: полный путь к текстовому файлу,в котором содержится список путей к 2D numpy файлам
    :param tar_f_name: полный путь к текстовому файлу,в котором содержится список путей к 3D numpy файлам
    пути к файлам должны идти в таком же порядке, как и в src_f_name
    :return:
    """

    # collect source data
    src_train_data, src_val_data = load_data_from_file(src_f_name, val_split)

    # collect target data
    tar_train_data, tar_val_data = load_data_from_file(tar_f_name, val_split)

    ###################
    # initialize model
    net = get_model()
    opt = Adam(lr=lr)
    net.compile(opt, loss)

    #################
    # train
    net.fit(src_train_data, tar_train_data, b_size, epochs, validation_data=(src_val_data, tar_val_data))
    net.save_weights(weights_path)


def predict(weights_path, test_f_name, save_to):
    # collect test data
    test_data, _ = load_data_from_file(test_f_name, 0)

    # load model
    net = get_model()
    net.load_weights(weights_path)
    
    # predict
    pred_data = net.predict(test_data)
    np.save(save_to, pred_data)
