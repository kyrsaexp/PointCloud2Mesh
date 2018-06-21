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


def train(src_f_name, tar_f_name, model_dir, lr=1e-2, loss='mse', b_size=100, epochs=5000, val_split=0.2):
    # collect source data
    src_train_data = load_data_from_file(src_f_name)
    print('SOURCE DATA SHAPE:', src_train_data.shape)

    # collect target data
    tar_train_data = load_data_from_file(tar_f_name)
    print('TARGET DATA SHAPE:', tar_train_data.shape)

    ###################
    # initialize model
    net = get_model()
    net.summary()
    opt = Adam(lr=lr)
    net.compile(opt, loss)

    #################
    # train
    weights_path = os.path.join(model_dir, 'weights.h5')
    checkpoints = os.path.join(model_dir, 'epoch_{epoch:04d}_loss_{loss:.4f}.h5')
    callbacks = [ModelCheckpoint(checkpoints, period=10)]

    net.fit(src_train_data, tar_train_data, b_size, epochs, callbacks=callbacks, validation_split=val_split)
    net.save_weights(weights_path)


def predict(model_dir, test_f_name, save_to):
    # load model
    weights_path = os.path.join(model_dir, 'weights.h5')

    if not os.path.exists(weights_path):
        weights_path = sorted(glob.glob(model_dir + '/*.h5'))[-1]

    net = get_model()
    net.load_weights(weights_path)

    # predict
    with open(test_f_name) as f:
        for fname in f.read().splitlines():
            print('predicting', fname)
            test_data = np.load(fname)
            pred_data = net.predict(test_data)

            name = 'pred_' + os.path.basename(fname)
            pred_fname = os.path.join(save_to_dir, name)
            np.save(pred_fname, pred_data)
