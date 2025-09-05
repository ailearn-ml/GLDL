import numpy as np
import os

base_path = os.path.dirname(__file__).replace('\\', '/') + '/..'


def load_dg_dataset(data_name):
    if data_name == 'VSD':
        x_train_Flickr = np.load(f'{base_path}/data/VSD/Flickr_LDL_train_features_imagenet_resnet18.npy')
        x_train_Twitter = np.load(f'{base_path}/data/VSD/Twitter_LDL_train_features_imagenet_resnet18.npy')
        x_val_Flickr = np.load(f'{base_path}/data/VSD/Flickr_LDL_test_features_imagenet_resnet18.npy')
        x_val_Twitter = np.load(f'{base_path}/data/VSD/Twitter_LDL_test_features_imagenet_resnet18.npy')
        x_test = np.load(f'{base_path}/data/VSD/Abstract_test_features_imagenet_resnet18.npy')
        y_train_Flickr = np.load(f'{base_path}/data/VSD/Flickr_LDL_train_labels.npy')
        y_train_Twitter = np.load(f'{base_path}/data/VSD/Twitter_LDL_train_labels.npy')
        y_val_Flickr = np.load(f'{base_path}/data/VSD/Flickr_LDL_test_labels.npy')
        y_val_Twitter = np.load(f'{base_path}/data/VSD/Twitter_LDL_test_labels.npy')
        y_test = np.load(f'{base_path}/data/VSD/Abstract_test_labels.npy')
        x_trains = [[x_train_Flickr, x_train_Twitter]]
        x_vals = [[x_val_Flickr, x_val_Twitter]]
        x_tests = [x_test]
        y_trains = [[y_train_Flickr, y_train_Twitter]]
        y_vals = [[y_val_Flickr, y_val_Twitter]]
        y_tests = [y_test]

    elif data_name == 'FBP':
        XTrains, XTests, YTrains, YTests = [], [], [], []
        x_trains, x_vals, x_tests, y_trains, y_vals, y_tests = [], [], [], [], [], []
        types = ['AF', 'AM', 'CF', 'CM']
        for split in range(len(types)):
            XTrains.append(
                np.load(f'{base_path}/data/FBP/FBP5500_{types[split]}_train_features_imagenet_resnet18.npy'))
            XTests.append(
                np.load(f'{base_path}/data/FBP/FBP5500_{types[split]}_test_features_imagenet_resnet18.npy'))
            YTrains.append(np.load(f'{base_path}/data/FBP/FBP5500_{types[split]}_train_labels.npy'))
            YTests.append(np.load(f'{base_path}/data/FBP/FBP5500_{types[split]}_test_labels.npy'))
        for split in range(len(types)):
            source_index = list(set(range(len(types))) - set([split]))
            XT, XV, YT, YV = [], [], [], []
            for i in range(len(source_index)):
                XT.append(XTrains[source_index[i]])
                XV.append(XTests[source_index[i]])
                YT.append(YTrains[source_index[i]])
                YV.append(YTests[source_index[i]])
            x_trains.append(XT)
            x_vals.append(XV)
            y_trains.append(YT)
            y_vals.append(YV)
            x_tests.append(XTests[split])
            y_tests.append(YTests[split])

    elif data_name == 'MRD':
        XTrains, XTests, YTrains, YTests = [], [], [], []
        x_trains, x_vals, x_tests, y_trains, y_vals, y_tests = [], [], [], [], [], []
        types = ['Drama', 'Comedy', 'Thriller', 'Romance', 'Action',
                 'Horror', 'Crime', 'Documentary', 'Adventure', 'Science Fiction']
        for split in range(len(types)):
            XTrains.append(np.load(f'{base_path}/data/MRD/MRD_{types[split]}_train_features.npy'))
            XTests.append(np.load(f'{base_path}/data/MRD/MRD_{types[split]}_test_features.npy'))
            YTrains.append(np.load(f'{base_path}/data/MRD/MRD_{types[split]}_train_labels.npy'))
            YTests.append(np.load(f'{base_path}/data/MRD/MRD_{types[split]}_test_labels.npy'))
        for split in range(len(types)):
            source_index = list(set(range(len(types))) - set([split]))
            XT, XV, YT, YV = [], [], [], []
            for i in range(len(source_index)):
                XT.append(XTrains[source_index[i]])
                XV.append(XTests[source_index[i]])
                YT.append(YTrains[source_index[i]])
                YV.append(YTests[source_index[i]])
            x_trains.append(XT)
            x_vals.append(XV)
            y_trains.append(YT)
            y_vals.append(YV)
            x_tests.append(XTests[split])
            y_tests.append(YTests[split])
    return x_trains, x_vals, x_tests, y_trains, y_vals, y_tests
