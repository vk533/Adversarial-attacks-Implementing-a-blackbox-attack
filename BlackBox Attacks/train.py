import warnings
warnings.filterwarnings('ignore')

import sys
import numpy as np

from keras.models import load_model

from BlackBox import BB_Ki_Virus

####################### DEFINING AND READING PARAMETERS #######################

test_data_file = sys.argv[1]
target_model_file = sys.argv[2]
bb_model_file = sys.argv[3]

test_labels_file = './data/test_label.npy'

main_epochs = 5

num_nodes = 100
bb_idxs_num = 200
bb_epochs = 10
bb_optimizer = 'adam'

####################### READING AND PROCESSING DATA #######################

mera_virus = BB_Ki_Virus(main_epochs=main_epochs,
                         num_nodes=num_nodes,
                         bb_optimizer=bb_optimizer,
                         bb_epochs=bb_epochs)

target_model = load_model(target_model_file)

data = np.load(test_data_file)
labels = np.load(test_labels_file)

data = data - np.mean(data, axis=0)

main_data = data[np.where(labels <= 1)]
main_labels = labels[np.where(labels <= 1)]

bb_min_val = 0
bb_max_val = len(main_labels)

bb_idxs = mera_virus.get_idxs(bb_min_val, bb_max_val, bb_idxs_num)
bb_data = main_data[bb_idxs]

bb_labels = mera_virus.get_bb_labels(bb_data, target_model)

test_data, test_labels = mera_virus.get_non_bb_data(main_data, main_labels, bb_idxs)

np.save('./data/processed_test_data', test_data)
np.save('./data/processed_test_labels', test_labels)

####################### TRAINING MODEL #######################

bb_model = mera_virus.get_bb_model(inp_shape=(32,32,3))

bb_model = mera_virus.train_adv(bb_data,
                                bb_labels,
                                bb_model,
                                target_model,
                                test_data=test_data,
                                test_labels=test_labels,
                                check_acc=True)

bb_model.save(bb_model_file)