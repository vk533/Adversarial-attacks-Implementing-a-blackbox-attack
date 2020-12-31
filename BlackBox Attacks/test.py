import warnings
warnings.filterwarnings('ignore')

import sys
import numpy as np

from keras.models import load_model
from sklearn.metrics import accuracy_score

from BlackBox import BB_Ki_Virus

################### DEFINING AND LOADING PARAMETERS ###################

test_data_file = sys.argv[1]
target_model_file = sys.argv[2]
bb_model_file = sys.argv[3]

test_label_file = './data/processed_test_labels.npy'

################### READING AND PROCESSING DATA ###################

test_data = np.load(test_data_file)
test_labels = np.load(test_label_file)

################### LOADING MODELS AND GETTING ACCURACIES ###################

target_model = load_model(target_model_file)

bb_model = load_model(bb_model_file)

org_pred = target_model.predict(test_data)
y_org = np.array(list(map(np.argmax, org_pred)))

org_acc = accuracy_score(test_labels, y_org)

mera_virus = BB_Ki_Virus()

test_acc = mera_virus.check_test_acc(test_data,
                                     test_labels,
                                     bb_model,
                                     target_model)

print("Original Accuracy :", org_acc)
print("Accuracy After Adverserial Attack :", test_acc)
print("Accuracy Dropped a Total By :", org_acc - test_acc)