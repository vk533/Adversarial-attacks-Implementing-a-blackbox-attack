import warnings
warnings.filterwarnings('ignore')

import numpy as np

from keras.models import Sequential
from keras.layers import Conv2D, Flatten, Dense

from keras import backend as K

from sklearn.metrics import accuracy_score

class BB_Ki_Virus:
    
    def __init__(self, main_epochs=10, num_nodes=100, bb_optimizer='adma', bb_epochs=10):
        
        self.num_nodes = num_nodes
        self.main_epochs = main_epochs
        self.bb_epochs = bb_epochs
        self.bb_optimizer = bb_optimizer
    
    def get_idxs(self, min_val, max_val, num):

        idxs = np.random.randint(min_val,max_val,num)

        while len(set(idxs)) != num:
            idxs = np.random.randint(min_val,max_val,num)

        return idxs
    
    def get_bb_labels(self, data, model):
        
        bb_labels = model.predict(data)
        bb_labels = np.array(list(map(np.argmax, bb_labels)))
        
        return bb_labels
    
    def get_bb_model(self, inp_shape):

        model = Sequential()

        model.add(Conv2D(16, (3,3), activation='relu', input_shape=inp_shape))
        model.add(Flatten())
        model.add(Dense(self.num_nodes, activation='relu'))
        model.add(Dense(self.num_nodes, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))
    
        model.compile(loss='binary_crossentropy', optimizer=self.bb_optimizer, metrics=['accuracy'])
    
        return model
    
    def get_gradient(self, data, model):
        
        gradients = K.gradients(model.output, model.input)[0]
        iterate = K.function(model.input, gradients)
        
        grad = iterate([data])
        evaluated_gradient = grad[0]
        
        return evaluated_gradient
    
    def get_sign_vec(self, vec_size):
        
        sign_vec = np.random.uniform(-1,1,size=vec_size)
        sign_vec = np.sign(sign_vec)

        return sign_vec
    
    def generate_adv(self, data, model, mult_val = 0.1):

        gradient = self.get_gradient(data, model)
        #print("############# GRADIENT :", gradient.sum())
        sign_vec = self.get_sign_vec(data.shape[0])
        sign_vec = mult_val * sign_vec
  
        gen_adv = data + sign_vec[:,None, None, None] * np.sign(gradient)

        return gen_adv
    
    def get_non_bb_data(self, data, labels, idxs):
        
        new_idxs = np.arange(data.shape[0])
        idxs_mask = np.isin(new_idxs, idxs, invert=True)
        new_idxs = new_idxs[idxs_mask]

        non_bb_data = data[new_idxs]
        non_bb_labels = labels[new_idxs]
  
        return non_bb_data, non_bb_labels
    
    def update_bb_data(self, data,gen_adv):

        new_data = np.append(data, gen_adv, axis=0)

        return new_data
    
    def train_adv(self, bb_data, bb_labels, bb_model, target_model, test_data=None, test_labels=None, check_acc=False):
        
        if check_acc:
            
            if type(test_data) == None or type(test_labels) == None:
                
                raise ValueError("To Check for Test Accuracy Provide test data and test labels!")
            
            pred_org = target_model.predict(test_data)
            y_org = np.array(list(map(np.argmax, pred_org)))
            
            org_acc = accuracy_score(test_labels, y_org)
        
        for epoch in range(self.main_epochs):
            
            bb_model.fit(bb_data, bb_labels, epochs=self.bb_epochs, verbose=0)

            gen_adv = self.generate_adv(bb_data, bb_model)
            
            bb_data = self.update_bb_data(bb_data, gen_adv)
            
            bb_labels = self.get_bb_labels(bb_data, target_model)
            
            if check_acc:
                
                test_acc = self.check_test_acc(test_data, test_labels, bb_model, target_model)
                print("######## EPOCH = {} ########".format(epoch+1))
                print("Original Accuracy :",org_acc)
                print("Adverserial Accuracy :", test_acc)
                print("Drop in Accuracy :", org_acc - test_acc)
                print("\n")
        return bb_model
    
    def check_test_acc(self, test_data, test_labels, bb_model, target_model):
        
        test_adv = self.generate_adv(test_data, bb_model, mult_val=0.0625)
        
        new_test_data = self.update_bb_data(test_data, test_adv)
        new_test_labels = np.append(test_labels, test_labels, axis=0)
        
        pred_adv = target_model.predict(new_test_data)
        y_pred = np.array(list(map(np.argmax, pred_adv)))
        
        adv_acc = accuracy_score(new_test_labels, y_pred)
        
        return adv_acc