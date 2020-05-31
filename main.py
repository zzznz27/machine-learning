import os
import cv2
import numpy as np
from tqdm import tqdm

class Tables():
    IMG_SIZE = 100
    TRAIN = "images/train"
    TESTING = "images/test"
    LABELS = {TRAIN: 0}
    training_data = []

    def make_training_data(self):
        for label in self.LABELS:
            for f in tqdm(os.listdir(label)):
                if "png" in f:
                    try:
                        # print(f)
                        path = os.path.join(label, f)
                        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                        img = cv2.resize(img, (self.IMG_SIZE, self.IMG_SIZE))
                        self.training_data.append([np.array(img), np.eye(2)[self.LABELS[label]]])  # do something like print(np.eye(2)[1]), just makes one_hot 
                        #print(np.eye(2)[self.LABELS[label]])
                    except Exception as e:
                        pass
                        #print(label, f, str(e))

            np.random.shuffle(self.training_data)
            np.save("training_data.npy", self.training_data)
    
tables = Tables()
tables.make_training_data()
print(tables.training_data)