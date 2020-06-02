import os
import cv2
import numpy as np
from tqdm import tqdm

class Tables():
    IMG_SIZE = 500
    TABLES = "images/train"
    PAGES = "images/train"
    TESTING = "images/test"
    LABELS = {TABLES: 0, PAGES: 0}
    training_data = []
    i = 0;
    directory = r'C:\Users\zache\Documents\Machine Learning\images\trainimg'

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
                        # os.chdir(self.directory)
                        # self.i +=1
                        # filename= str(self.i)+'.png'
                        # cv2.imwrite(filename,img)
                        #print(np.eye(2)[self.LABELS[label]])
                    except Exception as e:
                        
                        print(label, f, str(e))
                        pass
            np.random.shuffle(self.training_data)
            np.save("training_data.npy", self.training_data)
    
tables = Tables()
tables.make_training_data()
print(tables.training_data)