import cv2
import random
from tqdm import tqdm


NUM_OF_REC = 500

# path  
path = r'C:\Users\zache\Documents\Machine Learning\images\Rec/'

# Black color in BGR 
color = (0, 0, 0) 
window_name = 'Image'


for i in tqdm(range(NUM_OF_REC)):
    image = cv2.imread(path + 'base.png', 0) 
    thickness = random.randint(1,2)
    startx = random.randint(0,50)
    starty = random.randint(0,50)
    start_point = (startx, starty) 
    end_point = (random.randint(startx+5,100), random.randint(starty+5,100)) 


    image = cv2.rectangle(image, start_point, end_point, color, thickness) 
    # Displaying the image  
    cv2.imwrite(path + str(i)+ ".png", image) 