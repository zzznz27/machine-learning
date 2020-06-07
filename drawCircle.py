import cv2
import random
from tqdm import tqdm


NUM_OF_CIRCLE = 500

# path  
path = r'C:\Users\zache\Documents\Machine Learning\images\Circle/'

# Black color in BGR 
color = (0, 0, 0) 
window_name = 'Image'


for i in tqdm(range(NUM_OF_CIRCLE)):
    image = cv2.imread(path + 'base.png', 0) 
    thickness = random.randint(1,2)
    startx = random.randint(0,50)
    starty = random.randint(0,50)
    radius = random.randint(1,30)
    start_point = (startx+radius, starty+radius) 
    
    image = cv2.circle(image,start_point,radius,color,thickness)
    

    # Displaying the image  
    cv2.imwrite(path + str(i)+ ".png", image) 