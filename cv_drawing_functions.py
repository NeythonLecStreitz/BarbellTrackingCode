import cv2 as cv 
import numpy as np
# colors to use when drawing to the frame of a video
BLACK = (0,0,0)
WHITE = (255,255,255)
BLUE = (255,0,0)
RED = (0,0,255)
CYAN = (255,255,0)
YELLOW =(0,255,255)
MAGENTA = (255,0,255)
GREEN = (0,255,0)
PURPLE = (128,0,128)
ORANGE = (0,165,255)
PINK = [147,20,255]    
INDIGO=[75,0,130]   
VIOLET=[238,130,238]   
GRAY=[127,127,127]  

# Helper function for drawing text to a frame with only a bounding box.
def textBG(img, text, position, fonts ,scaling=1, color=(GREEN), thickness=1, padding=3):
    img_h, img_w = img.shape[:2]
    x, y = position
    (w, h ), p= cv.getTextSize(text, fonts, scaling, thickness)
    # print(w, h)
    cv.rectangle(img, (x-p, y+p), (x+w+p, y-h-p), (255, 0,234), -1)
    
    cv.putText(img, text, position,fonts, scaling,  color, thickness)

# Helper function for drawing text to frame with a bounding box and background.
def textBGoutline(img, text, position, fonts=cv.FONT_HERSHEY_COMPLEX ,scaling=1, text_color=(0,255,0), thickness=1, bg_color=(WHITE)):
    img_h, img_w = img.shape[:2]
    x, y = position
    (w, h ), p= cv.getTextSize(text, fonts, scaling, thickness)
    # print(w, h)
    cv.rectangle(img, (x-p, y+p), (x+w+p, y-h-p), bg_color, -1)
    cv.rectangle(img, (x-p, y+p), (x+w+p, y-h-p), text_color,thickness, cv.LINE_AA)
    
    cv.putText(img, text, position,fonts, scaling,  text_color, thickness, cv.LINE_AA)