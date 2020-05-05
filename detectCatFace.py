#!/usr/bin/env python
# coding: utf-8

# In[17]:


import matplotlib.pyplot as plt
import cv2


# In[18]:


#Change the path ! 
cat_face_cascade = cv2.CascadeClassifier('DATA/haarcascades/haarcascade_frontalcatface_extended.xml')


# In[19]:


#Show the selected image
cat = cv2.imread('DATA/photosTest/mustie0.jpg',0)
plt.imshow(cat,cmap='gray')


# In[20]:


#Cat face detection function
def detectCatFace(img):
    cat_img = img.copy()
    cat_rects = cat_face_cascade.detectMultiScale(cat_img)
    for (x,y,w,h) in cat_rects: 
        cv2.rectangle(cat_img, (x,y), (x+w,y+h), (255,255,255), 10) 
    return cat_img


# In[21]:


#function to extract cat's face    
def recoverCatFace(img):
    cat_img = img.copy()
    cat_rects = cat_face_cascade.detectMultiScale(cat_img)
    cat_face = ()
    if cat_rects != ():
        for (x,y,w,h) in cat_rects: 
            cat_face = cat_img[y:y+h,x:x+w]
    return cat_face


# In[22]:


#call the function and show the result     
result = detectCatFace(cat)
plt.imshow(result,cmap='gray')


# In[23]:


#call the extract function and show the result
face = recoverCatFace(cat)
plt.imshow(face,cmap='gray')


# In[24]:


#Change the path ! 
#Only doing that if "face" is not empty
cv2.imwrite('Data/Autres/randomCatFace.jpg',face)


# In[ ]:


#Change project --> Dog Flap


# In[ ]:




