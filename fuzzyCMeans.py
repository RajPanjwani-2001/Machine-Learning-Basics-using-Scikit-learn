import numpy as np
import cv2
from fcmeans import FCM

img = cv2.imread('leaf.jpg')
print(img.shape)

X = img.reshape(img.shape[0]*img.shape[1],3)
print('X shape: ',X.shape)

num_clusters = 5
fcm = FCM(n_clusters = num_clusters)
fcm.fit(X)

#Pixel Quantization
labeled_X = fcm.predict(X)

# Single image with all clusters
'''transformed_X = fcm.centers[labeled_X]
print(labeled_X)
print(transformed_X.shape)

quantized_array = transformed_X.reshape(img.shape[0], img.shape[1], 3)
quantized_array = quantized_array.astype('uint8')

print(quantized_array.shape)

cv2.imshow('input',img)
cv2.imshow('quantized',quantized_array)

cv2.waitKey(0)
cv2.destroyAllWindows()'''

obj = {}
for i in range(len(labeled_X),-1):
    obj[str(i)] = []

print(obj)
transformed_X = fcm.centers[labeled_X]
print(labeled_X)
print(transformed_X.shape)

quantized_array = transformed_X.reshape(img.shape[0], img.shape[1], 3)
quantized_array = quantized_array.astype('uint8')

print(quantized_array.shape)

cv2.imshow('input',img)
cv2.imshow('quantized',quantized_array)

cv2.waitKey(0)
cv2.destroyAllWindows()








































































