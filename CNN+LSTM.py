from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense
from keras import applications
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.applications.vgg16 import VGG16
from keras.layers import LSTM
import numpy as np
from keras import models
import cv2
import matplotlib.pyplot as plt



train_dir = '/home/john/Downloads/4th_year_thesis/Working_on_public/Data/Train/'
valid_dir = '/home/john/Downloads/4th_year_thesis/Working_on_public/Data/Valid/'
test_dir = '/home/john/Downloads/4th_year_thesis/Working_on_public/Data/Test/'
batch_size = 32

def bring_data_from_directory():
  train_datagen = ImageDataGenerator(rescale=1. / 255)
  valid_datagen = ImageDataGenerator(rescale=1. /225)
  train_generator = train_datagen.flow_from_directory(
    directory=train_dir,
    target_size=(224, 224),
    color_mode="rgb",
    batch_size=32,
    class_mode="categorical",
    shuffle=False
)


  validation_generator =  valid_datagen.flow_from_directory(
    directory=valid_dir,
    target_size=(224, 224),
    color_mode="rgb",
    batch_size=32,
    class_mode="categorical",
    shuffle=False
)
#  for x, y in train_generator:
#      for frames in x:
#          plt.imshow(frames, cmap='gray')
#          plt.show()
  return train_generator,validation_generator

def load_VGG16_model():
  base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224,224,3))
  model = models.Sequential()

# Add the vgg convolutional base model
  model.add(base_model)
 
# Add new layers
  model.add(Flatten())
  model.add(Dense(1024, activation='relu'))

  print("Model loaded..!")
  print(base_model.summary())
  return model

def extract_features_and_store(train_generator,validation_generator,base_model, load_data=False):
  if load_data == True:
      x_generator = None
      y_lable = None
      batch = 0
      for x,y in train_generator:
          if batch == (train_generator.n//batch_size):
              break
          print("predict on batch:",batch)
          batch+=1
          if batch == 1:
             x_generator = base_model.predict_on_batch(x)
             y_lable = y
             print(y)
          if batch == (train_generator.n//batch_size):
              break
          else:
             x_generator = np.append(x_generator,base_model.predict_on_batch(x),axis=0)
             y_lable = np.append(y_lable,y,axis=0)
      
      np.save('video_x_VGG16.npy',x_generator)
      np.save('video_y_VGG16.npy',y_lable)
      batch = 0
      x_generator = None
      y_lable = None
      for x,y in validation_generator:
          if batch == (validation_generator.n//batch_size):
              break
          print("predict on batch validate:",batch)
          batch+=1
          if batch==1:
             x_generator = base_model.predict_on_batch(x)
             y_lable = y
          if batch == (validation_generator.n//batch_size):
              break
          else:
             x_generator = np.append(x_generator,base_model.predict_on_batch(x),axis=0)
             y_lable = np.append(y_lable,y,axis=0)
     
      np.save('video_x_validate_VGG16.npy',x_generator)
      np.save('video_y_validate_VGG16.npy',y_lable)
  

  train_data = np.load('video_x_VGG16.npy')
  train_labels = np.load('video_y_VGG16.npy')
  
  validation_data = np.load('video_x_validate_VGG16.npy')
  validation_labels = np.load('video_y_validate_VGG16.npy')
 
 
  train_data = train_data.reshape(train_data.shape[0]//10,
                     10,
                     train_data.shape[1])
  validation_data = validation_data.reshape(validation_data.shape[0]//10,
                     10,
                     validation_data.shape[1])
  train_labels = train_labels[1::10]
  validation_labels = validation_labels[1::10]
  
  print(train_data.shape)
  print(train_labels.shape)
  print(validation_data.shape)
  print(validation_labels.shape)
  
  return train_data,train_labels,validation_data,validation_labels

def train_model(train_data,train_labels,validation_data,validation_labels):
  ''' used fully connected layers, SGD optimizer and 
      checkpoint to store the best weights'''

  model = Sequential()
  model.add(LSTM(256,dropout=0.2,input_shape=(10, 1024)))
  model.add(Dense(40, activation='softmax'))
  sgd = SGD(lr=0.00005, decay = 1e-6, momentum=0.9, nesterov=True)
  model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
  #model.load_weights('video_1_LSTM_1_512.h5')
  callbacks = [ EarlyStopping(monitor='val_loss', patience=10, verbose=0), ModelCheckpoint('video_1_LSTM_1_1024.h5', monitor='val_loss', save_best_only=True, verbose=0) ]
  nb_epoch = 10
  model.fit(train_data,train_labels,validation_data=(validation_data,validation_labels),batch_size=batch_size,nb_epoch=nb_epoch,callbacks=callbacks,shuffle=True,verbose=1)
  return model


  
if __name__ == '__main__':
  train_generator,validation_generator = bring_data_from_directory()
  base_model = load_VGG16_model()
  train_data,train_labels,validation_data,validation_labels = extract_features_and_store(train_generator,validation_generator,base_model, load_data=False)
  train_model(train_data,train_labels,validation_data,validation_labels)
#  test_on_whole_videos(train_data,train_labels,validation_data,validation_labels)