from dataHandling import images
from dataHandling import dataset 
from cnnModel import model
from tensorflow import keras

data = images.getImages('../dataset')
# images.showSampleImage(data,"beach",2)

batch_size = 32
img_height = 180
img_width = 180

# train,test = dataset.dataset_split(batch_size,img_height,img_width,data)
# compiled_model = model.compile(img_height,img_width)
# model.train(train,test,compiled_model)
model = keras.models.load_model('my_model.h5')


