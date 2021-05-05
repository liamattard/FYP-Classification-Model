from dataHandling import images
from dataHandling import dataset 
from cnnModel import model

import tensorflow as tf
from tensorflow import keras
import numpy as np

def main():

    data = images.getImages('../dataset')
    # images.showSampleImage(data,"beach",2)

    batch_size = 32
    img_height = 180
    img_width = 180

    train,test,class_names = dataset.dataset_split(batch_size,img_height,img_width,data)
    # compiled_model = model.compile(img_height,img_width)
    # model.train(train,test,compiled_model)


    model = keras.models.load_model('my_model')



    newBeachTestTwo_url = "https://selfgrowth.info/photos/free-beach-photos-without-copyright/big-beach-illustrations-free-royalty6361.jpg"
    newBeachTestTwo_path = tf.keras.utils.get_file('newBeachTestTwo', origin=newBeachTestTwo_url)

    img = keras.preprocessing.image.load_img(
        newBeachTestTwo_path, target_size=(img_height, img_width)
    )
    img_array = keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) # Create a batch

    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])

    print(
        "This image most likely belongs to {} with a {:.2f} percent confidence."
        .format(class_names[np.argmax(score)], 100 * np.max(score))
    )

if __name__ == "__main__":
    main()