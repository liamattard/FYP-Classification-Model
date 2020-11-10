import pathlib
import PIL
import matplotlib.pyplot as plt

def getImages(path):
    '''
    Loads images from a dataset.
    '''
    data = pathlib.Path(path)
    total_images = len(list(data.glob('*/*')))
    total_classes = len(list(data.glob('*')))
    print("Total number of images: " + str(total_images))
    print("Total number of classes: " + str(total_classes))
    return data
    
def showSampleImage(data, classifier, imageNumber):
    '''
    Show an image from a dataset given the classifier
    the number.
    '''
    label = list(data.glob(classifier + '/*'))
    im = PIL.Image.open(str(label[imageNumber]))
    im.show()

