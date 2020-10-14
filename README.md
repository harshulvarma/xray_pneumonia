## Classifying X-ray images with Pneumonia

*Jupyter Notebook Link:* <https://nbviewer.jupyter.org/github/harshulvarma/xray_pneumonia/blob/main/xray_pneumonia.ipynb>

*GitHub Repository Link:* <https://github.com/harshulvarma/xray_pneumonia>

### Overview

The goal of the project is to implement transfer learning using a fine-tuned ResNet50 Convolutional Neural Network (CNN) with data augmentaation in Keras and Tensorflow to classify X-ray images with Pneumonia.

<img src="xray.png?raw=true"/>

### Methods

The dataset consists of ~6000 images and the architecture might overfit. To counter that I added data augmentation with Keras image generator.

` datagen = ImageDataGenerator(
    horizontal_flip=True,
    zoom_range=0.2,
    rotation_range=10)`
    
 In addition to that last few layers of ResNet50 were frozen as well.

``` 
for layer in model.layers[:20]:
  layer.trainable=False
for layer in model.layers[20:]:
  layer.trainable=True
```

A final few layers were added in place with pooling layer and 3 fully connected layers.

```
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.5)(x)
x = Dense(512,activation='relu')(x)
x = Dropout(0.2)(x)
x = Dense(128,activation='relu')(x)
preds = Dense(2,activation='softmax')(x)
```

### Results

The model currently achieves 0.9 F1, precision and recall on the test said. This great but the model also is currently overfitting slightly as seen by the accuracy plots and confusion matrix below. The future iterations will include an aggrsive data aumentation and reducing complexity of the model to reduce that.

<img src="xray3.png?raw=true"/><img src="xray2.png?raw=true"/>

### Tools Utilised

- pandas
- cv2
- Tensorflow
- Keras
- scikit-learn
