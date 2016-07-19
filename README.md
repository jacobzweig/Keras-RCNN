# Keras-RCNN

Some experimenting with Keras to build Recurrent Convolutional Neural Networks, based on the paper [Recurrent Convolutional Neural Network for Object Recognition](http://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Liang_Recurrent_Convolutional_Neural_2015_CVPR_paper.pdf). 

```
# Build a model
model = BuildRCNN(nbChannels, shape1, shape2, nbClasses, nbRCL, nbFilters, filtersize)

#Compile it
model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

#fit
model.fit(X_train, y_train, batch_size=64, nb_epoch=100, validation_data = (X_valid, y_valid))
```


