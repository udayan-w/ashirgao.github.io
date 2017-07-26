---
layout: post
title:  "Paramters in a Convolutional Neural Network"
date:   2017-07-23 12:12:12 +0530
---

A convolutional neural network (CNN) ! From my experience, the deeper (:P) you study them, the more they amaze you with their capability and awe you with their structure (which you thought you already knew). 

If you are a beginner and have a cursory knowledge of how a Convolutional Neural Network works, you have hit a jackpot by reaching this page. In this post, I am going in depth through the forward propogation phase in the training of a CNN. I will discuss and calculate the number of parameters in a CNN as well as closely study the shape transformation of the training batch as we propogate through the network.

Understanding this will help you better grasp the working of a CNN and design better models. Note that the following post uses a keras example, but the number of parameter calculation and shape transformation you will learn here is same for all deep learning libraries.

For demonstration purposes let us study the model structure described [here](https://github.com/fchollet/keras/blob/master/examples/cifar10_cnn.py). Nothing special about that model, just that I don't have to write a fresh code from the ground up.  

I will not be showing the entire code here. I am just showing snippets of code that are relevant to the task we have undertaken.

We will be considering the [CIFAR-10 dataset](https://www.cs.toronto.edu/~kriz/cifar.html).
The dataset is made of
- 50000 training samples
- 10000 test samples

Each sample contains a RGB image of resolution 32*32 and a label mapping it to one of 10 output classes.


First, let us understand some terms in reference to the keras development environment :

- A sample - One element in the dataset. eg. A row in a .csv file or dataframe, a image for a CNN, etc.

- A batch - A batch is a collection of samples of specified length. Instead of updating model parameters after every single sample (online learning), we update them after every batch. We assume that a batch is a better approximation of the dataset than a single sample. Hence, it is better to shuffle your data to get benefits of using batches.

- An epoch - We say that an epoch is complete when the model has been through the whole dataset.


 
So, let us begin.

The shape of a single image depends on the configuration of your keras.
You can find configuration details of your keras installation in a hidden folder in your home directory. You can see its contents by

```Shell
$ cat ~/.keras/keras.json
```
	{
	    "image_data_format": "channels_last",
	    "floatx": "float32",
	    "epsilon": 1e-07,
	    "backend": "tensorflow"
	}

Note the folder name is ```.keras```. This means it is a hidden folder. You can view it in - 
- Nautilus (File Manager in Ubuntu) by opening it to your home directory and pressing ```Ctrl + H```.
- In terminal by doing ```$ ls -a ~```

If you cannot find ```.keras``` in your home directory you can still view and modify configuration details by entering the following in the python terminal ([more](https://keras.io/backend/))

```python
>>> from keras import backend as K
>>> K.image_data_format()	#View
'channels_first'
>>> K.set_image_data_format('channels_last') #Change
>>> K.image_data_format()
'channels_last'
```	
You can have image data format as 'channels_first' or 'channels_last'. For this post we will continue with ```image_data_format``` as ```'channels_last'```. 
So our image has shape ```(32,32,3)```. The 3 channels correspond to the Red, Green and Blue components of the input image.

```python
batch_size = 32
num_classes = 10
epochs = 200
```
The batch size here is 32 samples. This means the model parameters will be updated after it sees 32 images. Shape of each batch is ```(32,32,32,3)```. The first 32 in the tuple represents the batch-size.

```python
# The data, shuffled and split between train and test sets:
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')
```

    x_train shape: (50000, 32, 32, 3)
    50000 train samples
    10000 test samples

Our total dataset has shape ```(50000, 32, 32, 3)```.


```python
# Convert class vectors to binary class matrices.
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
```
```to_categorical``` converts a label (which is a integer) into a vector by doing 1-hot-encoding. 

ie. 1 becomes [0,1,0,0,0,0,0,0,0,0], 2 becomes [0,0,2,0,0,0,0,0,0,0] and so on.

Read comments in following snippet :

```python
model = Sequential()

model.add(Conv2D(32, (3, 3), padding='same',input_shape=x_train.shape[1:]))# a.k.a. conv2d_1
# 32 filters(kernels) of shape 3*3 
model.add(Activation('relu'))

model.add(Conv2D(32, (3, 3)))# a.k.a. conv2d_2
# 32 filters(kernels) of shape 3*3 
model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2, 2))) # a.k.a. max_pooling2d_1
# Downsampling 
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), padding='same'))# a.k.a. conv2d_3 
# 64 filters(kernels) of shape 3*3
model.add(Activation('relu'))

model.add(Conv2D(64, (3, 3)))# a.k.a. conv2d_4
# 64 filters(kernels) of shape 3*3 
model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))
# Downsampling 
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes))
model.add(Activation('softmax'))

# initiate RMSprop optimizer
opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)

# Let's train the model using RMSprop
model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

```
The above snippet defines the model. Now, let us see the output of ```model.summary()``` .

```python
model.summary()
```

    _________________________________________________________________
	Layer (type)                 Output Shape              Param #   
	=================================================================
	conv2d_1 (Conv2D)            (None, 32, 32, 32)        896       
	_________________________________________________________________
	activation_1 (Activation)    (None, 32, 32, 32)        0         
	_________________________________________________________________
	conv2d_2 (Conv2D)            (None, 30, 30, 32)        9248      
	_________________________________________________________________
	activation_2 (Activation)    (None, 30, 30, 32)        0         
	_________________________________________________________________
	max_pooling2d_1 (MaxPooling2 (None, 15, 15, 32)        0         
	_________________________________________________________________
	dropout_1 (Dropout)          (None, 15, 15, 32)        0         
	_________________________________________________________________
	conv2d_3 (Conv2D)            (None, 15, 15, 64)        18496     
	_________________________________________________________________
	activation_3 (Activation)    (None, 15, 15, 64)        0         
	_________________________________________________________________
	conv2d_4 (Conv2D)            (None, 13, 13, 64)        36928     
	_________________________________________________________________
	activation_4 (Activation)    (None, 13, 13, 64)        0         
	_________________________________________________________________
	max_pooling2d_2 (MaxPooling2 (None, 6, 6, 64)          0         
	_________________________________________________________________
	dropout_2 (Dropout)          (None, 6, 6, 64)          0         
	_________________________________________________________________
	flatten_1 (Flatten)          (None, 2304)              0         
	_________________________________________________________________
	dense_1 (Dense)              (None, 512)               1180160   
	_________________________________________________________________
	activation_5 (Activation)    (None, 512)               0         
	_________________________________________________________________
	dropout_3 (Dropout)          (None, 512)               0         
	_________________________________________________________________
	dense_2 (Dense)              (None, 10)                5130      
	_________________________________________________________________
	activation_6 (Activation)    (None, 10)                0         
	=================================================================
	Total params: 1,250,858
	Trainable params: 1,250,858
	Non-trainable params: 0
	_________________________________________________________________


In the summary, we can see that every output shape has ```None``` in place of the batch-size. This is so as to facilitate changing of batch size at runtime. 

Let us now go through each layer (that modifies the shape of the input tensor) in detail studying output shape and number of parameters for each. Shape of the input image is ```(None,32,32,3)```

- For ```conv2d_1``` the input shape is ```(None,32,32,3)``` ie our input batch. This layer, as commented in the code snippet earlier, has 32 filters of shape 3 * 3. But in actual the filter isn't 2D but 3D. The value for the 3rd dimesnion is the number of channels in the previous layers. So the filter shape is infact ```(3*3*3)```. Here the first 3 * 3 are the filter size and the last 3 being the number of channels in our input ie 3(RGB). Each filter in this layer has a total of 3 * 3 * 3= 27 weight parameters and 1 bias paramter amounting to a total 28 paramters. Also we have 32 such filters. Thus, total there are ```(27+1)*32 = 896 paramters``` in this layer. As we have specified ```padding='same'``` the image shape does not change. So, the output shape is now ```(None,32,32,32)```. Here the first two 32's represent the image shape and the third 32 specifies number of channels. [More about Conv2D layer in keras](https://keras.io/layers/convolutional/#conv2d).

![CNN](https://i.stack.imgur.com/T2RWP.png)
source:https://www.nvidia.cn/content/tesla/pdf/machine-learning/imagenet-classification-with-deep-convolutional-nn.pdf

The above diagram might help you visualize a CNN. The values and layers in the diagram do not match our example model. But I have still included this diagram for the lack of a better one. As you can see a filter is actually a 3D tensor of weights and each convolution ie. this 3D weight tensor and the input tensor produces a single value in the next layer.

- For ```conv2d_2``` the input shape is ```(None,32,32,32)``` ie the output from ```activation_1``` layer. This layer too has 32 filters of shape 3 * 3. Here the actual shape of our filters is ```(3*3*32)``` ie. (3*3) as specified in the model and the 3rd dimension is given by the number of channels in the input 32 (the last 32 in the input shape). Number of parameters per filter is one for weight parameter for every location in the filter ie 3 * 3 * 32 =  288 and an additional 1 bias parameter making the total to 289. Thus, total number of paramters for 32 such filters is ```(288+1)*32 = 9248 paramters```. Here, we haven't specified any value for the padding parameter. Thus, it defaults to 'valid', ie. no zeros are appended and thus the image size is now (30,30). We again have 32 channels, so the output shape is ```(None,30,30,32)```. [More about Conv2D layer in keras](https://keras.io/layers/convolutional/#conv2d).
- For ```max_pooling2d_1``` the input shape is ```(None,30,30,32)```. Here we have specified ```pool_size=(2,2)```. This means that this layer will reduce a 2 * 2 area of a channel to a single value ie 4 values to 1. But this is not what reduces the size of the tensor. The ```MaxPooling2D``` layer also accepts a paramter called ```stride``` if which not specified defaults to pool_size. Stride is a 2 value tuple that specifies the number of values to move in respective direction before doing a 4-to-1 pool. Thus, in this case the stride is (2,2). Thus our output tensor shape is now ```(None,15,15,32)```. Note how pooling does not affect the channels.[More about MaxPooling2D layer in keras](https://keras.io/layers/pooling/#maxpooling2d).


Similarly for the second block.
- For ```conv2d_3``` the input shape is ```(None,15,15,32)``` ie the output from ```dropout_1``` layer. This layer 64 filters of shape 3 * 3. We can see in this post and many other cases that the filter size of 3 *3 is very common (VGG16 also uses 3 * 3 filters). The intuition behind choosing them is that the small receptive field digests pattern details better than larger (read sparser) receptive fields.  Here the actual shape of our filters is ```(3*3*32)``` ie. (3 * 3) as specified in the model and the 3rd dimension is given by the number of channels in the input 32 (the last 32 in the input shape). Number of parameters per filter is one for weight parameter for every location in the filter ie 3 * 3 * 32 =  288 and an additional 1 bias parameter making the total to 289. Thus, total number of paramters for 64 such filters is ```(288+1)*64 = 18496 paramters```. As we have specified ```padding='same'``` the image shape does not change after convolution. So, the output shape is now ```(None,15,15,64)```. Here the first two 15's represent the image shape and the third 64 specifies number of channels(number of filters in this layer). [More about Conv2D layer in keras](https://keras.io/layers/convolutional/#conv2d).

- For ```conv2d_4``` the input shape is ```(None,15,15,64)``` ie the output from ```activation_3``` layer. This layer too has 64 filters of shape 3 * 3. Here the actual shape of our filters is ```(3*3*64)``` ie. (3*3) as specified in the model and the 3rd dimension is given by the number of channels in the input 64 (the last 64 in the input shape). Number of parameters per filter is one for weight parameter for every location in the filter ie 3 * 3 * 64 =  576 and an additional 1 bias parameter making the total to 577. Thus, total number of paramters for 64 such filters is ```(576+1)*64 = 36928 paramters```. Here, we haven't specified any value for the padding parameter. Thus, it defaults to 'valid', ie. no zeros are appended and thus the image size is now (13,13). We again have 64 channels, so the output shape is ```(None,13,13,64)```. [More about Conv2D layer in keras](https://keras.io/layers/convolutional/#conv2d).
- For ```max_pooling2d_2``` the input shape is ```(None,13,13,64)```. Here we have specified ```pool_size=(2,2)``` and the stride also defaults to (2,2). This means that this layer will reduce a 2 * 2 area of a channel to a single value ie 4 values to 1 in that channel and will repeat this after a taking stride of (2,2) in respective directions. Thus our output tensor shape is now ```(None,6,6,64)```. Note how pooling does not affect the channels.[More about MaxPooling2D layer in keras](https://keras.io/layers/pooling/#maxpooling2d).

Now for the final layers in our model.
- ```flatten_1``` is a [Flatten layer](https://keras.io/layers/core/#flatten). Input for this layer has shape ```(None,6,6,64)```. As the name suggests, this layer will flatten the input. The length of the flattened vector will be ```6*6*64 = 2304```. Thus, the output shape will be ```(None,2304)```.
- ```dense_1``` is a [Dense layer](https://keras.io/layers/core/#dense). This layer has 512 unit. The input has shape ```(None,2304)```. So, there will be connections from each 2304 units from the Flatten layer to each and every 512 unit in this Dense. Each connection will have a weight parameter. Also each unit will have one bias parameter. So total number of parameters is ```2304*512``` weight parameters  + ```512``` bias parameters  = ```1180160```. The new output shape is ```(None,512)```.
- Similarly for ```dense_2``` which has 10 units, we can calculate number of parameters as the sum of number of weight parameters ```512*10``` and bias parameters ```10``` totalling to ```5130``` parameters. Output shape for this layer is ```(None,512)```


I hope this post has helped you improve your understanding of Convolution Neural Networks. Let me know what you think in the comments below!

{% include disqus_comments.html %}
