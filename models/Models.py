### Predefined Keras models

import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
import keras.backend as K
import torch.nn as nn
import torch.nn.functional as F
import torch.cuda as cuda
from torch.optim import SGD
from torch.autograd import Variable
import sys

def make_model(model_name):
    """Constructs the Keras model indicated by model_name"""
    model_maker_dict = {
            'example':make_example_model,
            'mnist':make_mnist_model,
            'mnist_pytorch': make_mnist_pytorch_model
            }
    return model_maker_dict[model_name]()

def make_example_model():
    """Example model from keras documentation"""
    model = Sequential()
    model.add(Dense(output_dim=64, input_dim=100))
    model.add(Activation("relu"))
    model.add(Dense(output_dim=10))
    model.add(Activation("softmax"))
    return model

def make_mnist_model(**args):
    """MNIST ConvNet from keras/examples/mnist_cnn.py"""
    np.random.seed(1337)  # for reproducibility
    nb_classes = 10
    # input image dimensions
    img_rows, img_cols = 28, 28
    # number of convolutional filters to use
    filters = args.get('nb_filters',32)
    # size of pooling area for max pooling
    ps = args.get('pool_size',2)
    pool_size = (ps,ps)
    # convolution kernel size
    ks = args.get('kernel_size',3)
    kernel_size = (ks, ks)
    do = args.get('drop_out', 0.25)
    dense = args.get('dense', 128)
    input_shape = (1, img_rows, img_cols)  # for agreement with pytorch
    K.set_image_dim_ordering('th')
#    if K.image_dim_ordering() == 'th':
#        input_shape = (1, img_rows, img_cols)
#    else:
#        input_shape = (img_rows, img_cols, 1)
    model = Sequential()
    model.add(Conv2D(filters, kernel_size,
                        padding='valid',
                        input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(Conv2D(filters, kernel_size))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=pool_size))
    model.add(Dropout(do))
    model.add(Flatten())
    model.add(Dense(dense))
    model.add(Activation('relu'))
    model.add(Dropout(do))
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))
    return model

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)
        
    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

class KerasWrapperForPytorchCPU(Net):
    def __init__(self):
        super().__init__()
        self.add_keras_variables()

    def add_keras_variables(self):
        self.loss = None
        self.optimizer = None
        self.metrics = []
        self.loss_functions = None
        self.metrics_names = ["loss"]
        self.callbacks = []

    def get_weights(self):
        return [i.data.numpy() for i in list(self.parameters())]
   
    def set_weights(self, weights=[]):
        import torch # Don't put it outside because it will break Tensorflow
        for i,weight in enumerate(weights):
            list(self.parameters())[i].data.copy_(torch.from_numpy(weight))
        return

    def compile(self, **kwargs)
        self.loss = nn.NLLLoss()
        for metric in kwargs['metrics']:
            if metric.lower() == 'acc' or metric.lower() == 'accuracy':
                self.metrics_names.append('acc')
        self.optimizer = SGD(self.parameters(), 1.)

        return

    def _accuracy(self, output, target, topk=(1,)):
        """Computes the precision@k for the specified values of k"""
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(1. / batch_size))
        return res
   
    def train_on_batch(self, x=None, y=None, *args, **kwargs):
        '''Perform a single gradient update on a single batch of data.
        Attributes:
        x: Pytorch tensor of training data
        y: Pytorch tensor of target data

        Return:
        A list of scalar training loss and a metric specified in the compile method.
        '''
        self.train()
        self.optimizer.zero_grad()
        pred = self.forward(Variable(x))
        target = y.long().max(1)[1] # Pytorch doesn't need 1-hot encoded label. Only the indices of classes.
        loss = self.loss(pred, Variable(target)) 
        loss.backward()
        self.optimizer.step()
        self.metrics = [loss.data.numpy()[0]]
        if 'acc' in self.metrics_names: # compute the accuracy
            acc = self._accuracy(pred.data, target, topk=(1,))[0]
            self.metrics.append(acc.numpy()[0])
        return self.metrics

    def test_on_batch(self, x=None, y=None, *args, **kwargs):
        '''Test the model on a single batch of samples. No gradient update is performed.
        Attributes:
        x: Pytorch tensor of test data
        y: Pytorch tensor of target data

        Return:
        A list of scalar training loss and a metric specified in the compile method.
        '''
        self.eval()
        pred = self.forward(Variable(x, volatile=True))
        target = y.long().max(1)[1] # Pytorch doesn't need 1-hot encoded label. Only the indices of classes.
        loss = self.loss(pred, Variable(target, volatile=True)) 
        self.metrics = [loss.data.numpy()[0]]
        if 'acc' in self.metrics_names: # compute the accuracy
            acc = self._accuracy(pred.data, target, topk=(1,))[0]
            self.metrics.append(acc.numpy()[0])
        return self.metrics

class DataParallelKerasInterface(nn.DataParallel):
    # To be implemented
    def __init__(self):
        super().__init__()

def make_mnist_pytorch_model():
    model = KerasWrapperForPytorchCPU()
    #import torch
    #if torch.cuda.is_available():  # To be implemented
    #    model = torch.nn.DataParallel(model).cuda()
    return model
