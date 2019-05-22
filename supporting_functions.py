import numpy
import torch
from math import floor
import torch.nn as nn
import csv
import torch.optim as optim
from PytorchDataset import PytorchDataset

def get_generalized_approximation(x_values):
    # returns the corresponding y values for the molcule independent approximation function
    # x in this scenario is the distance between the molecules, with y being the force
    # at x==0 the molecules are guaranteed to collapse into a black hole, and no answer exists
    return 4 * 1 * ((1 / x_values) ** 12 - (1 / x_values) ** 6)


def equal_spaced_n(start, stop, n):
    # returns n (rounded up if necessary) equal spaced values between [start, stop>
    return numpy.arange(start, stop, (stop-start)/n)

def uniform_random_n(start, stop, n):
    # returns n uniform randomly picked values between [start,stop>
    return numpy.random.uniform(start, stop, n)

def datasets_generator(numpy_x, numpy_y, train_ratio, dataloader_params, **kwargs):
    # numpy data should be rows of x, y
    # returns a training and test torch dataset, according to trainig_ratio

    # if train_ratio+validate_ratio>=1:
    #     print("datasets generator got invalid configuration, exiting")
    #     exit()

    numpy_data = numpy.array(list(zip(numpy_x, numpy_y)))

    total_objects = len(numpy_data)
    train_objects = floor(train_ratio * total_objects)
    # validate_objects = floor(validate_ratio * total_objects)

    numpy_train = numpy_data[:train_objects]
    # numpy_validate = numpy_data[train_objects:(train_objects+validate_objects)]
    numpy_test = numpy_data[train_objects:]

    partition_train, labels_train = numpy_to_x_y(numpy_train)
    # partition_validate, labels, validate = nu
    partition_test, labels_test = numpy_to_x_y(numpy_test)

    trainset = PytorchDataset(partition_train, labels_train)
    trainloader = torch.utils.data.DataLoader(trainset, **dataloader_params)

    # validateset = PytorchDataset(partition_validate, labels_validate)
    # validateloader = torch.utils.data.DataLoader(validateset, **dataloader_params)

    testset = PytorchDataset(partition_test, labels_test)
    testloader = torch.utils.data.DataLoader(testset, **dataloader_params)

    return trainloader, testloader

def realset_generator(csv_path, dataloader_params):
    with open(csv_path) as csv_file:
        csv_reader = csv.DictReader(csv_file, delimiter = " ")
        real_x = []
        real_y = []
        # todo figure out what MRCI is and correct it
        for line in csv_reader:
            real_x.append(float(line['R [A]']))
            real_y.append(float(line['MRCI1 [eV]']))
        print(real_x)
        realtestset = PytorchDataset(real_x, real_y)
        realTestLoader = torch.utils.data.DataLoader(realtestset, **dataloader_params)

        # todo should probably return realtrainset aswell
        return realTestLoader


def numpy_to_x_y(numpy):
    print(numpy)
    return numpy[:,0], numpy[:,1]

def train_scenario(net, criterion, trainloader, learning_rate=0.01, momentum=0.9):


    # use_gpu = torch.cuda.is_available()


    # if use_gpu:
    #     net = net.cuda()


    optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=momentum)

    for i, data in enumerate(trainloader):
        # get the inputs
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        outputs = net(inputs)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    return net

def predict_scenario(net, criterion, testloader):
    total_loss = 0
    for i, data in enumerate(testloader):
        inputs, labels = data

        predictions = net(inputs)
        print("predictions: {} labels: {}".format(predictions, labels))
        loss = criterion(predictions, labels)
        total_loss+=loss
    print(i)
    return total_loss/(i+1)