import supporting_functions as sf

import torch
import torch.nn as nn
import numpy
import matplotlib.pyplot as plt
import torch.nn.functional as F

# correct for scale difference neural network and expected range

if __name__ == '__main__':
    # freeze_support()
    accuracies = []
    runs = 1
    lower_boundary=0.85
    upper_boundary=1.5
    learning_rate=0.5
    x_values = sf.uniform_random_n(lower_boundary, upper_boundary, 300)
    file_path = 'data/pig.csv'
    true_data_epochs = 100
    dataloader_params = {
        'batch_size': 1,
        'shuffle': True,
        'num_workers': 1
    }

    predictions = []


    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()

            self.fc1 = nn.Linear(1, 10)
            self.fc2 = nn.Linear(10, 10)
            self.fc3 = nn.Linear(10, 1)


        def forward(self, x):
            x = x.view(-1, 1)
            x = F.logsigmoid(self.fc1(x))
            x = F.logsigmoid(self.fc2(x))
            x = self.fc3(x)
            return x

    for i in range(runs):
        trainloader, testloader = sf.datasets_generator(x_values, sf.get_generalized_approximation(x_values), 0.8, dataloader_params)


        # todo consider initializing to the mean of trainloader. and reinitializing for realtest
        criterion = nn.MSELoss()
        # pretraining on artificial data
        net = sf.train_scenario(Net(), criterion, trainloader, learning_rate=learning_rate)

        realTrainLoader, realTestLoader = sf.realset_generator(file_path, 0.2, dataloader_params, lower_boundary, upper_boundary)

        scale_factor = sf.get_scaling_factor(realTrainLoader)
        scaled_real_trainLoader = sf.scale(realTrainLoader, scale_factor)
        scaled_real_testloader = sf.scale(realTestLoader, scale_factor)

        # criterion = nn.MSELoss()
        # for i in range(true_data_epochs):
        #
        #     sf.train_scenario(Net(), criterion, scaled_real_trainLoader)

        print("scaled factor: {}".format(scale_factor))
        print(scaled_real_testloader.labels)
        # todo i might be doubly dividing the error  by count, ensure  this isn't the case
        # prediction_criterion = nn.MSELoss()
        loss, count, predictions = sf.predict_scenario(net, criterion, scaled_real_testloader)

        print("total loss is: {} and there are {} objects".format(loss, count))
        print("average loss is: {}".format(loss / count))
        accuracies.append(float(loss / count))

        # print(predictions)
        predictions = predictions.detach().numpy()
        x_values = torch.tensor(scaled_real_testloader.list_IDs).detach().numpy()
        true_y_values = torch.tensor(scaled_real_testloader.labels).detach().numpy()
        # print(x_values)

        indexes = numpy.argsort(x_values)
        sorted_x_values = []
        sorted_predictions = []
        sorted_true_y_values = []
        for index in indexes:
            sorted_x_values.append(x_values[index])
            sorted_predictions.append(predictions[index])
            sorted_true_y_values.append(true_y_values[index])

        print(list(zip(sorted_x_values, sorted_predictions)))



        plt.plot(sorted_x_values, sorted_predictions, label="predicted values")
        plt.plot(sorted_x_values, sorted_true_y_values, label="true data")
        plt.plot(sorted_x_values, sf.get_generalized_approximation(numpy.array(sorted_x_values)), label="generalized function")
        # plt.plot(sorted_x_values, )
        plt.xlabel("Range [A]")
        plt.ylabel("force deviation from mean [eV]")
        plt.legend()
        plt.show()

    print("average loss over {} runs is {}".format(runs, numpy.mean(accuracies)))






