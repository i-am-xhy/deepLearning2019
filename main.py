import supporting_functions as sf

import torch
import torch.nn as nn

# correct for scale difference neural network and expected range

if __name__ == '__main__':
    # freeze_support()

        # for line in csv_reader:
        #     print(line)

        x_values = sf.uniform_random_n(0.85, 3.5, 3000)
        # print(x_values)
        # print(sf.get_generalized_approximation(x_values))

        dataloader_params = {
                'batch_size': 1,
                'shuffle': True,
                'num_workers': 1
            }

        class Net(nn.Module):
            def __init__(self):
                super(Net, self).__init__()

                self.fc3 = nn.Linear(1, 1)

            def forward(self, x):
                x = x.view(-1, 1)
                x = self.fc3(x)
                return x

        trainloader, testloader = sf.datasets_generator(x_values, sf.get_generalized_approximation(x_values), 0.5, dataloader_params)
        # todo consider initializing to the mean of trainloader. and reinitializing for realtest
        criterion = nn.MSELoss()
        # pretraining on artificial data
        net = sf.train_scenario(Net(), criterion, trainloader)
        # todo training on real data
        realTestLoader = sf.realset_generator('data/pig.csv', dataloader_params)
        print("total loss is: {}".format(sf.predict_scenario(net, criterion, realTestLoader)))






