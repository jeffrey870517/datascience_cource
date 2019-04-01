import torch
from torch import nn
from  torch.autograd import Variable
import  matplotlib.pyplot as plt
import csv

EPOCH = 40000

def open_csv(file,rowname):
    with open(file) as power:
        rows = csv.DictReader(power)
        list = []
        for row in rows:
            data1 = float(row[rowname])
            list.append(data1)
        return  list
x = []
y = []
test_x = []
test_y = []

taipei_forecast = [21,23,27,25,27,30,25]
taichung_forecast = [26,26,29,29,31,33,32]
kaohsiung_forecast = [27,28,29,30,30,30,30]
haulien_forecast = [22,25,27,27,27,27,26]
forecast = []

power = open_csv('power_datasets.csv','max_power')
taipei_max = open_csv('taipei.csv','body_temp')
taichung_max = open_csv('taichung.csv','body_temp')
kaohsiung_max = open_csv('kaohsiung.csv','body_temp')
haulien_max = open_csv('haulien.csv','body_temp')

for count in range(len(taipei_max)):
    if (count % 10 == 0):
        data = float(0.38 * taipei_max[count] + 0.28 * taichung_max[count] + 0.33 * kaohsiung_max[count] + 0.01 * haulien_max[count])
        test_x.append([data])
        data2 = [float(power[count])]
        test_y.append(data2)
    else:
        data = float(0.38 * taipei_max[count] + 0.28 * taichung_max[count] + 0.33 * kaohsiung_max[count] + 0.01 * haulien_max[count])
        x.append([data])
        data2 = [float(power[count])]
        y.append(data2)

for i in range(len(taipei_forecast)):
    data = float(0.38 * taipei_forecast[i] + 0.28 * taichung_forecast[i] + 0.33 * kaohsiung_forecast[i] + 0.01 * haulien_forecast[i])
    forecast.append([data])

x, y,forecast = torch.Tensor(x), torch.Tensor(y), torch.Tensor(forecast)
x, y,forecast = Variable(x), Variable(y),Variable(forecast).cuda()  # noisy y data (tensor), shape=(100, 1)
test_x, test_y = torch.Tensor(test_x), torch.Tensor(test_y)
test_x, test_y = Variable(test_x), Variable(test_y)
test_x= test_x.cuda()

x=x.cuda()
y=y.cuda()

net = torch.nn.Sequential(
    torch.nn.Linear(1, 15),
    torch.nn.ReLU(),
    torch.nn.Linear(15, 5),
    torch.nn.ReLU(),
    torch.nn.Linear(5, 1),
)
net.cuda()

optimizer = torch.optim.RMSprop(net.parameters(), lr=0.5)
loss_func = nn.MSELoss()

for epoch in range(EPOCH):

    prediction = net(x)
    loss = loss_func(prediction, y)     # must be (1. nn output, 2. target)

    optimizer.zero_grad()   # clear gradients for next train
    loss.backward()         # backpropagation, compute gradients
    optimizer.step()        # apply gradients
    if epoch %1000 == 0:
        time=[]
        test_output = net(test_x)  #
        test_output = test_output.cpu()
        pred_y = torch.Tensor(torch.max(test_output, 1)[0].data.numpy()).squeeze()
        index = 0
        for index in range(pred_y.size(0)):
            num = ((pred_y[index]-test_y[index])/test_y[index] <= 0.01)
            time.append(num)
        time = torch.Tensor(time)
        accuracy = float(torch.sum(time)) / float(test_y.size(0))
        print('Epoch: ', epoch, '| train loss: %.4f' % loss.data.cpu().numpy(), '| test accuracy: %.2f' % accuracy)

output = net(forecast).cpu()
output = output.tolist()
with open('submission1.csv','w',newline='') as csv_sub:
    writer = csv.writer(csv_sub)
    writer.writerow(['date','peak_load(MW)'])
    writer.writerow(['20180402', '%.1f' % output[0][0]])
    writer.writerow(['20180403', '%.1f' % output[1][0]])
    writer.writerow(['20180404', '%.1f' % output[2][0]])
    writer.writerow(['20180405', '%.1f' % output[3][0]])
    writer.writerow(['20180406', '%.1f' % output[4][0]])
    writer.writerow(['20180407', '%.1f' % output[5][0]])
    writer.writerow(['20180408', '%.1f' % output[6][0]])
    writer.writerow(['accuracy', '%.2f' % accuracy])

