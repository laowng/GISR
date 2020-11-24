import numpy as np
import torch
from torch import nn

import matplotlib.pyplot as plt

"""
G Yonv1943 Zen4 Jia1 hao2
https://github.com/Yonv1943/DL_RL_Zoo/blob/master/RNN
The source of training data 
https://github.com/L1aoXingyu/
code-of-learn-deep-learning-with-pytorch/blob/master/
chapter5_RNN/time-series/lstm-time-series.ipynb
"""


def run_train_lstm():
    inp_dim = 2
    out_dim = 1
    mid_dim = 8
    mid_layers = 1
    batch_size = 12 * 4
    mod_dir = '.'

    '''load data'''
    data = load_data()
    data_x = data[:, -2:]
    data_y = data[:, 0:1]/400
    assert data_x.shape[1] == inp_dim

    train_size = int(len(data_x) * 0.75)

    train_x = data_x[:train_size-1]
    train_y = data_y[1:train_size]


    '''build model'''
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = RegLSTM(inp_dim, out_dim, mid_dim, mid_layers).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-2)

    '''train'''
    var_x = torch.tensor(train_x, dtype=torch.float32, device=device)
    var_y = torch.tensor(train_y, dtype=torch.float32, device=device)

    batch_var_x=var_x.unsqueeze(1)
    batch_var_y = var_y.unsqueeze(1)



    batch_var_x = list()
    batch_var_y = list()
    # gg=var_x.shape[0]//batch_size
    # yy=var_x.shape[0]%batch_size
    # for i in range(batch_size):
    #     j = i*gg
    #     batch_var_x.append(var_x[j:])
    #     batch_var_y.append(var_y[j:])
    for i in range(batch_size):
        j = train_size - i
        batch_var_x.append(var_x[j:])
        batch_var_y.append(var_y[j:])
    from torch.nn.utils.rnn import pad_sequence
    batch_var_x = pad_sequence(batch_var_x)
    batch_var_y = pad_sequence(batch_var_y)

    print("Training Start")
    for e in range(2048):
        out = net(batch_var_x)

        loss = criterion(out, batch_var_y)
        # loss = (out - batch_var_y) ** 2 * weights
        # loss = loss.mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if e % 64 == 0:
            print('Epoch: {:4}, Loss: {:.5f}'.format(e, loss.item()))
            net = net.eval()

            test_x = data_x.copy()
            test_x = test_x[:-1, np.newaxis, :]
            test_x = torch.tensor(test_x, dtype=torch.float32, device=device)
            pred_y = net(test_x)
            pred_y = pred_y[:, 0, 0].data.numpy()
            test_y=data_y[:,0]
            diff_y = pred_y[train_size:] - test_y[train_size+1:]
            l1_loss = np.mean(np.abs(diff_y))
            l2_loss = np.mean(diff_y ** 2)
            print("L1: {:.3f}    L2: {:.3f}".format(l1_loss, l2_loss))

            plt.clf()
            plt.plot(pred_y, 'r', label='pred')
            plt.plot(test_y, 'b', label='real', alpha=0.3)
            plt.plot([train_size, train_size], [-1, 2], color='k', label='train | pred')
            plt.legend(loc='best')
            plt.savefig('lstm_reg.png')
            plt.pause(2)
    torch.save(net.state_dict(), '{}/net.pth'.format(mod_dir))
    print("Save in:", '{}/net.pth'.format(mod_dir))

    '''eval'''
    net.load_state_dict(torch.load('{}/net.pth'.format(mod_dir), map_location=lambda storage, loc: storage))
    net = net.eval()

    test_x = data_x.copy()
    test_x = test_x[:-1, np.newaxis, :]
    test_x = torch.tensor(test_x, dtype=torch.float32, device=device)
    pred_y = net(test_x)
    pred_y = pred_y[:, 0, 0].data.numpy()
    test_y = data_y[:, 0]
    diff_y = pred_y[train_size:] - test_y[train_size + 1:]
    l1_loss = np.mean(np.abs(diff_y))
    l2_loss = np.mean(diff_y ** 2)
    print("L1: {:.3f}    L2: {:.3f}".format(l1_loss, l2_loss))

    plt.clf()
    plt.plot(pred_y, 'r', label='pred')
    plt.plot(test_y, 'b', label='real', alpha=0.3)
    plt.plot([train_size, train_size], [-1, 2], color='k', label='train | pred')
    plt.legend(loc='best')
    plt.savefig('lstm_reg.png')
    plt.pause(8)




class RegLSTM(nn.Module):
    def __init__(self, inp_dim, out_dim, mid_dim, mid_layers):
        super(RegLSTM, self).__init__()

        self.rnn = nn.LSTM(inp_dim, mid_dim, 2,bidirectional=False)  # rnn
        self.reg = nn.Sequential(
            nn.Linear(mid_dim, mid_dim),
            nn.Tanh(),
            nn.Linear(mid_dim, out_dim),
        )  # regression

    def forward(self, x):
        y = self.rnn(x)[0]  # y, (h, c) = self.rnn(x)

        seq_len, batch_size, hid_dim = y.shape
        y = y.view(-1, hid_dim)
        y = self.reg(y)
        y = y.view(seq_len, batch_size, -1)
        return y

    """
    PyCharm Crtl+click nn.LSTM() jump to code of PyTorch:
    Examples::
        >>> rnn = nn.LSTM(10, 20, 2)
        >>> input = torch.randn(5, 3, 10)
        >>> h0 = torch.randn(2, 3, 20)
        >>> c0 = torch.randn(2, 3, 20)
        >>> output, (hn, cn) = rnn(input, (h0, c0))
    """

    def output_y_hc(self, x, hc):
        y, hc = self.rnn(x, hc)  # y, (h, c) = self.rnn(x)

        seq_len, batch_size, hid_dim = y.size()
        y = y.view(-1, hid_dim)
        y = self.reg(y)
        y = y.view(seq_len, batch_size, -1)
        return y, hc


class RegGRU(nn.Module):
    def __init__(self, inp_dim, out_dim, mod_dim, mid_layers):
        super(RegGRU, self).__init__()

        self.rnn = nn.GRU(inp_dim, mod_dim, mid_layers)
        self.reg = nn.Linear(mod_dim, out_dim)

    def forward(self, x):
        x, h = self.rnn(x)  # (seq, batch, hidden)

        seq_len, batch_size, hid_dim = x.shape
        x = x.view(-1, hid_dim)
        x = self.reg(x)
        x = x.view(seq_len, batch_size, -1)
        return x

    def output_y_h(self, x, h):
        y, h = self.rnn(x, h)

        seq_len, batch_size, hid_dim = y.size()
        y = y.view(-1, hid_dim)
        y = self.reg(y)
        y = y.view(seq_len, batch_size, -1)
        return y, h


def load_data():
    # passengers number of international airline , 1949-01 ~ 1960-12 per month
    seq_number = np.array(
        [112., 118., 132., 129., 121., 135., 148., 148., 136., 119., 104.,
         118., 115., 126., 141., 135., 125., 149., 170., 170., 158., 133.,
         114., 140., 145., 150., 178., 163., 172., 178., 199., 199., 184.,
         162., 146., 166., 171., 180., 193., 181., 183., 218., 230., 242.,
         209., 191., 172., 194., 196., 196., 236., 235., 229., 243., 264.,
         272., 237., 211., 180., 201., 204., 188., 235., 227., 234., 264.,
         302., 293., 259., 229., 203., 229., 242., 233., 267., 269., 270.,
         315., 364., 347., 312., 274., 237., 278., 284., 277., 317., 313.,
         318., 374., 413., 405., 355., 306., 271., 306., 315., 301., 356.,
         348., 355., 422., 465., 467., 404., 347., 305., 336., 340., 318.,
         362., 348., 363., 435., 491., 505., 404., 359., 310., 337., 360.,
         342., 406., 396., 420., 472., 548., 559., 463., 407., 362., 405.,
         417., 391., 419., 461., 472., 535., 622., 606., 508., 461., 390.,
         432.], dtype=np.float32)
    # assert seq_number.shape == (144, )
    # plt.plot(seq_number)
    # plt.ion()
    # plt.pause(1)
    seq_number = seq_number[:, np.newaxis]

    # print(repr(seq))
    # 1949~1960, 12 years, 12*12==144 month
    seq_year = np.arange(12)
    seq_month = np.arange(12)
    seq_year_month = np.transpose(
        [np.repeat(seq_year, len(seq_month)),
         np.tile(seq_month, len(seq_year))],
    )  # Cartesian Product

    seq = np.concatenate((seq_number, seq_year_month), axis=1)

    # normalization
    # seq = (seq - seq.mean(axis=0)) / seq.std(axis=0)
    return seq


if __name__ == '__main__':
    run_train_lstm()
    # run_train_gru()
    # run_origin()
    # Training
    # Start
    # Epoch: 0, Loss: 0.99855
    # Epoch: 64, Loss: 0.02532
    # Epoch: 128, Loss: 0.00725
    # Epoch: 192, Loss: 0.00326
    # Epoch: 256, Loss: 0.00243
    # Epoch: 320, Loss: 0.00208
    # Save in:./ net.pth
    # L1: 0.258
    # L2: 0.121
    # Training
    # Start
    # Epoch: 0, Loss: 0.17629
    # Epoch: 64, Loss: 0.01222
    # Epoch: 128, Loss: 0.00410
    # Epoch: 192, Loss: 0.00248
    # Epoch: 256, Loss: 0.00193
    # Epoch: 320, Loss: 0.00159
    # Save in:./ net.pth
    # L1: 0.217
    # L2: 0.071
