import torch
import torch.nn as nn
import torch.nn.functional as F
def MC_Dropout(x, p=0.5, sample=True):
    return F.dropout(x, p=p, training=sample, inplace=False)


# Residual Block
class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=True, pooling=False, MCDropout=True):
        super(ResBlock, self).__init__()
        self.MCDropout = MCDropout
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ELU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.maxpool = nn.MaxPool2d(2, stride=2)
        self.downsample = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channels)
        )
        self.downsampleOrNot = downsample
        assert downsample
        self.pooling = pooling
        self.dropout = nn.Dropout(0.5)

    def forward(self, x, to_sample=True):
        maybe_dropout = lambda x_: MC_Dropout(x_, sample=self.training or to_sample) if self.MCDropout else x_
        out = self.conv1(maybe_dropout(x))
        out = self.bn1(out)
        out = self.relu(out)
        # out = self.dropout(out)
        out = self.conv2(maybe_dropout(out))
        out = self.bn2(out)
        if self.downsampleOrNot:
            residual = self.downsample(maybe_dropout(x))
            out += residual
        if self.pooling:
            out = self.maxpool(out)
        if not self.MCDropout: out = self.dropout(out)
        return out

class CNNSleep(nn.Module):
    def __init__(self, n_dim, avg_pool=None, base_channels=2,
                 MCDropout=False, sample_input=False,
                 ):
        super(CNNSleep, self).__init__()

        if avg_pool is not None:
            self.avg_pool = torch.nn.AvgPool1d(avg_pool)
        else:
            self.avg_pool = None

        self.MCDropout = MCDropout
        self.sample_input = sample_input
        assert not sample_input
        self.conv1 = nn.Sequential(
            nn.Conv2d(2*base_channels, 3*base_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(3*base_channels),
            nn.ELU(inplace=True),
        )
        self.conv2 = ResBlock(3*base_channels, 4*base_channels, 2, True, False, MCDropout=MCDropout)
        self.conv3 = ResBlock(4*base_channels, 8*base_channels, 2, True, True, MCDropout=MCDropout)
        self.conv4 = ResBlock(8*base_channels, 16*base_channels, 2, True, True, MCDropout=MCDropout)
        self.n_dim = n_dim

        self.sup = nn.Sequential(
            nn.Linear(64*base_channels, 16*base_channels, bias=True),
            nn.ReLU(),
            nn.Linear(16*base_channels, 5, bias=True),
        )


    def torch_stft(self, X_train):
        signal = []

        for s in range(X_train.shape[1]):
            spectral = torch.stft(X_train[:, s, :],
                                  n_fft=256,
                                  hop_length=256 * 1 // 4,
                                  center=False,
                                  onesided=True)
            signal.append(spectral)

        signal1 = torch.stack(signal)[:, :, :, :, 0].permute(1, 0, 2, 3)
        signal2 = torch.stack(signal)[:, :, :, :, 1].permute(1, 0, 2, 3)

        return torch.cat([torch.log(torch.abs(signal1) + 1e-8), torch.log(torch.abs(signal2) + 1e-8)], dim=1)

    def forward(self, x, to_sample=False):
        maybe_dropout = lambda x_: MC_Dropout(x_, sample=self.training or to_sample) if self.MCDropout else x_
        if self.avg_pool is not None: x = self.avg_pool(x)
        x = self.torch_stft(x)
        if self.sample_input: x = maybe_dropout(x)
        x = self.conv1(x)
        x = self.conv2(x, to_sample=to_sample)
        x = self.conv3(x, to_sample=to_sample)
        x = self.conv4(x, to_sample=to_sample)
        x = x.reshape(x.shape[0], -1)
        return self.sup(maybe_dropout(x))

#=======================================MINA
class KnowledgeAttn(nn.Module):
    def __init__(self, input_features, attn_dim):
        """
        This is the general knowledge-guided attention module.
        It will transform the input and knowledge with 2 linear layers, computes attention, and then aggregate.
        :param input_features: the number of features for each
        :param attn_dim: the number of hidden nodes in the attention mechanism
        """
        super(KnowledgeAttn, self).__init__()
        self.input_features = input_features
        self.attn_dim = attn_dim
        self.n_knowledge = 1

        self.att_W = nn.Linear(self.input_features + self.n_knowledge, self.attn_dim, bias=False)
        self.att_v = nn.Linear(self.attn_dim, 1, bias=False)

        self.init()

    def init(self):
        nn.init.normal_(self.att_W.weight)
        nn.init.normal_(self.att_v.weight)

    @classmethod
    def attention_sum(cls, x, attn):
        """
        :param x: of shape (-1, D, nfeatures)
        :param attn: of shape (-1, D, 1)
        """
        return torch.sum(torch.mul(attn, x), 1)


    def forward(self, x, k):
        """
        :param x: shape of (-1, D, input_features)
        :param k: shape of (-1, D, 1)
        :return:
            out: shape of (-1, input_features), the aggregated x
            attn: shape of (-1, D, 1)
        """
        tmp = torch.cat([x, k], dim=-1)
        e = self.att_v(torch.tanh(self.att_W(tmp)))
        attn = F.softmax(e, 1)
        out = self.attention_sum(x, attn)
        return out, attn

#============================================================

class BeatNet_Dropout(nn.Module):
    #Attention for the CNN step/ beat level/local information
    def __init__(self, n=3000, T=50, conv_out_channels=64):
        """
        :param n: size of each 10-second-demos
        :param T: size of each smaller segment used to capture local information in the CNN stage
        :param conv_out_channels: also called number of filters/kernels
        """
        super(BeatNet_Dropout, self).__init__()
        self.n, self.M, self.T = n, int(n/T), T
        self.conv_out_channels = conv_out_channels
        self.conv_kernel_size = 32
        self.conv_stride = 2
        #Define conv and conv_k, the two Conv1d modules
        self.conv = nn.Conv1d(in_channels=1,
                              out_channels=self.conv_out_channels,
                              kernel_size=self.conv_kernel_size,
                              stride=self.conv_stride)

        self.conv_k = nn.Conv1d(in_channels=1,
                                out_channels=1,
                                kernel_size=self.conv_kernel_size,
                                stride=self.conv_stride)

        self.att_cnn_dim = 8
        #Define attn, the KnowledgeAttn module
        self.attn = KnowledgeAttn(self.conv_out_channels, self.att_cnn_dim)

    def forward(self, x, k_beat):
        """
        :param x: shape (batch, n)
        :param k_beat: shape (batch, n)
        :return:
            out: shape (batch, M, self.conv_out_channels)
            alpha: shape (batch * M, N, 1) where N is a result of convolution
        """
        x = x.view(-1, self.T).unsqueeze(1)
        k_beat = k_beat.view(-1, self.T).unsqueeze(1)
        x = F.relu(self.conv(x))  # Here number of filters K=64
        k_beat = F.relu(self.conv_k(k_beat))  # Conv1d(1, 1, kernel_size=(32,), stride=(2,)) => k_beat:[128*60,1,10].

        x = x.permute(0, 2, 1)  # x:[128*60,10,64]
        k_beat = k_beat.permute(0, 2, 1)
        out, alpha = self.attn(x, k_beat)
        out = out.view(-1, self.M, self.conv_out_channels)
        return out, alpha


class RhythmNet_Dropout(nn.Module):
    def __init__(self, n=3000, T=50, input_size=64, rhythm_out_size=8):
        """
        :param n: size of each 10-second-demos
        :param T: size of each smaller segment used to capture local information in the CNN stage
        :param input_size: This is the same as the # of filters/kernels in the CNN part.
        :param rhythm_out_size: output size of this netowrk
        """
        #input_size is the cnn_out_channels
        super(RhythmNet_Dropout, self).__init__()
        self.n, self.M, self.T = n, int(n/T), T
        self.input_size = input_size

        self.rnn_hidden_size = 32
        ### define lstm: LSTM Input is of shape (batch size, M, input_size)
        self.lstm = nn.LSTM(input_size=self.input_size, #self.conv_out_channels,
                            hidden_size=self.rnn_hidden_size,
                            num_layers=1, batch_first=True, bidirectional=True)

        ### Attention mechanism: define attn to be a KnowledgeAttn
        self.att_rnn_dim = 8
        self.attn = KnowledgeAttn(2 * self.rnn_hidden_size, self.att_rnn_dim)

        ### Define the Dropout and fully connecte layers (fc and do)
        self.out_size = rhythm_out_size
        self.fc = nn.Linear(2 * self.rnn_hidden_size, self.out_size)
        self.do = nn.Dropout(p=0.5)

    def forward(self, x, k_rhythm):
        """
        :param x: shape (batch, M, self.input_size)
        :param k_rhythm: shape (batch, M)
        :return:
            out: shape (batch, self.out_size)
            beta: shape (batch, M, 1)
        """
        maybe_dropout = lambda x_: MC_Dropout(x_, sample=self.training)

        ### reshape for rnn
        k_rhythm = k_rhythm.unsqueeze(-1)  # [128, 60, 1]
        ### rnn
        o, (ht, ct) = self.lstm(x)  # o:[batch,60,64] (in the paper this is called h

        x, beta = self.attn(o, k_rhythm)
        ### fc and Dropout
        x = F.relu(self.fc(maybe_dropout(x)))  # [128, 64->8]
        out = self.do(x)
        return out, beta

class FreqNet_Dropout(nn.Module):
    def __init__(self, n_channels=4, n=3000, T=50, n_class = 4):
        """
        :param n_channels: number of channels (F in the paper). We will need to define this many BeatNet & RhythmNet nets.
        :param n: size of each 10-second-demos
        :param T: size of each smaller segment used to capture local information in the CNN stage
        """
        super(FreqNet_Dropout, self).__init__()
        self.n, self.M, self.T = n, int(n / T), T
        self.n_class = n_class
        self.n_channels = n_channels
        self.conv_out_channels=64
        self.rhythm_out_size=8

        self.beat_nets = nn.ModuleList()
        self.rhythm_nets = nn.ModuleList()
        #use self.beat_nets.append() and self.rhythm_nets.append() to append 4 BeatNets/RhythmNets
        for channel_i in range(self.n_channels):
            self.beat_nets.append(BeatNet_Dropout(self.n, self.T, self.conv_out_channels))
            self.rhythm_nets.append(RhythmNet_Dropout(self.n, self.T, self.conv_out_channels, self.rhythm_out_size))


        self.att_channel_dim = 2
        ### Add the frequency attention module using KnowledgeAttn (attn)
        self.attn = KnowledgeAttn(self.rhythm_out_size, self.att_channel_dim)

        ### Create the fully-connected output layer (fc)
        self.fc = nn.Linear(self.rhythm_out_size, self.n_class)


    def forward(self, data_tuple_):
        """
        :param x: shape (n_channels, batch, n)
        :param k_beats: (n_channels, batch, n)
        :param k_rhythms: (n_channels, batch, M)
        :param k_freq: (n_channels, batch, 1)
        :return:
            out: softmax output for each demos point, shpae (batch, n_class)
            gama: the attention value on channels
        """
        maybe_dropout = lambda x_: MC_Dropout(x_, sample=self.training)
        x, k_beats, k_rhythms, k_freq = data_tuple_
        new_x = [None for _ in range(self.n_channels)]
        for i in range(self.n_channels):
            tx, _ = self.beat_nets[i](x[i], k_beats[i])
            new_x[i], _ = self.rhythm_nets[i](tx, k_rhythms[i])
        x = torch.stack(new_x, 1)  # [128,8] -> [128,4,8]

        # ### attention on channel
        k_freq = k_freq.permute(1, 0, 2) #[4,128,1] -> [128,4,1]
        x, gama = self.attn(x, k_freq)

        ### fc
        #out = F.softmax(self.fc(x), 1) #CrossEntropy expects unnormalized scores.
        out = self.fc(maybe_dropout(x))
        return out