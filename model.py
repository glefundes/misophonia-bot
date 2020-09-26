import math
import torch
import torch.nn as nn

import torchaudio

from melspec_strech import MelspectrogramStretch

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ELU(),
        )

    def forward(self, x):
        x = self.layers(x)
        return x

class MisophoniaModel(nn.Module):

    def __init__(self, stereo=True, dropout=0.1):
        super().__init__()
        in_channels = 2 if stereo else 1
        self.spec = MelspectrogramStretch(hop_length=None, 
                                                                num_mels=128, 
                                                                fft_length=2048, 
                                                                norm='whiten', 
                                                                stretch_param=[0.4, 0.4])
        
        self.features = nn.Sequential(*[
            ConvBlock(in_channels=2, out_channels=32, kernel_size=3, stride=1),
            nn.MaxPool2d(3,3),
            nn.Dropout(p=dropout),
            ConvBlock(in_channels=32, out_channels=64, kernel_size=3, stride=1),
            nn.MaxPool2d(4,4),
            nn.Dropout(p=dropout),
            ConvBlock(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.MaxPool2d(4,4),
            nn.Dropout(p=dropout),
        ])
        self.min_len = 80896
        self.rnn = nn.GRU(128, 64, num_layers=2) 
        self.ret = nn.Sequential(*[nn.Linear(64,1), nn.Sigmoid()])
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.shape[0] * m.weight.shape[1]
                m.weight.data.normal_(0, math.sqrt(2./n))
                m.bias.data.zero_()
                
  
    
    def modify_lengths(self, lengths):
        def safe_param(elem):
            return elem if isinstance(elem, int) else elem[0]

        for name, layer in self.features.named_children():
            if isinstance(layer, (nn.Conv2d, nn.MaxPool2d)):
                p, k, s = map(safe_param, [layer.padding, layer.kernel_size,layer.stride]) 
                lengths = ((lengths + 2*p - k)//s + 1).long()

        return torch.where(lengths > 0, lengths, torch.tensor(1, device=lengths.device))

    def _many_to_one(self, t, lengths):
        return t[torch.arange(t.size(0)), lengths - 1]

    def forward(self, wave, lengths):
        x = wave
        raw_lengths = lengths
        xt = x.float().transpose(1,2)
        xt, lengths = self.spec(xt, raw_lengths)
        xt = self.features(xt)
        lengths = self.modify_lengths(lengths)
        x = xt.transpose(1, -1)
        batch, time = x.size()[:2]
        x = x.reshape(batch, time, -1)
        lengths = lengths.clamp(max=x.shape[1])
        x_pack = torch.nn.utils.rnn.pack_padded_sequence(x, lengths.clamp(max=x.shape[1]), batch_first=True)
        
        x_pack, self.hidden = self.rnn(x_pack)
        x, _ = torch.nn.utils.rnn.pad_packed_sequence(x_pack, batch_first=True)
        x = self._many_to_one(x, lengths)
        x = self.ret(x)
        return x

    def init_hidden(self, batch_size=1):
        return torch.zeros(2, batch_size, 64)

    def predict(self, audio_file):
        self.hidden = self.init_hidden()
        waveform, sample_rate = torchaudio.load(audio_file)
        # Is audio silent?
        if waveform.var() < 1e-6:
            return -1
        # Normalize waveform
        waveform = normalize_waveform(waveform).permute(1,0)
        # Pad waveform if shortan than the minimum length acceptable for the model to handle
        if waveform.shape[0] < self.min_len:
            padded = torch.zeros(self.min_len, waveform.shape[1])
            padded[:waveform.shape[0], :] = waveform            
            waveform = padded
        # Run inference
        length = torch.tensor(waveform.shape[0])
        with torch.no_grad():
            pred = self(waveform.unsqueeze(0), length.unsqueeze(0))
        pred = pred.squeeze().item()
        return pred
        
def normalize_waveform(tensor):
    if tensor.shape[0] == 1:
        tensor = tensor.expand(2, tensor.shape[1])
    # Subtract the mean, and scale to the interval [-1,1]
    tensor_minusmean = tensor - tensor.mean()
    return tensor_minusmean/tensor_minusmean.abs().max()