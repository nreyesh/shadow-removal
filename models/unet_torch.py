import torch
from torch import nn

class conv_block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv2d_1 = nn.Conv2d(in_channels=in_channels,
                                 out_channels=out_channels,
                                 kernel_size=(3,3),
                                 stride=1,
                                 padding=1)
        self.relu_1 = nn.ReLU()
        self.conv2d_2 = nn.Conv2d(in_channels=out_channels,
                                 out_channels=out_channels,
                                 kernel_size=(3,3),
                                 stride=1,
                                 padding=1)
        self.relu_2 = nn.ReLU()
        
    def forward(self, x):
        return self.relu_2(self.conv2d_2(self.relu_1(self.conv2d_1(x))))   


class encoder_block(nn.Module):
    def __init__(self, in_features=[3,64,128,256,512]):
        super().__init__()
        self.encBlock = nn.ModuleList([conv_block(in_features[x],
                                                  in_features[x+1]) 
                                       for x in range(len(in_features)-1)])
        self.maxpool = nn.MaxPool2d(kernel_size=(2, 2), 
                                    stride=2, 
                                    padding=0)

    def forward(self, x):
        encBlock_out = []
        for block in self.encBlock:
            x = block(x)
            encBlock_out.append(x)
            x = self.maxpool(x)
        return encBlock_out
        

class decoder_block(nn.Module):
    def __init__(self, in_features=[512,256,128,64]):
        super().__init__()
        self.channels = in_features
        self.decBlock = nn.ModuleList([conv_block(in_features[x],
                                                  in_features[x+1]) 
                                       for x in range(len(in_features)-1)])
        self.upscaling = nn.ModuleList(nn.ConvTranspose2d(in_features[x],
                                                  in_features[x+1],
                                            kernel_size=(3,3),
                                            stride=2,
                                            padding=1,
                                            output_padding=1) for x in range(len(in_features)-1))

    def forward(self, x, encFeatures):
        for i in range(len(self.channels)-1):
            #print(f'[INFO]: Iter #{i}')
            x = self.upscaling[i](x)
            #print(f'Shape Up: {x.shape}')
            #print(f'Shape encFeat: {encFeatures[i].shape}')
            x = torch.cat([x,encFeatures[i]],dim=1)
            #print(f'Shape Concatenation: {x.shape}')
            x = self.decBlock[i](x)                             
        return x
        

class Unet(nn.Module):
    def __init__(self,enc_channels=[3,64,128,256,512],
                    dec_channels=[512,256,128,64],
                    n_classes = 1,
                    out_size=(256,256)):
        super().__init__()
        self.encoder = encoder_block(enc_channels)
        self.decoder = decoder_block(dec_channels)

        self.head = nn.Conv2d(dec_channels[-1], n_classes,
                             kernel_size=(3,3), stride=1, padding=1)
        self.out_size = out_size  


    def forward(self, x):
        encFeatures = self.encoder(x)
        
        decFeatures = self.decoder(encFeatures[::-1][0],encFeatures[::-1][1:])
        
        classifier = self.head(decFeatures)
        
        return classifier       