import torch
import torch.nn as  nn
import torch.nn.functional as F
try:
  from timm.models.layers.norm_act import BatchNormAct2d
except:
  from timm.models.layers import BatchNormAct2d

class PatchEmbed(nn.Module):
  def __init__(self, image_size, embed_dims=512):
    super().__init__()
    self.image_size = image_size

    self.patch_size = (image_size//4, image_size//2)
    self.n_patches = (image_size // self.patch_size[0])*((image_size // self.patch_size[1]))


    # convolutional layer that does both the splitting into patches and their embedding 
    self.cv2D_layer = nn.Conv2d(3, embed_dims, kernel_size=self.patch_size, stride=self.patch_size)
  
  def forward(self, x):
    '''
      input: X tensor - Shape: (n_samples, in_chans, image_size, image_size).
      output: tensor - Shape: (n_samples, n_patches, embed_dim)
    '''
    x = self.cv2D_layer(x) # Shape: (n_samples, embed_dims, patch_size // 2, patch_size // 2)
    x = x.flatten(2) # Shape: (n_samples, embed_dims, n_patches)
    x = x.transpose(1, 2) # Shape: (n_samples, n_patches, embed_dims)

    return x

class Get_Scalar:
    def __init__(self, value):
        self.value = value

    def get_value(self, iter):
        return self.value

    def __call__(self, iter):
        return self.value

class FeatureExtractor(nn.Module):
    def __init__(self, embed_dims):
        super().__init__()   

        self.conv_stem = nn.Conv2d(3, embed_dims, kernel_size=(3, 3), stride=(1, 2), padding=(1, 1), bias=False)
        self.bn_conv = BatchNormAct2d(embed_dims, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True,act_layer = nn.SiLU)
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(embed_dims, embed_dims)
        
    def forward(self, x):
        x = self.conv_stem(x)
        x = self.bn_conv(x)

        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)

        return x