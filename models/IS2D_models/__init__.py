from .fcn import *

def IS2D_model(model_name, image_size, num_channels, num_classes) :
    if model_name == 'FCN':
        from models.IS2D_models.fcn import fcn8s
        return fcn8s(num_classes)
    elif model_name == 'UNet':
        from models.IS2D_models.unet import Unet
        return Unet(num_channels, num_classes, 64)
    elif model_name == 'UNet++' :
        from models.IS2D_models.unetplusplus import NestedUNet
        return NestedUNet(num_channels)
    elif model_name == 'DeepLabV3+' :
        from models.IS2D_models.deeplabv3plus import DeepLabv3_plus
        return DeepLabv3_plus(nInputChannels=num_channels, n_classes=num_classes)