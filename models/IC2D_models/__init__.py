from .vgg import *
from .resnet import *
from .densenet import *
from .wideresnet import *

def IC2D_model(model_name, linear_node, image_size, num_channels, num_classes) :
    if model_name == 'VGG11' :
        from models.IC2D_models.vgg import vgg11
        return vgg11(image_size, linear_node, num_channels, num_classes)
    elif model_name == 'VGG13' :
        from models.IC2D_models.vgg import vgg13
        return vgg13(image_size, linear_node, num_channels, num_classes)
    elif model_name == 'VGG16' :
        from models.IC2D_models.vgg import vgg16
        return vgg16(image_size, linear_node, num_channels, num_classes)
    elif model_name == 'VGG19' :
        from models.IC2D_models.vgg import vgg19
        return vgg19(image_size, linear_node, num_channels, num_classes)
    elif model_name == 'GooLeNet':
        from models.IC2D_models.googlenet import GoogLeNet
        return GoogLeNet(image_size, num_channels, num_classes)
    elif model_name == 'ResNet_18':
        from models.IC2D_models.resnet import resnet18
        return resnet18(num_classes, num_channels)
    elif model_name == 'ResNet_34':
        from models.IC2D_models.resnet import resnet34
        return resnet34(num_classes, num_channels)
    elif model_name == 'ResNet_50':
        from models.IC2D_models.resnet import resnet50
        return resnet50(num_classes, num_channels)
    elif model_name == 'ResNet_101':
        from models.IC2D_models.resnet import resnet101
        return resnet101(num_classes, num_channels)
    elif model_name == 'DenseNet_121':
        from models.IC2D_models.densenet import densenet121
        return densenet121(num_classes, num_channels)
    elif model_name == 'WRN_40_2' :
        from models.IC2D_models import wideresnet
        return wideresnet(num_classes, num_channels, depth=40, widen_factor=2)
    elif model_name == 'WRN_28_10' :
        from models.IC2D_models import wideresnet
        return wideresnet(num_classes, num_channels, depth=28, widen_factor=10)