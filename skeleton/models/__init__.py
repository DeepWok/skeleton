from .lenet5 import LeNet5
from .cifarnet import CifarNet
# from .mobilenets import (
#     MobileNetV2, MobileNetV3, MobileNetV3Small,
#     MobileNetV3Search, MobileNetV3SearchSmall)
# from .mnasnet import MNasNet
# from .resnet import (
#     ResNet110a, ResNet110b, ResearchNet56, WideResearchNet56, 
#     SmallResearchNet56, ResNet56, ResNet32, ResNet44, ResNet18, ResNet34)
# from .wideresnet import WideResearchNet28, WideResearchNet40


factory = {
    'lenet5': LeNet5,
    'cifarnet': CifarNet,
}
#     'fbnet': FBNet,
#     'fbnetsmall': FBNetSmall,
#     'finefbnet': FineFBNet,
#     'finefbnetsmall': FineFBNetSmall,
#     'shufflenet': ShuffleNet,
#     'efficientnet': EfficientNet,
#     'efficientnetsmall': EfficientNetSmall,
#     'efficientnetb0': EfficientNetB0,
#     'mobilenetv2': MobileNetV2,
#     'mobilenetv3': MobileNetV3,
#     'robnetfree': RobNetFree,
#     'robnetlargev1': RobNetLargeV1,
#     'robnetlargev2': RobNetLargeV2,
#     'mobilenetv3small': MobileNetV3Small,
#     'mobilenetv3search': MobileNetV3Search,
#     'mobilenetv3searchsmall': MobileNetV3SearchSmall,
#     'mnasnet': MNasNet,
#     'resnet18': ResNet18,
#     'resnet32': ResNet32,
#     'resnet34': ResNet34,
#     'resnet44': ResNet44,
#     'resnet56': ResNet56,
#     'resnet110a': ResNet110a,
#     'resnet110b': ResNet110b,
#     'researchnet56': ResearchNet56,
#     'wideresearchnet56': WideResearchNet56,
#     'smallresearchnet56': SmallResearchNet56,
#     'wideresearchnet28-10': WideResearchNet28,
#     'wideresearchnet40-4': WideResearchNet40,
# }
