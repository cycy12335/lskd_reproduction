from .resnet import resnet8x4, resnet32x4

def build_model(name):
    return {"resnet8x4": resnet8x4, "resnet32x4": resnet32x4}[name](num_classes=100)
