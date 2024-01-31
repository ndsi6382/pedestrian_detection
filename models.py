# COMP9444 Project Models

# This file does not need to be run.

from project_requirements import *

def model_chooser(name):
    if name == "frcnn1":
        backbone = torchvision.models.mobilenet_v2(weights="DEFAULT").features
        backbone.out_channels = 1280
        anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),),aspect_ratios=((0.5, 1.0, 2.0),))
        roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'],output_size=7,sampling_ratio=2)
        model = FasterRCNN(backbone,num_classes=3,rpn_anchor_generator=anchor_generator,box_roi_pool=roi_pooler)
        optimiser = torch.optim.SGD(model.parameters(), lr=0.002, weight_decay=0.002, momentum=0.9)

    elif name == "frcnn1b":
        backbone = torchvision.models.mobilenet_v2().features
        backbone.out_channels = 1280
        anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),),aspect_ratios=((0.5, 1.0, 2.0),))
        roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'],output_size=7,sampling_ratio=2)
        model = FasterRCNN(backbone,num_classes=3,rpn_anchor_generator=anchor_generator,box_roi_pool=roi_pooler)
        optimiser = torch.optim.SGD(model.parameters(), lr=0.002, weight_decay=0.002, momentum=0.9)

    elif name == "frcnn2":
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 3)
        optimiser = torch.optim.SGD(model.parameters(), lr=0.002, weight_decay = 0.001, momentum = 0.9)

    elif name == "frcnn2b":
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn()
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 3)
        optimiser = torch.optim.SGD(model.parameters(), lr=0.002, weight_decay = 0.001, momentum = 0.9)

    elif name == "frcnn3":
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(weights="DEFAULT")
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 3)
        optimiser = torch.optim.SGD(model.parameters(), lr=0.001, weight_decay = 0.01, momentum = 0.9)

    elif name == "frcnn3b":
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2()
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 3)
        optimiser = torch.optim.SGD(model.parameters(), lr=0.001, weight_decay = 0.01, momentum = 0.9)

    elif name == "frcnn4":
        model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(weights="DEFAULT")
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 3)
        optimiser = torch.optim.SGD(model.parameters(), lr=0.002, weight_decay = 0.002, momentum = 0.9)

    elif name == "frcnn4b":
        model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn()
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 3)
        optimiser = torch.optim.SGD(model.parameters(), lr=0.002, weight_decay = 0.002, momentum = 0.9)

    return (model, optimiser)