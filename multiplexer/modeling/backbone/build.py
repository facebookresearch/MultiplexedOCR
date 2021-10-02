from multiplexer.utils.registry import Registry

BACKBONE_REGISTRY = Registry("BACKBONE")


# @registry.BACKBONES.register("R-18-FPN")
# @registry.BACKBONES.register("R-34-FPN")
# @registry.BACKBONES.register("R-50-FPN")
# @registry.BACKBONES.register("R-101-FPN")
# @registry.BACKBONES.register("R-152-FPN")
# def build_resnet_fpn_backbone(cfg):
#     if cfg.MODEL.RESNET34:
#         from . import resnet34 as resnet

#         body = resnet.ResNet(layers=cfg.MODEL.RESNETS.LAYERS)
#     else:
#         from . import resnet

#         body = resnet.ResNet(cfg)
#     in_channels_stage2 = cfg.MODEL.RESNETS.RES2_OUT_CHANNELS
#     out_channels = cfg.MODEL.RESNETS.BACKBONE_OUT_CHANNELS
#     fpn = fpn_module.FPN(
#         in_channels_list=[
#             in_channels_stage2,
#             in_channels_stage2 * 2,
#             in_channels_stage2 * 4,
#             in_channels_stage2 * 8,
#         ],
#         out_channels=out_channels,
#         conv_block=conv_with_kaiming_uniform(
#             cfg.MODEL.FPN.USE_GN, cfg.MODEL.FPN.USE_RELU
#         ),
#         top_blocks=fpn_module.LastLevelMaxPool(),
#     )
#     model = nn.Sequential(OrderedDict([("body", body), ("fpn", fpn)]))
#     model.out_channels = out_channels
#     return model


# @registry.BACKBONES.register("R-50-FPN-RETINANET")
# @registry.BACKBONES.register("R-101-FPN-RETINANET")
# def build_resnet_fpn_p3p7_backbone(cfg):
#     from . import resnet

#     body = resnet.ResNet(cfg)
#     in_channels_stage2 = cfg.MODEL.RESNETS.RES2_OUT_CHANNELS
#     out_channels = cfg.MODEL.RESNETS.BACKBONE_OUT_CHANNELS
#     in_channels_p6p7 = (
#         in_channels_stage2 * 8 if cfg.MODEL.RETINANET.USE_C5 else out_channels
#     )
#     fpn = fpn_module.FPN(
#         in_channels_list=[
#             0,
#             in_channels_stage2 * 2,
#             in_channels_stage2 * 4,
#             in_channels_stage2 * 8,
#         ],
#         out_channels=out_channels,
#         conv_block=conv_with_kaiming_uniform(
#             cfg.MODEL.FPN.USE_GN, cfg.MODEL.FPN.USE_RELU
#         ),
#         top_blocks=fpn_module.LastLevelP6P7(in_channels_p6p7, out_channels),
#     )
#     model = nn.Sequential(OrderedDict([("body", body), ("fpn", fpn)]))
#     model.out_channels = out_channels
#     return model


def build_backbone(cfg):
    """
    Build a backbone from `cfg.MODEL.BACKBONE.NAME`.
    Returns:
        an instance of backbone
    """

    backbone = BACKBONE_REGISTRY.get(cfg.MODEL.BACKBONE.NAME)(cfg)
    if cfg.MODEL.BACKBONE.FROZEN:
        for p in backbone.parameters():
            p.requires_grad = False
    return backbone
