def freeze_except_context(model):
    for name, param in model.named_parameters():
        if 'fc_context' not in name:
            param.requires_grad = False


def unfreeze_all(model):
    for param in model.parameters():
        param.requires_grad = True
