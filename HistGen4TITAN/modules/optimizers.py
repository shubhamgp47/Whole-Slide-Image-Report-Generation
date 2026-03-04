import torch

def build_optimizer(args, model):
    # Check if using TITAN embeddings (no visual extractor to train)
    if hasattr(args, 'use_titan_embeddings') and args.use_titan_embeddings:
        # TITAN version - only train decoder and projection layers
        optimizer = getattr(torch.optim, args.optim)(
            model.parameters(),  # All trainable parameters (projection + decoder)
            lr=args.lr_ed,       # Single learning rate
            weight_decay=args.weight_decay,
            amsgrad=args.amsgrad
        )
    else:
        # Original HistGen version - separate learning rates for visual encoder and decoder
        ve_params = list(map(id, model.visual_extractor.parameters()))
        ed_params = filter(lambda x: id(x) not in ve_params, model.parameters())
        optimizer = getattr(torch.optim, args.optim)(
            [{'params': model.visual_extractor.parameters(), 'lr': args.lr_ve},
             {'params': ed_params, 'lr': args.lr_ed}],
            weight_decay=args.weight_decay,
            amsgrad=args.amsgrad
        )
    return optimizer

def build_lr_scheduler(args, optimizer):
    lr_scheduler = getattr(torch.optim.lr_scheduler, args.lr_scheduler)(optimizer, args.step_size, args.gamma)
    return lr_scheduler
