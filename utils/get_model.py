import torch
from torch.nn.parallel import DistributedDataParallel as DDP


def save_model_v2(args, i_epoch, model, optimizer, scheduler):
    checkpoint = {"epoch": i_epoch,
                  "state_dict": model.module.state_dict(),
                  "optimizer": optimizer.state_dict(),
                  "scheduler": scheduler.state_dict()}
    torch.save(checkpoint, f"{args.save_path}/epoch{i_epoch}.pt")


def get_model_v2(args):
    from model.forecasting_tgn import Forecasting_TGN
    rank = args.device

    # loading/creating model
    model = Forecasting_TGN(args)
    if args.pretrained_vectornet:
        model_dict = model.state_dict()
        pretrained_dict = torch.load(args.pretrain_vectornet_path)
        # pretrained_dict = pretrained_dict["state_dict"]
        for name, param in pretrained_dict.items():
            if name.startswith("vectornet"):
                model_dict[name].copy_(param)

        model.load_state_dict(model_dict)

    if args.load_epoch is not None:
        assert args.pretrain_path is not None
        path = f"{args.pretrain_path}/epoch{args.load_epoch}.pt"
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint["state_dict"], strict=False)

    model.to(rank)
    model = DDP(model, device_ids=[rank], find_unused_parameters=True)

    if (not args.test) and (not args.validate):
        # loading/creating optimizer
        optimizer = torch.optim.Adam(
            model.parameters(), lr=args.learning_rate)
        if args.load_epoch is not None:
            assert args.pretrain_path is not None
            path = f"{args.pretrain_path}/epoch{args.load_epoch}.pt"
            checkpoint = torch.load(path)
            optimizer.load_state_dict(checkpoint["optimizer"])

        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[3218*24, 3218*30], gamma=0.15)

        if args.load_epoch is not None:
            assert args.pretrain_path is not None
            path = f"{args.pretrain_path}/epoch{args.load_epoch}.pt"
            checkpoint = torch.load(path)
            scheduler.load_state_dict(checkpoint["scheduler"])

        # set start epoch
        if args.load_epoch is not None:
            start_epoch = args.load_epoch + 1
        else:
            start_epoch = 0

    elif args.test or args.validate:
        optimizer = None
        start_epoch = None
        scheduler = None

    return model, optimizer, start_epoch, scheduler
