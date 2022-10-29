from models import resnet, preactresnet, lenoresnet

def build_model(args):

    if args.arch == 'resnet':
        assert args.model_depth in [50, 101, 152, 200]

        if args.model_depth == 50:
            model = resnet.ResNet50(
                num_classes=args.n_classes,
                zero_init_residual=args.zero_init_residual)
        elif args.model_depth == 101:
            model = resnet.ResNet101(
                num_classes=args.n_classes,
                zero_init_residual=args.zero_init_residual)
        elif args.model_depth == 152:
            model = resnet.ResNet152(
                num_classes=args.n_classes,
                zero_init_residual=args.zero_init_residual)
        elif args.model_depth == 200:
            model = resnet.ResNet200(
                num_classes=args.n_classes,
                zero_init_residual=args.zero_init_residual)
   
    if args.arch == 'preactresnet':
        assert args.model_depth in [50, 101, 152, 200]

        if args.model_depth == 50:
            model = preactresnet.PreActResNet50(
                num_classes=args.n_classes,
                zero_init_residual=args.zero_init_residual)
        elif args.model_depth == 101:
            model = preactresnet.PreActResNet101(
                num_classes=args.n_classes,
                zero_init_residual=args.zero_init_residual)
        elif args.model_depth == 152:
            model = preactresnet.PreActResNet152(
                num_classes=args.n_classes,
                zero_init_residual=args.zero_init_residual)
        elif args.model_depth == 200:
            model = preactresnet.PreActResNet200(
                num_classes=args.n_classes,
                zero_init_residual=args.zero_init_residual)

    if args.arch == 'lenoresnet':
        assert args.model_depth in [50, 101, 152, 200, 401, 500]

        if args.model_depth == 50:
            model = lenoresnet.LeNo_ResNet50(
                num_classes=args.n_classes,
                zero_init_residual=args.zero_init_residual)
        elif args.model_depth == 101:
            model = lenoresnet.LeNo_ResNet101(
                num_classes=args.n_classes,
                zero_init_residual=args.zero_init_residual)
        elif args.model_depth == 152:
            model = lenoresnet.LeNo_ResNet152(
                num_classes=args.n_classes,
                zero_init_residual=args.zero_init_residual)
        elif args.model_depth == 200:
            model = lenoresnet.LeNo_ResNet200(
                num_classes=args.n_classes,
                zero_init_residual=args.zero_init_residual)
        elif args.model_depth == 401:
            model = lenoresnet.LeNo_ResNet401(
                num_classes=args.n_classes,
                zero_init_residual=args.zero_init_residual)
        elif args.model_depth == 500:
            model = lenoresnet.LeNo_ResNet500(
                num_classes=args.n_classes,
                zero_init_residual=args.zero_init_residual)				
				
    return model