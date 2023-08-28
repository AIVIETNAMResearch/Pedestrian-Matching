from torch.optim.lr_scheduler import LambdaLR
import math

def create_scheduler(args, optimizer):
    if 'num_training_steps' not in args:
        args['num_training_steps'] = args['epochs'] * args['step_per_epoch']
    print("### num_training_steps, ", args['num_training_steps'], flush=True)

    if 'min_rate' not in args:
        args['min_rate'] = 0.0

    if isinstance(args['num_warmup_steps'], float):
        assert 0 <= args['num_warmup_steps'] < 1
        args['num_warmup_steps'] = int(args['num_training_steps'] * args['num_warmup_steps'])
    print("### num_warmup_steps, ", args['num_warmup_steps'], flush=True)

    if args.sched == 'linear':
        def lr_lambda(current_step: int):
            if current_step < args.num_warmup_steps:
                return float(current_step) / float(max(1, args.num_warmup_steps))
            return max(
                args['min_rate'], float(args.num_training_steps - (1-args['min_rate'])*current_step) / float(
                    max(1, args.num_training_steps - args.num_warmup_steps))
            )

        lr_scheduler = LambdaLR(optimizer, lr_lambda, last_epoch=-1)
    if args.sched == "cosine":
        def cosine_lr_lambda(current_step: int):
            if current_step < args.num_warmup_steps:
                return float(current_step) / float(max(1, args.num_warmup_steps))
            progress = float(current_step - args.num_warmup_steps) / float(max(1, args.num_training_steps - args.num_warmup_steps))
            return max(args['min_rate'], 0.5 * (1.0 + math.cos(math.pi * progress)))

        lr_scheduler = LambdaLR(optimizer, cosine_lr_lambda, last_epoch=-1)
    else:
        raise NotImplementedError(f"args.sched == {args.sched}")

    return lr_scheduler
