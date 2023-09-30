import gc
import argparse
import os
import random
import os.path as osp
import sys
import datetime
import time
import math
import json
import jstyleson
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torchvision import datasets, transforms
import dino_training.utils as utils

import dino_training.vision_transformer as vits
from dino_training.vision_transformer import DINOHead, DINOLoss, FlowDINOHead, FlowLoss

from dataset.video import VideoDataset, VideoDatasetV2
from dino_training.flow_utils import DataAugmentationDINOFlow

data_name_to_json_path = {
    'vos': './dataset/indexes/uvo.json',
    'vspw': './dataset/indexes/vspw.json',
    'uvo': './dataset/indexes/youtube_vos.json'
}


class Parameters:
    name = f'experiment_name'

    video_datasets = 'uvo,vspw,vos'
    global_crops_scale = (0.4, 1.)
    local_crops_number = 8
    local_crops_scale = (0.05, 0.4)
    num_workers = 16

    clip_grad= 3.0
    drop_path_rate= 0.1
    out_dim = 8192

    patch_size= 16
    saveckp_freq= 1
    teacher_temp= 0.03
    warmup_teacher_temp= 0.04
    warmup_teacher_temp_epochs= 0
    
    momentum_teacher = 0.996
    use_bn_in_head = False
    norm_last_layer = True

    output_dir = "/filer/tmp1/xz653/new_checkpoints"

    use_fp16 = True
    arch = "vit_small"
    batch_size_per_gpu = 14 # 8
    stop_epoch = 1

    epochs = 5
    warmup_epochs= 0
    lr= 1e-4   # 5e-4
    min_lr= 1e-6
    weight_decay= 0.1 # 0.04
    weight_decay_end= 0.4

    initial_checkpoint =  "/filer/tmp1/xz653/new_checkpoints/train/s16_baseline/full.pth"
    use_teacher_as_initial_student = True

    flow_out_dim = 1024
    use_imagenet = False

    flow_temp = 0.1
    flow_radius = 0.7
    flow_kernel_size = 3
    flow_stride = 2
    flow_static_threshold = 0.1
    flow_loss_weight_margin = 0.01
    flow_loss_weight_mode = 'norm'

    opt = 'adamw'
    flow_loss_weight = 1.0
    imagenet_mix_training = True
    normalize_flow = True

    checkpoint_every = 5000
    accum_iter = 1



def get_flow_head(arch='vit_small'):
    if 'small' in arch:
        embed_size = 384
    elif 'base' in arch:
        embed_size = 768
    else:
        raise NotImplementedError()
    return FlowDINOHead(embed_size)
        


def train_dino(args: Parameters):
    if args.imagenet_mix_training:
        args.out_dim = 65536

    os.makedirs(args.output_dir, exist_ok=True)
    utils.init_distributed_mode(args)
    utils.fix_random_seeds(random.randint(0, 1000))
    print("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items())))
    cudnn.benchmark = True

    # ============ preparing data ... ============
    if args.imagenet_mix_training:
        imagenet_transform = utils.DataAugmentationDINO(args.global_crops_scale, args.local_crops_scale, args.local_crops_number)

        flow_transform = DataAugmentationDINOFlow(args.global_crops_scale, args.local_crops_scale, args.local_crops_number, flow_mode=True)
        video_dataset = VideoDataset(list(data_name_to_json_path.values()), flow=True,
                                    transform=flow_transform, return_norm_stats=True, 
                                    normalize_flow=args.normalize_flow)

        video_dataset_len = len(video_dataset)

        class CombinedDataset(datasets.ImageFolder):
            def __getitem__(self, index):
                images, labels = super().__getitem__(index)
                return images + video_dataset[random.randint(0, video_dataset_len-1)], labels

        dataset = CombinedDataset("/common/users/xz653/Dataset/ImageNet/imagenet/train", transform=imagenet_transform)
    else:
        transform = DataAugmentationDINOFlow(args.global_crops_scale, args.local_crops_scale, args.local_crops_number, flow_mode=True)
        jsons = [data_name_to_json_path[dname] for dname in args.video_datasets.split(',')]
        video_dataset = VideoDataset(jsons, flow=True, transform=transform, return_norm_stats=True, normalize_flow=True)
        dataset = video_dataset

    sampler = torch.utils.data.DistributedSampler(dataset, shuffle=True)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        sampler=sampler,
        batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    print(f"Data loaded: there are {len(dataset)} images.")

    # ============ building student and teacher networks ... ============
    student = vits.__dict__[args.arch](
        patch_size=args.patch_size,
        drop_path_rate=args.drop_path_rate,  # stochastic depth
        return_feats=True
    )
    teacher = vits.__dict__[args.arch](patch_size=args.patch_size,  return_feats=False)


    embed_dim = student.embed_dim
    # multi-crop wrapper handles forward with inputs of different resolutions
    # only activation `return_backbone_output` for video input
    student = utils.MultiCropWrapper(student, 
        DINOHead(embed_dim, args.out_dim, use_bn=args.use_bn_in_head, norm_last_layer=args.norm_last_layer), 
        flow_head=get_flow_head(args.arch))

    teacher = utils.MultiCropWrapper(teacher,
        DINOHead(embed_dim, args.out_dim, args.use_bn_in_head, norm_last_layer=args.norm_last_layer))

    # move networks to gpu
    student, teacher = student.cuda(), teacher.cuda()
    # synchronize batch norms (if any)
    if utils.has_batchnorms(student):
        student = nn.SyncBatchNorm.convert_sync_batchnorm(student)
        teacher = nn.SyncBatchNorm.convert_sync_batchnorm(teacher)

        # we need DDP wrapper to have synchro batch norms working...
        teacher = nn.parallel.DistributedDataParallel(teacher, device_ids=[args.gpu])
        teacher_without_ddp = teacher.module
    else:
        # teacher_without_ddp and teacher are the same thing
        teacher_without_ddp = teacher

    student = nn.parallel.DistributedDataParallel(student, device_ids=[args.gpu], find_unused_parameters=False)
    # teacher and student start with the same weights
    teacher_without_ddp.load_state_dict(student.module.state_dict(), strict=False)
    # there is no backpropagation through the teacher, so no need for gradients
    for p in teacher.parameters():
        p.requires_grad = False
    teacher.eval()
    print(f"Student and Teacher are built: they are both {args.arch} network.")

    # ============ preparing loss ... ============
    flow_corr_loss = FlowLoss(flow_temp=args.flow_temp, radius=args.flow_radius,
                                kernel_size=args.flow_kernel_size, stride=args.flow_stride, 
                                loss_weight_margin=args.flow_loss_weight_margin,
                                loss_weight_mode=args.flow_loss_weight_mode,
                                static_threshold=args.flow_static_threshold).cuda()

    dino_loss = DINOLoss(args.out_dim, args.local_crops_number + 2, args.warmup_teacher_temp, 
                        args.teacher_temp, args.warmup_teacher_temp_epochs, args.epochs).cuda()

    # ============ preparing optimizer ... ============
    params_groups = utils.get_params_groups(student, exclude="flow_head")
    flow_head_params_groups = utils.get_params_groups(student, include="flow_head") 
    if len(flow_head_params_groups) > 0:
        if len(flow_head_params_groups[0]['params']) == 0:
            flow_head_params_groups = []
        
    if args.opt == 'adamw':
        optimizer = torch.optim.AdamW(params_groups + flow_head_params_groups) 
    elif args.opt == 'lars':
        optimizer = utils.LARS(params_groups + flow_head_params_groups) 
    else:
        raise NotImplementedError

    fp16_scaler = None
    if args.use_fp16:
        fp16_scaler = torch.cuda.amp.GradScaler()

    # ============ init schedulers ... ============
    lr_schedule = utils.cosine_scheduler(
        args.lr * (args.batch_size_per_gpu * args.accum_iter * utils.get_world_size()) / 256.,  # linear scaling rule
        args.min_lr,
        args.epochs, len(data_loader),
        warmup_epochs=args.warmup_epochs,
    )
    wd_schedule = utils.cosine_scheduler(
        args.weight_decay,
        args.weight_decay_end,
        args.epochs, len(data_loader),
    )
    linear_schedule = np.linspace(0.0, 1.0, args.epochs * len(data_loader))

    # momentum parameter is increased to 1. during training with a cosine schedule
    momentum_schedule = utils.cosine_scheduler(args.momentum_teacher, 1, args.epochs, len(data_loader))
    print(f"Loss, optimizer and schedulers ready.")

    # ============ optionally resume training ... ============
    to_restore = {"epoch": 0}

    if args.initial_checkpoint:
        def check_load_result(res):
            assert len(res.unexpected_keys) == 0
            for k in res.missing_keys:
                if not args.apply_sinkhorn:
                    assert 'flow_head' in k or 'ext' in k

        ckpt = torch.load(args.initial_checkpoint)
        if "teacher" in ckpt:
            states = ckpt['teacher']

            if args.imagenet_mix_training:
                assert 'head.mlp.4.bias' in states

            load_result = student.module.load_state_dict(states, strict=False)           
            print(load_result)
            check_load_result(load_result)
            load_result = teacher.load_state_dict(states, strict=False)  
            check_load_result(load_result)
            to_restore['epoch'] = ckpt.get('epoch', 0)
        elif 'model' in ckpt:
            ckpt = ckpt['model']
            ckpt = {k: v for k, v in ckpt.items() if 'decoder' not in k and 'mask_token' not in k}

            load_result = student.module.backbone.load_state_dict(ckpt, strict=False)    
            check_load_result(load_result)

            load_result = teacher_without_ddp.backbone.load_state_dict(ckpt, strict=False) 
            check_load_result(load_result)
        else:
            _ = student.module.backbone.load_state_dict(ckpt, strict=False)
            assert len(_.unexpected_keys) == 0
            if args.ext is not None:
                _ = teacher_without_ddp.backbone.load_state_dict(ckpt, strict=False)
                assert len(_.unexpected_keys) == 0
            else:
                teacher_without_ddp.backbone.load_state_dict(ckpt)
    else:
        utils.restart_from_checkpoint(
            os.path.join(args.output_dir, "checkpoint.pth"),
            run_variables=to_restore,
            student=student,
            teacher=teacher,
            optimizer=optimizer,
            fp16_scaler=fp16_scaler,
            dino_loss=dino_loss,
            flow_corr_loss=flow_corr_loss
        )

    start_epoch = to_restore['epoch']
    print(f"START EPOCH = {start_epoch}")

    start_time = time.time()
    print("Starting DINO training !")
    for epoch in range(start_epoch, args.epochs):
        data_loader.sampler.set_epoch(epoch)

        # ============ training one epoch of DINO ... ============
        train_stats = train_one_epoch(student, teacher, teacher_without_ddp, dino_loss, flow_corr_loss,
            data_loader, optimizer, lr_schedule, wd_schedule, linear_schedule, momentum_schedule,
            epoch, fp16_scaler, args)

        # ============ writing logs ... ============
        save_dict = {
            'student': student.state_dict(),
            'teacher': teacher.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch + 1,
            'args': args,
            'dino_loss': dino_loss.state_dict(),
            'flow_corr_loss': flow_corr_loss.state_dict() if flow_corr_loss is not None else None,
        }
        if fp16_scaler is not None:
            save_dict['fp16_scaler'] = fp16_scaler.state_dict()
        utils.save_on_master(save_dict, os.path.join(args.output_dir, 'checkpoint.pth'))
        if args.saveckp_freq and epoch % args.saveckp_freq == 0:
            utils.save_on_master(save_dict, os.path.join(args.output_dir, f'checkpoint{epoch:04}.pth'))
        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     'epoch': epoch}
        if utils.is_main_process():
            with (Path(args.output_dir) / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


def train_one_epoch(student, teacher, teacher_without_ddp, dino_loss, flow_corr_loss, data_loader,
                    optimizer, lr_schedule, wd_schedule, linear_schedule, momentum_schedule, epoch,
                    fp16_scaler, args):
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Epoch: [{}/{}]'.format(epoch, args.epochs)


    def student_backward(loss, retain_graph=False):
        if fp16_scaler is None:
            loss.backward()
        else:
            fp16_scaler.scale(loss).backward(retain_graph=retain_graph)

    def optimizer_step():
        if fp16_scaler is None:
            if args.clip_grad:
                param_norms = utils.clip_gradients(student, args.clip_grad)
            optimizer.step()
        else:
            if args.clip_grad:
                fp16_scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
                param_norms = utils.clip_gradients(student, args.clip_grad)
            fp16_scaler.step(optimizer)
            fp16_scaler.update() 
        
        
    def check_inf(loss, msg=""):
        if not math.isfinite(loss.item()):
            print("{} Loss is {}, stopping training".format(msg, loss.item()), force=True)
            sys.exit(1)
    
    accum_iter = args.accum_iter

    for it, batch in enumerate(metric_logger.log_every(data_loader, 10, header)):
        if isinstance(batch, (list, tuple)) and len(batch) == 2: batch = batch[0]

        # update weight decay and learning rate according to their schedule
        it = len(data_loader) * epoch + it  # global training iteration

        if args.checkpoint_every > 0 and it > 0 and (it % args.checkpoint_every) == 0:
            utils.save_on_master({'teacher': teacher.state_dict()}, os.path.join(args.output_dir, f'checkpoint-{epoch}-{it}.pth'))

        # update parameters
        lr, wd = lr_schedule[it], wd_schedule[it]
        optimizer.param_groups[0]['lr'] = optimizer.param_groups[1]['lr'] = lr
        optimizer.param_groups[0]['weight_decay'] = wd

        if len(optimizer.param_groups) == 4:
            optimizer.param_groups[2]['lr'] = optimizer.param_groups[3]['lr'] = lr
            optimizer.param_groups[2]['weight_decay'] = wd

        num_crops = 2+args.local_crops_number
        images = batch[:num_crops] 
        flow_batch = batch[num_crops:]
        images = [im.cuda(non_blocking=True) for im in images] 

        # teacher and student forward passes + compute dino loss
        with torch.cuda.amp.autocast(fp16_scaler is not None):
            dn_loss = 0.0
            teacher_output, _ = teacher(images[:2])

            if args.imagenet_mix_training:
                vid_images = flow_batch[:num_crops]
                vid_images = [im.cuda(non_blocking=True) for im in vid_images] 

                student_output, _ = student(images, flow_on_local_crops=False)
                flow_batch = flow_batch[num_crops:]

                flow_list = [torch.cat(flow_batch[:2], dim=0)]
                flow_max_norm_list = [flow_batch[-1].repeat(2, 1), ]

                flow_list.append(torch.cat(flow_batch[2:-1], dim=0))
                flow_max_norm_list.append(flow_batch[-1].repeat(args.local_crops_number, 1))

                corr_loss = 0.0
                student_flow_output_cls, student_flow_output = student(vid_images, flow_on_local_crops=True)

                for _1, _2, _3 in zip(student_flow_output, flow_list, flow_max_norm_list):
                    corr_loss += (flow_corr_loss(_1, _2, _3) * args.flow_loss_weight)
                check_inf(corr_loss, "flow_dino")
                corr_loss /= len(flow_list)            
            else:
                student_output, student_flow_output = student(images, flow_on_local_crops=args.use_flow_corr)

                if len(flow_batch) > 0:
                    flow_list = [torch.cat(flow_batch[:2], dim=0)]
                    flow_max_norm_list = [flow_batch[-1].repeat(2, 1), ]

                    flow_list.append(torch.cat(flow_batch[2:-1], dim=0))
                    flow_max_norm_list.append(flow_batch[-1].repeat(args.local_crops_number, 1))

                corr_loss = 0.0
                for _1, _2, _3 in zip(student_flow_output, flow_list, flow_max_norm_list):
                    corr_loss += (flow_corr_loss(_1, _2, _3) * args.flow_loss_weight)
                check_inf(corr_loss, "flow_dino")
                corr_loss /= len(flow_list)

            dn_loss += dino_loss(student_output, teacher_output, epoch, update_center=(it + 1) % accum_iter == 0)
            check_inf(dn_loss, "dino")

            loss = corr_loss + dn_loss
            loss /= accum_iter

            if (it + 1) % accum_iter == 0:
                optimizer.zero_grad()
            student_backward(loss)
            if (it + 1) % accum_iter == 0:
                optimizer_step()

        # EMA update for the teacher
        if (it + 1) % accum_iter == 0:
            with torch.no_grad():
                m = momentum_schedule[it]  # momentum parameter
                student_parameters = dict(student.module.named_parameters())
                for q, param_k in  teacher_without_ddp.named_parameters():
                    param_q = student_parameters[q]
                    param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)

        # logging
        torch.cuda.synchronize()
        metric_logger.update(dino_loss=dn_loss.item(), flow_corr_loss=corr_loss.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(wd=optimizer.param_groups[0]["weight_decay"])

        if len(optimizer.param_groups) >= 3 and args.rescale_flow_head_lr_wd:
            metric_logger.update(fh_lr=optimizer.param_groups[2]["lr"])
            metric_logger.update(fh_wd=optimizer.param_groups[2]["weight_decay"])
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


if __name__ == '__main__':
    args = argparse.Namespace()

    def inject_args(args, cls):
        for k, v in vars(cls).items():
            if not k.startswith("_") and not callable(v):
                setattr(args, k, v)

        if '--config' in sys.argv:
            config_file = sys.argv[sys.argv.index('--config') + 1]
            kwargs = jstyleson.load(open(config_file))
            for k, v in kwargs.items():
                setattr(args, k, v)
        return args

    args = inject_args(args, Parameters)

    args.dist_url = "env://"
    args.local_rank = int(os.environ["LOCAL_RANK"])

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    train_dino(args)
