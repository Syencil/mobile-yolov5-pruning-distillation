import argparse

import os
import torch

import torch.distributed as dist
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import yaml
from torch.utils.tensorboard import SummaryWriter

import test  # import test.py to get mAP after each epoch
from models.yolo import Model
from utils.datasets import *
from utils.utils import *

mixed_precision = True
try:  # Mixed precision training https://github.com/NVIDIA/apex
    from apex import amp
except:
    print('Apex recommended for faster mixed precision training: https://github.com/NVIDIA/apex')
    mixed_precision = False  # not installed



# Hyperparameters
hyp = {'lr0': 0.01,  # initial learning rate (SGD=1E-2, Adam=1E-3)
       'momentum': 0.937,  # SGD momentum
       'weight_decay': 5e-4,  # optimizer weight decay
       'l1': False,  # smooth l1 loss or iou loss
       'giou': 0.05,  # giou loss gain
       'cls': 0.58,  # cls loss gain
       'cls_pw': 1.0,  # cls BCELoss positive_weight
       'obj': 1.0,  # obj loss gain (*=img_size/320 if img_size != 320)
       'obj_pw': 1.0,  # obj BCELoss positive_weight
       'iou_t': 0.20,  # iou training threshold
       'anchor_t': 4.0,  # anchor-multiple threshold
       'fl_gamma': 0.0,  # focal loss gamma (efficientDet default is gamma=1.5)
       'hsv_h': 0.014,  # image HSV-Hue augmentation (fraction)
       'hsv_s': 0.68,  # image HSV-Saturation augmentation (fraction)
       'hsv_v': 0.36,  # image HSV-Value augmentation (fraction)
       'degrees': 0.0,  # image rotation (+/- deg)
       'translate': 0.0,  # image translation (+/- fraction)
       'scale': 0.5,  # image scale (+/- gain)
       'shear': 0.0}  # image shear (+/- deg)

# Overwrite hyp with hyp*.txt (optional)
f = glob.glob('hyp*.txt')
if f:
    print('Using %s' % f[0])
    for k, v in zip(hyp.keys(), np.loadtxt(f[0])):
        hyp[k] = v

# Print focal loss if gamma > 0
if hyp['fl_gamma']:
    print('Using FocalLoss(gamma=%g)' % hyp['fl_gamma'])


def train(hyp):
    epochs = opt.epochs  # 300
    batch_size = opt.batch_size  # 64
    weights = opt.weights  # initial training weights

    # Configure
    init_seeds(1)
    with open(opt.data) as f:
        data_dict = yaml.load(f, Loader=yaml.FullLoader)  # model dict
    train_path = data_dict['train']
    test_path = data_dict['val']
    nc = 1 if opt.single_cls else int(data_dict['nc'])  # number of classes

    # Remove previous results
    for f in glob.glob(os.path.join(rdir, '*_batch*.jpg')) + glob.glob(results_file):
        os.remove(f)

    # Create model
    model = Model(opt.cfg).to(device)
    if opt.ft:
        new = torch.load(weights, map_location=device)
        model = new['model']
        print(model)
        print("Finetune Mode...")
    assert model.md['nc'] == nc, '%s nc=%g classes but %s nc=%g classes' % (opt.data, nc, opt.cfg, model.md['nc'])

    # Image sizes
    gs = int(max(model.stride))  # grid size (max stride)
    imgsz, imgsz_test = [check_img_size(x, gs) for x in opt.img_size]  # verify imgsz are gs-multiples

    # Optimizer
    nbs = 64  # nominal batch size
    accumulate = max(round(nbs / batch_size), 1)  # accumulate loss before optimizing
    hyp['weight_decay'] *= batch_size * accumulate / nbs  # scale weight_decay
    if opt.sl > 0:
        hyp['sl'] *= batch_size * accumulate / nbs
    pg0, pg1, pg2 = [], [], []  # optimizer parameter groups
    for k, v in model.named_parameters():
        if v.requires_grad:
            if '.bias' in k:
                pg2.append(v)  # biases
            elif '.weight' in k and '.bn' not in k:
                pg1.append(v)  # apply weight decay
            else:
                pg0.append(v)  # all else

    optimizer = optim.Adam(pg0, lr=hyp['lr0']) if opt.adam else \
        optim.SGD(pg0, lr=hyp['lr0'], momentum=hyp['momentum'], nesterov=True)
    optimizer.add_param_group({'params': pg1, 'weight_decay': hyp['weight_decay']})  # add pg1 with weight_decay
    optimizer.add_param_group({'params': pg2})  # add pg2 (biases)
    print('Optimizer groups: %g .bias, %g conv.weight, %g other' % (len(pg2), len(pg1), len(pg0)))
    del pg0, pg1, pg2

    # Load Model
    google_utils.attempt_download(weights)
    start_epoch, best_fitness = 0, 0.0
    if weights.endswith('.pt') and not opt.ft:  # pytorch format
        ckpt = torch.load(weights, map_location=device)  # load checkpoint

        # load model
        try:

            dic = {}
            for k, v in ckpt['model'].float().state_dict().items():
                if k in model.state_dict() and model.state_dict()[k].shape == v.shape:
                    dic[k] = v
            ckpt['model'] = dic
            # ckpt['model'] = {k: v for k, v in ckpt['model'].float().state_dict().items()
            #                  if model.state_dict()[k].shape == v.shape}  # to FP32, filter
            model.load_state_dict(ckpt['model'], strict=False)
        except KeyError as e:
            s = "%s is not compatible with %s. Specify --weights '' or specify a --cfg compatible with %s." \
                % (opt.weights, opt.cfg, opt.weights)
            raise KeyError(s) from e
        if opt.resume:
            # load optimizer
            if ckpt['optimizer'] is not None:
                optimizer.load_state_dict(ckpt['optimizer'])
                best_fitness = ckpt['best_fitness']

            # load results
            if ckpt.get('training_results') is not None:
                with open(results_file, 'w') as file:
                    file.write(ckpt['training_results'])  # write results.txt

            start_epoch = ckpt['epoch'] + 1
        del ckpt
    elif weights.endswith('.pth'):
        ckpt = torch.load(weights, map_location=device)  # load checkpoint
        # load model
        try:
            dic = {}
            for k in ckpt:
                v = ckpt[k]
                n_name = k.replace("features", "model")
                if n_name in model.state_dict() and model.state_dict()[n_name].shape == v.shape:
                    dic[n_name] = v
            ckpt['model'] = dic
            # ckpt['model'] = {k: v for k, v in ckpt['model'].float().state_dict().items()
            #                  if model.state_dict()[k].shape == v.shape}  # to FP32, filter
            model.load_state_dict(dic, strict=False)
            print("restore %d vars from %s" % (len(dic), weights))
        except KeyError as e:
            s = "%s is not compatible with %s. Specify --weights '' or specify a --cfg compatible with %s." \
                % (opt.weights, opt.cfg, opt.weights)
            raise KeyError(s) from e
        del ckpt

    if opt.dist:
        print("load t-model from", opt.t_weights)
        t_model = torch.load(opt.t_weights, map_location=torch.device('cpu'))
        if t_model.get("model", None) is not None:
            t_model = t_model["model"]
        t_model.to(device)
        t_model.float()
        t_model.train()

        if opt.d_feature:
            activation = {}
            def get_activation(name):
                def hook(model, inputs, outputs):
                    activation[name] = outputs
                return hook

            def get_hooks():
                hooks = []
                # S-model
                hooks.append(model.model._modules["6"].register_forward_hook(get_activation("s_f1")))
                hooks.append(model.model._modules["13"].register_forward_hook(get_activation("s_f2")))
                hooks.append(model.model._modules["17"].register_forward_hook(get_activation("s_f3")))
                # T-model
                hooks.append(t_model.model._modules["4"].register_forward_hook(get_activation("t_f1")))
                hooks.append(t_model.model._modules["6"].register_forward_hook(get_activation("t_f2")))
                hooks.append(t_model.model._modules["10"].register_forward_hook(get_activation("t_f3")))
                return hooks
            # feature convert
            from models.common import Converter
            c1 = 128
            c2 = 256
            c3 = 512
            if opt.type == "dfmvocs_l":
                c1 = 256
                c2 = 512
                c3 = 1024
            # S_Converter_1 = Converter(32, c1, act=True)
            # S_Converter_2 = Converter(96, c2, act=True)
            # S_Converter_3 = Converter(320, c3, act=True)
            # S_Converter_1.to(device)
            # S_Converter_2.to(device)
            # S_Converter_3.to(device)
            # S_Converter_1.train()
            # S_Converter_2.train()
            # S_Converter_3.train()

            # T_Converter_1 = nn.ReLU6()
            # T_Converter_2 = nn.ReLU6()
            # T_Converter_3 = nn.ReLU6()
            T_Converter_1 = Converter(c1, 32, act=True)
            T_Converter_2 = Converter(c2, 96, act=True)
            T_Converter_3 = Converter(c3, 320, act=True)
            T_Converter_1.to(device)
            T_Converter_2.to(device)
            T_Converter_3.to(device)
            T_Converter_1.train()
            T_Converter_2.train()
            T_Converter_3.train()

    # Mixed precision training https://github.com/NVIDIA/apex
    if mixed_precision:
        model, optimizer = amp.initialize(model, optimizer, opt_level='O1', verbosity=0)

    # Scheduler https://arxiv.org/pdf/1812.01187.pdf
    lf = lambda x: (((1 + math.cos(x * math.pi / epochs)) / 2) ** 1.0) * 0.9 + 0.1  # cosine
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    scheduler.last_epoch = start_epoch - 1  # do not move
    # https://discuss.pytorch.org/t/a-problem-occured-when-resuming-an-optimizer/28822
    # plot_lr_scheduler(optimizer, scheduler, epochs)

    # Initialize distributed training
    if device.type != 'cpu' and torch.cuda.device_count() > 1 and torch.distributed.is_available():
        dist.init_process_group(backend='nccl',  # distributed backend
                                init_method='tcp://127.0.0.1:9999',  # init method
                                world_size=1,  # number of nodes
                                rank=0)  # node rank
        model = torch.nn.parallel.DistributedDataParallel(model)
        if opt.dist:
            raise NotImplementedError("Distillation do not support DDP!")

    # Dataset
    dataset = LoadImagesAndLabels(train_path, imgsz, batch_size,
                                  augment=True,
                                  hyp=hyp,  # augmentation hyperparameters
                                  rect=opt.rect,  # rectangular training
                                  cache_images=opt.cache_images,
                                  single_cls=opt.single_cls)
    mlc = np.concatenate(dataset.labels, 0)[:, 0].max()  # max label class
    assert mlc < nc, 'Label class %g exceeds nc=%g in %s. Correct your labels or your model.' % (mlc, nc, opt.cfg)

    # Dataloader
    batch_size = min(batch_size, len(dataset))
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=batch_size,
                                             num_workers=nw,
                                             shuffle=not opt.rect,  # Shuffle=True unless rectangular training is used
                                             pin_memory=True,
                                             collate_fn=dataset.collate_fn)

    # Testloader
    testloader = torch.utils.data.DataLoader(LoadImagesAndLabels(test_path, imgsz_test, batch_size,
                                                                 hyp=hyp,
                                                                 rect=True,
                                                                 cache_images=opt.cache_images,
                                                                 single_cls=opt.single_cls),
                                             batch_size=batch_size,
                                             num_workers=nw,
                                             pin_memory=True,
                                             collate_fn=dataset.collate_fn)

    # Model parameters
    hyp['cls'] *= nc / 80.  # scale coco-tuned hyp['cls'] to current dataset
    model.nc = nc  # attach number of classes to model
    model.hyp = hyp  # attach hyperparameters to model
    model.gr = 1.0  # giou loss ratio (obj_loss = 1.0 or giou)
    model.class_weights = labels_to_class_weights(dataset.labels, nc).to(device)  # attach class weights
    model.names = data_dict['names']

    # Class frequency
    labels = np.concatenate(dataset.labels, 0)
    c = torch.tensor(labels[:, 0])  # classes
    # cf = torch.bincount(c.long(), minlength=nc) + 1.
    # model._initialize_biases(cf.to(device))
    plot_labels(labels, os.path.join(rdir, "label.png"))
    tb_writer.add_histogram('classes', c, 0)

    # Check anchors
    check_anchors(dataset, model=model, thr=hyp['anchor_t'], imgsz=imgsz)

    # Exponential moving average
    ema = torch_utils.ModelEMA(model)

    # Start training
    t0 = time.time()
    nb = len(dataloader)  # number of batches
    n_burn = max(3 * nb, 1e3)  # burn-in iterations, max(3 epochs, 1k iterations)
    maps = np.zeros(nc)  # mAP per class
    results = (0, 0, 0, 0, 0, 0, 0)  # 'P', 'R', 'mAP', 'F1', 'val GIoU', 'val Objectness', 'val Classification'
    print('Image sizes %g train, %g test' % (imgsz, imgsz_test))
    print('Using %g dataloader workers' % nw)
    print('Starting training for %g epochs...' % epochs)

    if opt.sl > 0:
        print("Sparse Learning Model!")
        print("===> Sparse learning rate is ", opt.sl)
        ignore_idx = [230, 260, 290]
        prunable_modules = []
        prunable_module_type = (nn.BatchNorm2d, )
        for i, m in enumerate(model.modules()):
            if i in ignore_idx:
                continue
            if isinstance(m, prunable_module_type):
                prunable_modules.append(m)
    # torch.autograd.set_detect_anomaly(True)
    for epoch in range(start_epoch, epochs):  # epoch ------------------------------------------------------------------
        if opt.dist and opt.d_feature:
            hooks = get_hooks()
        model.train()
        # Update image weights (optional)
        if dataset.image_weights:
            w = model.class_weights.cpu().numpy() * (1 - maps) ** 2  # class weights
            image_weights = labels_to_image_weights(dataset.labels, nc=nc, class_weights=w)
            dataset.indices = random.choices(range(dataset.n), weights=image_weights, k=dataset.n)  # rand weighted idx

        mloss = torch.zeros(4, device=device)  # mean losses
        print(('\n' + '%10s' * 8) % ('Epoch', 'gpu_mem', 'GIoU', 'obj', 'cls', 'total', 'targets', 'img_size'))
        pbar = tqdm(enumerate(dataloader), total=nb)  # progress bar
        for i, (imgs, targets, paths, _) in pbar:  # batch -------------------------------------------------------------
            ni = i + nb * epoch  # number integrated batches (since train start)
            imgs = imgs.to(device).float() / 255.0  # uint8 to float32, 0 - 255 to 0.0 - 1.0

            # Burn-in
            if ni <= n_burn:
                xi = [0, n_burn]  # x interp
                # model.gr = np.interp(ni, xi, [0.0, 1.0])  # giou loss ratio (obj_loss = 1.0 or giou)
                accumulate = max(1, np.interp(ni, xi, [1, nbs / batch_size]).round())
                for j, x in enumerate(optimizer.param_groups):
                    # bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
                    x['lr'] = np.interp(ni, xi, [0.1 if j == 2 else 0.0, x['initial_lr'] * lf(epoch)])
                    if 'momentum' in x:
                        x['momentum'] = np.interp(ni, xi, [0.9, hyp['momentum']])

            # Multi-scale
            if opt.multi_scale:
                sz = random.randrange(np.ceil(imgsz * 0.66), np.ceil(imgsz * 1.33 + 32)) // gs * gs  # size
                sf = sz / max(imgs.shape[2:])  # scale factor
                if sf != 1:
                    ns = [math.ceil(x * sf / gs) * gs for x in imgs.shape[2:]]  # new shape (stretched to gs-multiple)
                    imgs = F.interpolate(imgs, size=ns, mode='bilinear', align_corners=False)

            # Forward
            pred = model(imgs)
            if opt.dist:
                if opt.d_online:
                    t_pred = t_model(imgs)
                else:
                    with torch.no_grad():
                        t_pred = t_model(imgs)
                if opt.d_feature:
                    # s_f1 = S_Converter_1(activation["s_f1"])
                    # s_f2 = S_Converter_2(activation["s_f2"])
                    # s_f3 = S_Converter_3(activation["s_f3"])
                    # s_f = [s_f1, s_f2, s_f3]
                    s_f = (activation["s_f1"], activation["s_f2"], activation["s_f3"])
                    t_f1 = T_Converter_1(activation["t_f1"])
                    t_f2 = T_Converter_2(activation["t_f2"])
                    t_f3 = T_Converter_3(activation["t_f3"])
                    t_f = [t_f1, t_f2, t_f3]
                    # t_f = (activation["t_f1"], activation["t_f2"], activation["t_f3"])
            # Loss
            loss, loss_items = compute_loss(pred, targets.to(device), model)

            # Sparse Learning
            if opt.sl > 0:
                loss = compute_pruning_loss(pred, prunable_modules, model, loss)

            # distillation
            if opt.dist:
                if opt.d_online:
                    loss, _ = compute_loss(t_pred, targets.to(device), model, loss)
                loss = compute_distillation_output_loss(pred, t_pred, model, loss)
                if opt.d_feature:
                    loss = compute_distillation_feature_loss(s_f, t_f, model, loss)

            if not torch.isfinite(loss):
                print('WARNING: non-finite loss, ending training ', loss_items)
                return results

            # Backward
            if mixed_precision:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            # Optimize
            if ni % accumulate == 0:
                optimizer.step()
                optimizer.zero_grad()
                ema.update(model)

            # Print
            mloss = (mloss * i + loss_items) / (i + 1)  # update mean losses
            mem = '%.3gG' % (torch.cuda.memory_cached() / 1E9 if torch.cuda.is_available() else 0)  # (GB)
            s = ('%10s' * 2 + '%10.4g' * 6) % (
                '%g/%g' % (epoch, epochs - 1), mem, *mloss, targets.shape[0], imgs.shape[-1])
            pbar.set_description(s)

            # Plot
            if ni < 3:
                f = os.path.join(rdir, 'train_batch%g.jpg' % i)  # filename
                res = plot_images(images=imgs, targets=targets, paths=paths, fname=f)
                if tb_writer:
                    tb_writer.add_image(f, res, dataformats='HWC', global_step=epoch)
                    # tb_writer.add_graph(model, imgs)  # add model to tensorboard

            # end batch ------------------------------------------------------------------------------------------------

        if opt.dist and opt.d_feature:
            for hook in hooks:
                hook.remove()
        # Scheduler
        scheduler.step()

        # mAP
        ema.update_attr(model)
        final_epoch = epoch + 1 == epochs
        if not opt.notest or final_epoch:  # Calculate mAP
            results, maps, times = test.test(opt.data,
                                             batch_size=batch_size,
                                             imgsz=imgsz_test,
                                             save_json=final_epoch and opt.data.endswith(os.sep + 'coco.yaml'),
                                             model=ema.ema,
                                             single_cls=opt.single_cls,
                                             dataloader=testloader,
                                             fast=ni < n_burn)

        # Write
        with open(results_file, 'a') as f:
            f.write(s + '%10.4g' * 7 % results + '\n')  # P, R, mAP, F1, test_losses=(GIoU, obj, cls)
        if len(opt.name) and opt.bucket:
            os.system('gsutil cp results.txt gs://%s/results/results%s.txt' % (opt.bucket, opt.name))

        # Tensorboard
        if tb_writer:
            tags = ['train/giou_loss', 'train/obj_loss', 'train/cls_loss',
                    'metrics/precision', 'metrics/recall', 'metrics/mAP_0.5', 'metrics/F1',
                    'val/giou_loss', 'val/obj_loss', 'val/cls_loss']
            for x, tag in zip(list(mloss[:-1]) + list(results), tags):
                tb_writer.add_scalar(tag, x, epoch)

        # Update best mAP
        fi = fitness(np.array(results).reshape(1, -1))  # fitness_i = weighted combination of [P, R, mAP, F1]
        if fi > best_fitness:
            best_fitness = fi

        # Save model
        save = (not opt.nosave) or (final_epoch and not opt.evolve)
        if save:
            with open(results_file, 'r') as f:  # create checkpoint
                ckpt = {'epoch': epoch,
                        'best_fitness': best_fitness,
                        'training_results': f.read(),
                        'model': ema.ema.module if hasattr(model, 'module') else ema.ema,
                        'optimizer': None if final_epoch else optimizer.state_dict()}

            # Save last, best and delete
            torch.save(ckpt, last)
            if (best_fitness == fi) and not final_epoch:
                torch.save(ckpt, best)
            del ckpt

        # end epoch ----------------------------------------------------------------------------------------------------
    # end training

    n = opt.name
    if len(n):
        n = '_' + n if not n.isnumeric() else n
        fresults, flast, fbest = 'results%s.txt' % n, wdir + 'last%s.pt' % n, wdir + 'best%s.pt' % n
        for f1, f2 in zip([wdir + 'last.pt', wdir + 'best.pt', 'results.txt'], [flast, fbest, fresults]):
            if os.path.exists(f1):
                os.rename(f1, f2)  # rename
                ispt = f2.endswith('.pt')  # is *.pt
                strip_optimizer(f2) if ispt else None  # strip optimizer
                os.system('gsutil cp %s gs://%s/weights' % (f2, opt.bucket)) if opt.bucket and ispt else None  # upload

    if not opt.evolve:
        plot_results()  # save as results.png
    print('%g epochs completed in %.3f hours.\n' % (epoch - start_epoch + 1, (time.time() - t0) / 3600))
    dist.destroy_process_group() if torch.cuda.device_count() > 1 else None
    torch.cuda.empty_cache()
    return results


if __name__ == '__main__':
    check_git_status()
    parser = argparse.ArgumentParser()
    parser.add_argument("--type", type=str, help="Custom training type")
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--cfg', type=str, help='*.cfg path')
    parser.add_argument('--data', type=str, help='*.data path')
    parser.add_argument('--img-size', nargs='+', type=int, default=[640, 640], help='train,test sizes')
    parser.add_argument('--rect', action='store_true', help='rectangular training')
    parser.add_argument('--resume', action='store_true', help='resume training from last.pt')
    parser.add_argument('--nosave', action='store_true', help='only save final checkpoint')
    parser.add_argument('--notest', action='store_true', help='only test final epoch')
    parser.add_argument('--evolve', action='store_true', help='evolve hyperparameters')
    parser.add_argument('--bucket', type=str, default='', help='gsutil bucket')
    parser.add_argument('--cache-images', action='store_true', help='cache images for faster training')
    parser.add_argument('--weights', type=str, help='initial weights path')
    parser.add_argument('--name', default='', help='renames results.txt to results_name.txt if supplied')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--adam', action='store_true', help='use adam optimizer')
    parser.add_argument('--multi-scale', default=True, action='store_true', help='vary img-size +/- 50%')
    parser.add_argument('--single-cls', action='store_true', help='train as single-class dataset')
    parser.add_argument('--nw', type=int, default=None, help='num of worker')
    # pruning
    parser.add_argument('--sl', default=0, type=float, help='sparse learning')
    parser.add_argument('--ft', action='store_true', help='fine-tune')
    # distillation
    parser.add_argument('--dist', action='store_true', help='distillation')
    parser.add_argument('--t_weights', type=str, help='teacher model for distillation')
    parser.add_argument('--d_feature', action='store_true', help='if true, distill both feature and output layers')
    parser.add_argument('--d_online', action='store_true', help='if true, using online-distillation')
    opt = parser.parse_args()

    if opt.type == "mcocos":
        opt.cfg = 'models/mobile-yolo5s.yaml'
        opt.data = "data/coco.yaml"
        opt.weights = "outputs/dmvocs/weights/best_dmvocs.pt"
        opt.name = opt.type
        opt.epochs = 50
        opt.batch_size = 24
        opt.multi_scale = False

    if opt.type == "dmcocos":
        opt.cfg = 'models/mobile-yolo5s.yaml'
        opt.data = "data/coco.yaml"
        opt.name = opt.type
        opt.weights = "outputs/dmvocs/weights/best_dmvocs.pt"
        opt.epochs = 50
        opt.batch_size = 24
        opt.multi_scale = False
        opt.dist = True
        opt.t_weights = "/data/checkpoints/yolov5/yolov5s.pt"
        hyp["dist"] = 1

    if opt.type == "vocs":
        opt.cfg = 'models/yolov5s_voc.yaml'
        opt.data = "data/voc.yaml"
        opt.weights = "/data/checkpoints/yolov5/yolov5s.pt"
        opt.name = opt.type
        opt.epochs = 50
        opt.batch_size = 24
        opt.multi_scale = False

    if opt.type == "vocl":
        opt.cfg = 'models/yolov5l_voc.yaml'
        opt.data = "data/voc.yaml"
        opt.weights = "/data/checkpoints/yolov5/yolov5l.pt"
        opt.name = opt.type
        opt.epochs = 50
        opt.batch_size = 24
        opt.multi_scale = False

    if opt.type == "vocx":
        opt.cfg = 'models/yolov5x_voc.yaml'
        opt.data = "data/voc.yaml"
        opt.weights = "/data/checkpoints/yolov5/yolov5l.pt"
        opt.name = opt.type
        opt.epochs = 50
        opt.batch_size = 24
        opt.multi_scale = False

    if opt.type == "mvocs":
        opt.cfg = 'models/mobile-yolo5s_voc.yaml'
        opt.data = "data/voc.yaml"
        opt.weights = "/root/.cache/torch/checkpoints/mobilenet_v2-b0353104.pth"
        opt.name = opt.type
        opt.epochs = 50
        opt.batch_size = 24
        opt.multi_scale = False

    if opt.type == "mvocl":
        opt.cfg = 'models/mobile-yolo5l_voc.yaml'
        opt.data = "data/voc.yaml"
        opt.name = opt.type
        opt.weights = "/root/.cache/torch/checkpoints/mobilenet_v2-b0353104.pth"
        opt.epochs = 50
        opt.batch_size = 24
        opt.multi_scale = False

    if opt.type == "mvoc3":
        opt.cfg = 'models/mobile-yolo3_voc.yaml'
        opt.data = "data/voc.yaml"
        opt.name = opt.type
        opt.weights = "/root/.cache/torch/checkpoints/mobilenet_v2-b0353104.pth"
        opt.epochs = 50
        opt.batch_size = 24
        opt.multi_scale = False

    if opt.type == "smvocs":
        opt.cfg = 'models/mobile-yolo5s_voc.yaml'
        opt.data = "data/voc.yaml"
        opt.name = opt.type
        opt.weights = "/root/.cache/torch/checkpoints/mobilenet_v2-b0353104.pth"
        opt.epochs = 50
        opt.batch_size = 24
        opt.multi_scale = False
        opt.sl = 6e-4
        hyp["sl"] = opt.sl

    if opt.type == "fsmvocs05":
        opt.cfg = 'models/mobile-yolo5s_voc.yaml'
        opt.data = "data/voc.yaml"
        opt.name = opt.type
        opt.weights = "outputs/smvocs/weights/pruned_5.pt"
        opt.epochs = 20
        opt.batch_size = 24
        opt.multi_scale = False
        opt.ft = True

    if opt.type == "fsmvocs":
        opt.cfg = 'models/mobile-yolo5s_voc.yaml'
        opt.data = "data/voc.yaml"
        opt.name = opt.type
        opt.weights = "outputs/smvocs/weights/pruned_auto.pt"
        opt.epochs = 20
        opt.batch_size = 24
        opt.multi_scale = False
        opt.ft = True

    if opt.type == "dmvocs":
        opt.cfg = 'models/mobile-yolo5s_voc.yaml'
        opt.data = "data/voc.yaml"
        opt.name = opt.type
        opt.weights = "/root/.cache/torch/checkpoints/mobilenet_v2-b0353104.pth"
        opt.epochs = 50
        opt.batch_size = 24
        opt.multi_scale = False
        opt.dist = True
        opt.t_weights = "outputs/voc/weights/best_voc.pt"
        hyp["dist"] = 1

    if opt.type == "dvocs_l":
        opt.cfg = 'models/yolov5s_voc.yaml'
        opt.data = "data/voc.yaml"
        opt.name = opt.type
        opt.weights = "/data/checkpoints/yolov5/yolov5s.pt"
        opt.epochs = 50
        opt.batch_size = 24
        opt.multi_scale = False
        opt.dist = True
        opt.t_weights = "outputs/vocl/weights/best_vocl.pt"
        hyp["dist"] = 1

    if opt.type == "dmvocs_l":
        opt.cfg = 'models/mobile-yolo5s_voc.yaml'
        opt.data = "data/voc.yaml"
        opt.name = opt.type
        opt.weights = "/root/.cache/torch/checkpoints/mobilenet_v2-b0353104.pth"
        opt.epochs = 50
        opt.batch_size = 24
        opt.multi_scale = False
        opt.dist = True
        opt.t_weights = "outputs/vocl/weights/best_vocl.pt"
        hyp["dist"] = 1

    if opt.type == "dfmvocs":
        opt.cfg = 'models/mobile-yolo5s_voc.yaml'
        opt.data = "data/voc.yaml"
        opt.name = opt.type
        opt.weights = "/root/.cache/torch/checkpoints/mobilenet_v2-b0353104.pth"
        opt.epochs = 100
        opt.batch_size = 24
        opt.multi_scale = False
        opt.dist = True
        opt.t_weights = "outputs/vocs/weights/best_vocs.pt"
        opt.d_feature = True
        hyp["dist"] = 1.0

    if opt.type == "dfsmvocs":
        opt.cfg = 'models/mobile-yolo5s_voc.yaml'
        opt.data = "data/voc.yaml"
        opt.name = opt.type
        opt.weights = "/root/.cache/torch/checkpoints/mobilenet_v2-b0353104.pth"
        opt.epochs = 50
        opt.batch_size = 24
        opt.multi_scale = False
        opt.dist = True
        opt.t_weights = "outputs/vocs/weights/best_vocs.pt"
        opt.d_feature = True
        hyp["dist"] = 1.0

    if opt.type == "dfmvocs_l":
        opt.cfg = 'models/mobile-yolo5s_voc.yaml'
        opt.data = "data/voc.yaml"
        opt.name = opt.type
        opt.weights = "/root/.cache/torch/checkpoints/mobilenet_v2-b0353104.pth"
        opt.epochs = 50
        opt.batch_size = 24
        opt.multi_scale = False
        opt.dist = True
        opt.t_weights = "outputs/vocl/weights/best_vocl.pt"
        opt.d_feature = True
        hyp["dist"] = 1.0

    if opt.type == "dtamvocs":
        opt.cfg = 'models/mobile-yolo5s_voc.yaml'
        opt.data = "data/voc.yaml"
        opt.name = opt.type
        opt.weights = "/root/.cache/torch/checkpoints/mobilenet_v2-b0353104.pth"
        opt.epochs = 50
        opt.batch_size = 24
        opt.multi_scale = False
        opt.dist = True
        opt.t_weights = "outputs/dvocs_l/weights/best_dvocs_l.pt"
        opt.d_feature = True
        hyp["dist"] = 1.0

    if opt.type == "domvocs":
        # TODO
        opt.cfg = 'models/mobile-yolo5s_voc.yaml'
        opt.data = "data/voc.yaml"
        opt.name = opt.type
        opt.weights = "/root/.cache/torch/checkpoints/mobilenet_v2-b0353104.pth"
        opt.epochs = 50
        opt.batch_size = 24
        opt.multi_scale = False
        opt.dist = True
        opt.d_online = True
        opt.t_weights = "outputs/voc/weights/best_voc.pt"
        hyp["dist"] = 1

    if opt.nw is None:
        nw = min([os.cpu_count(), opt.batch_size if opt.batch_size > 1 else 0, 8])  # number of workers
    else:
        nw = opt.nw


    print("Using", opt.type, "default config")

    rdir = 'outputs' + os.sep + opt.name
    wdir = rdir + os.sep + 'weights' + os.sep  # weights dir
    os.makedirs(wdir, exist_ok=True)
    last = wdir + 'last.pt'
    best = wdir + 'best.pt'
    if opt.resume:
        opt.weights = last
    results_file = os.path.join(rdir, 'results.txt')

    opt.cfg = check_file(opt.cfg)  # check file
    opt.data = check_file(opt.data)  # check file
    print(hyp)
    print(opt)
    opt.img_size.extend([opt.img_size[-1]] * (2 - len(opt.img_size)))  # extend to 2 sizes (train, test)
    device = torch_utils.select_device(opt.device, apex=mixed_precision, batch_size=opt.batch_size)
    print("Using device: ", device)
    if device.type == 'cpu':
        mixed_precision = False

    # Train
    if not opt.evolve:
        tb_writer = SummaryWriter(comment=opt.name)
        print('Start Tensorboard with "tensorboard --logdir=runs", view at http://localhost:6006/')
        train(hyp)

    # Evolve hyperparameters (optional)
    else:
        tb_writer = None
        opt.notest, opt.nosave = True, True  # only test/save final epoch
        if opt.bucket:
            os.system('gsutil cp gs://%s/evolve.txt .' % opt.bucket)  # download evolve.txt if exists

        for _ in range(10):  # generations to evolve
            if os.path.exists('evolve.txt'):  # if evolve.txt exists: select best hyps and mutate
                # Select parent(s)
                parent = 'single'  # parent selection method: 'single' or 'weighted'
                x = np.loadtxt('evolve.txt', ndmin=2)
                n = min(5, len(x))  # number of previous results to consider
                x = x[np.argsort(-fitness(x))][:n]  # top n mutations
                w = fitness(x) - fitness(x).min()  # weights
                if parent == 'single' or len(x) == 1:
                    # x = x[random.randint(0, n - 1)]  # random selection
                    x = x[random.choices(range(n), weights=w)[0]]  # weighted selection
                elif parent == 'weighted':
                    x = (x * w.reshape(n, 1)).sum(0) / w.sum()  # weighted combination

                # Mutate
                mp, s = 0.9, 0.2  # mutation probability, sigma
                npr = np.random
                npr.seed(int(time.time()))
                g = np.array([1, 1, 1, 1, 1, 1, 1, 0, .1, 1, 0, 1, 1, 1, 1, 1, 1, 1])  # gains
                ng = len(g)
                v = np.ones(ng)
                while all(v == 1):  # mutate until a change occurs (prevent duplicates)
                    v = (g * (npr.random(ng) < mp) * npr.randn(ng) * npr.random() * s + 1).clip(0.3, 3.0)
                for i, k in enumerate(hyp.keys()):  # plt.hist(v.ravel(), 300)
                    hyp[k] = x[i + 7] * v[i]  # mutate

            # Clip to limits
            keys = ['lr0', 'iou_t', 'momentum', 'weight_decay', 'hsv_s', 'hsv_v', 'translate', 'scale', 'fl_gamma']
            limits = [(1e-5, 1e-2), (0.00, 0.70), (0.60, 0.98), (0, 0.001), (0, .9), (0, .9), (0, .9), (0, .9), (0, 3)]
            for k, v in zip(keys, limits):
                hyp[k] = np.clip(hyp[k], v[0], v[1])

            # Train mutation
            results = train(hyp.copy())

            # Write mutation results
            print_mutation(hyp, results, opt.bucket)

            # Plot results
            # plot_evolution_results(hyp)
