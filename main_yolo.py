import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torch.optim as optim
import os
import random
import cv2
import math 
import dataset 
import darknet
import detect 
import distiller 
import argparse
# Frequently used pruning package: 
import utility as utl 
import torchnet.meter as tnt 
import performance_tracker as pt 
import checkpoint as ckpt 
import summary
import sensitivity_analysis as sa 
import collector
import logging
import config
import train
import evaluate 
# End here.
from functools import partial
from torch.autograd import Variable
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from collections import defaultdict, OrderedDict
from tqdm import tqdm

# Add dataset class here (should normalize the cx,cy to be the offset with respect to the grid cell 
# and the weight and lenght should devide by the images width, lengh, respectively)
# Then build the darket here, jitter option is ignored (since dataset includes all augumentation procedure, 
# and the rescore is also ignores as it is just used for the switch function in C.)
# Whether the final output function we use, I think we should prove it by the empirical results ( sigmoid v.s. linear)
# Transfer function for testing at the end of this python code.
# 

# Set logger to make sure the training information will not lose.
msglogger = logging.getLogger()
def _init_logger(args, script_dir):
    global msglogger
    if script_dir is None or not hasattr(args, "output_dir") or args.output_dir is None:
        msglogger.logdir = None
        return None

    if not os.path.exists(args.output_dir):
       os.makedirs(args.output_dir)
    name = args.name
    if args.name == '':
        name = args.stage + '_' + args.arch + "_" + args.dataset + "_log"

    # ***
    # This line may raise alarm, but it doesn't have influence on the execution.
    # ***
    msglogger = config.config_pylogger(os.path.join(script_dir, 'logging.conf'),
                                             name, args.output_dir, args.verbose)
    # Log various details about the execution environment.  It is sometimes useful
    # to refer to past experiment executions and this information may be useful.
    #apputils.log_execution_env_state(
    #    filter(None, [args.compress, args.qe_stats_file]),  # remove both None and empty strings
    #    msglogger.logdir)
    config.log_execution_env_state(
        filter(None, [args.compress, args.qe_stats_file]),  # remove both None and empty strings
        msglogger.logdir)
    msglogger.debug("Distiller: %s", distiller.__version__)
    return msglogger.logdir


# Sensitivity analysis will be suppportive in the near future.
def sensitivity_analysis(model, criterion, device, args, sparsities, logger):
    # TODO how to do the sensitivity analysis in this regression problem?

    # This sample application can be invoked to execute Sensitivity Analysis on your
    # model.  The ouptut is saved to CSV and PNG. 
    # criterion, device, num_classes, loggers, args=None, parameter_name=None
    msglogger.info("Running Sensitivity Test (analysis).")
    test_fnc = partial(test, criterion=criterion, device=device, num_classes=num_classes, args=args, loggers=logger)
    which_params = [(param_name, torch.std(param).item()) for param_name, param in model.named_parameters()]
    sensitivity, eval_scores_dict = sa.perform_sensitivity_analysis(model, net_params=which_params, sparsities=sparsities,
                                                  test_func=test_fnc, group=args.sensitivity)
    if not os.path.isdir('sensitivity_analysis'):
        os.mkdir('sensitivity_analysis')
    name = '_' + args.arch + '_' + args.dataset
    #sa.sensitivities_to_png(sensitivity, os.path.join('sensitivity_analysis', 'sensitivity_'+args.sensitivity + name +'.png'))
    #sa.sensitivities_to_csv(sensitivity, os.path.join('sensitivity_analysis', 'sensitivity_'+args.sensitivity + name+'.csv'))
    sa.sensitivities_to_png(sensitivity, os.path.join(msglogger.logdir, 'sensitivity_'+args.sensitivity + name +'.png'))
    sa.sensitivities_to_csv(sensitivity, os.path.join(msglogger.logdir, 'sensitivity_'+args.sensitivity + name+'.csv'))
    sa._pickle_eval_scores_dict(eval_scores_dict, os.path.join(msglogger.logdir, 'greedy_selection_eval_scores_dict.pkl'))


# Plot the prediction bounding box in the testing images.
# TODO if the bounding box exceeds the image boundary, how to deal with it??
def visualize_boxes(image_bgr, boxes, class_names, probs, name_bgr_dict=None, line_thickness=2):
    if name_bgr_dict is None:
        name_bgr_dict = detect.VOC_CLASS_BGR

    image_boxes = image_bgr.copy()
    for box, class_name, prob in zip(boxes, class_names, probs):
        # Draw box on the image.
        left_top, right_bottom = box
        left, top = int(left_top[0]), int(left_top[1])
        right, bottom = int(right_bottom[0]), int(right_bottom[1])
        bgr = name_bgr_dict[class_name]
        cv2.rectangle(image_boxes, (left, top), (right, bottom), bgr, thickness=line_thickness)

        # Draw text on the image.
        text = '%s %.2f' % (class_name, prob)
        size, baseline = cv2.getTextSize(text,  cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, thickness=2)
        text_w, text_h = size

        x, y = left, top
        x1y1 = (x, y)
        x2y2 = (x + text_w + line_thickness, y + text_h + line_thickness + baseline)
        cv2.rectangle(image_boxes, x1y1, x2y2, bgr, -1)
        cv2.putText(image_boxes, text, (x + line_thickness, y + 2*baseline + line_thickness),
            cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.4, color=(255, 255, 255), thickness=1, lineType=8)
    return image_boxes


def test(model, loggers, args=None, parameter_name=None, gpu_id=0):
    model.eval()
    # Different from _validate, I want this testing function can apply on COCO dataset and generate the visualized results:
    voc_class_names = list(detect.VOC_CLASS_BGR.keys())
    targets = defaultdict(list)
    preds = defaultdict(list)
    msglogger.info("Preparing ground-truth data...")
    # Load annotations from label file.
    annotations = []
    test_path = os.path.join(args.label_dir, 'test')
    test_files = os.listdir(test_path)
    # TODO if existing multiple files, use iteratively read the filename.
    #label_path = os.path.join(test_path, test_files)
    for file_name in test_files:
        label_path = os.path.join(test_path, file_name)
        with open(label_path, 'r') as f:
            lines = f.readlines()
        for line in lines:
            anno = line.strip().split()
            annotations.append(anno)
    
    # Prepare ground-truth data. 
    image_fnames = []
    for anno in annotations:
        filename = anno[0]
        image_fnames.append(filename)
        num_boxes = (len(anno) - 1) // 5
        for b in range(num_boxes):
            x1 = int(anno[5*b + 1])
            y1 = int(anno[5*b + 2])
            x2 = int(anno[5*b + 3])
            y2 = int(anno[5*b + 4])
            class_label = int(anno[5*b + 5])
            class_name = voc_class_names[class_label]
            targets[(filename, class_name)].append([x1, y1, x2, y2])

    msglogger.info("Visualize the empirical results...")
    # YOLO default configration for testing, it can show more convinced/fair results.
    yolo = detect.YOLODetector(model, gpu_id=gpu_id, conf_thresh=0.005, prob_thresh=0.005, nms_thresh=0.45)
    count = 0 

    #os.makedirs(os.path.join(msglogger.logdir,"visualization"), exist_ok=True )
    save_path = os.path.join(msglogger.logdir, "visualization")
    os.makedirs(save_path, exist_ok=True)
    # Detect objects with the model.
    for filename in tqdm(image_fnames):
        # TODO should be modified:
        # Use log directory to save the output picture.
        path = os.path.join(save_path, str(count)+'.png')
        image_path = os.path.join(args.data_dir, filename)
        image = cv2.imread(image_path)
        boxes, class_names, probs = yolo.detect(image)
        if count < 100:
            image_box = visualize_boxes(image, boxes, class_names, probs)
            cv2.imwrite(path, image_box)
        count +=1
        for box, class_name, prob in zip(boxes, class_names, probs):
            x1y1, x2y2 = box
            x1, y1 = int(x1y1[0]), int(x1y1[1])
            x2, y2 = int(x2y2[0]), int(x2y2[1])
            preds[class_name].append([filename, prob, x1, y1, x2, y2])

    msglogger.info("Evaluate the detection result...")
    aps = evaluate.evaluate_map(preds, targets, class_names=voc_class_names)
    return np.mean(aps)


if __name__ == "__main__":
    # Classifiy the argparse is the first priority thing we TODO.
    # Imagenet dataset path: /home/swai01/imagenet_datasets/raw-data
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    parser.add_argument('--dataset', type=str, required=True, help='Specify the dataset for creating model and loaders')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to the datafolder and it includes train/test ')
    parser.add_argument('--label_dir', type=str, required=True, help='Path to the labelfolder and it includes train/test ')
    parser.add_argument('--config_path', type=str, required=True, help='Pretrained weight file downloaded in your local machine.')
    parser.add_argument('--weight_path', type=str, required=True, help='Pretrained weight file downloaded in your local machine.')
    parser.add_argument('--stage', type=str, required=True, help='The first learning procedue of your yaml file configuration' 
                                                                 'and it supports combine, naive, retrain, sensitivity and test')
    parser.add_argument('--image_size', type=int, required=True)
    parser.add_argument('--compress', type=str, help='Path to compress configure file.')
    parser.add_argument('--arch', '-a', type=str, required=True, help='Name of used Architecture.')
    parser.add_argument('--name', default='', type=str, help='Save file name.')
    parser.add_argument('--pretrained', default=False, action='store_true', help="using pretrained model from imagenet.")
    parser.add_argument('--resume_from', default=False, action='store_true', help="using the ckpt from local trained model.")    
    parser.add_argument('--output_dir', type=str, default='/home/bwtseng/Downloads/object-detection/log/', help='Path to your saved file. ')
    parser.add_argument('--model_path', type=str, default="", help="Locally pretrained model.")
    # Training arguments:
    parser.add_argument('--apex', action='store_true', default=False, help='whether to use the Pytorch accelerator based on NVIDA APEX')
    parser.add_argument('--batch_size', '-b', type=int, default=32, help='batch_size for the dataloaders.')    
    parser.add_argument('--lr_pretrain', type=float, default=0.001, help="Initial learning rate for pretrain phase.")
    parser.add_argument('--lr_prune', type=float, default=0.001, help="Initial learning rate for pruner.")
    parser.add_argument('--lr_retrain', type=float, default=0.001, help="Initial learning rate for retrain phase.")
    parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float, metavar='W', help='weight decay (default: 1e-4)')
    parser.add_argument('--learning-rate-decay', '--lrd', default=0.7, type=float, metavar='W', help='learning rate decay (default: 0.7)')
    parser.add_argument('--workers', default=4, type=int, help='number of dataloader workers')    
    parser.add_argument('--cpu', default=False, action='store_true', help="If GPU is full process.")
    parser.add_argument('--train', default=False, action='store_true', help="Training phase.")
    parser.add_argument('--test', default=False, action='store_true', help="Testing phase.")
    parser.add_argument('--num_best_scores', default=1, type=int, help="num_best_score")
    parser.add_argument('--epoch', default=100, type=int, help="Epoch")
    parser.add_argument('--parallel', default=False, action='store_true', help="Parallel or not")
    parser.add_argument('--print_freq', type=int, default=100, help='Record frequency')
    
    # Sensitivity analysis arguments:
    parser.add_argument('--sensitivity', '--sa', choices=['element', 'filter', 'channel'],
                        type=lambda s: s.lower(), help='test the sensitivity of layers to pruning')
    parser.add_argument('--sensitivity_range', '--sr', type=float, nargs=3, default=[0.0, 0.95, 0.05], #
                        help='an optional parameter for sensitivity testing '
                             'providing the range of sparsities to test.\n'
                             'This is equivalent to creating sensitivities = np.arange(start, stop, step)')

    # Below are two useful training mechansim, proposed by facebook and google respectively. (terminology: mix-up and warm-up)
    parser.add_argument('--alpha', default=0.0, type=float, help='Parameter of mixup data'
                                                                            'function.')
    parser.add_argument('--mixup', default=False, action='store_true', help='Turn on data'
                                                                            'augumentation mechanisms.')    
    parser.add_argument('--warmup', default=False, action='store_true', help='Wrap optimizer')
    parser.add_argument('--smooth_eps', default=0.0, type=float, help='Parameter of smooth factor.')
    # Adversarial attack argument (left for the robustness version.)
    parser.add_argument('--epsilon', default=8.0, type=float, help='PGD model parameter')
    parser.add_argument('--num_steps', default=10, type=int, help='PGD model parameter')
    parser.add_argument('--step_size', default=2.0, type=float, help='PGD model parameter')
    parser.add_argument('--random_start', default=True, type=bool, help='PGD model parameter')
    #Natural training will not include this line, if you want robustness training, please see more detail in main_adv.py
    #parser.add_argument('--adv_train', default=False, action='store_true', help='Turn on adversarial training.')
    #parser.add_argument('--robustness', action='store_true', default=False, help='test model robustness using naive noise perturbance, we may adopt outlier sampling in the near future.')
    parser.add_argument('--robustness', default=0.0, type=float, help='test model robustness using naive noise perturbance, we may adopt outlier sampling in the near future.')
    parser.add_argument('--post_qe_test', default=False, action='store_true', help='whether testing with quantization model.')
    parser.add_argument('--verbose', '-v', action='store_true', help='Emit debug log messages')
    parser.add_argument('--log_params_histograms', action='store_true', default=False,
                        help='log the parameter tensors histograms to file (WARNING: this can use significant disk space)')    
    distiller.quantization.add_post_train_quant_args(parser, add_lapq_args=True)
    args = parser.parse_args()
    print("\n Argument property: {}.".format(args))
    log_dir = _init_logger(args, script_dir=os.path.dirname(__file__))
    if not log_dir:
        pylogger = tflogger = NullLogger()
    
    else: 
        # **********************************************************************
        # Wrap the msglogger into this two modules, one for tensorboard visualizarion
        # the other is just send the msglogger to connect other summary functions.
        # **********************************************************************
        tflogger = config.TensorBoardLogger(msglogger.logdir)
        pylogger = config.PythonLogger(msglogger)

    if args.train:
        # Create dataloaders and the table for dataset size information: 
        train_labels_name = os.listdir(os.path.join(args.label_dir, 'train'))
        test_label_name = os.listdir(os.path.join(args.label_dir, 'test'))
        train_labels = []
        for i in train_labels_name:
            train_labels.append(os.path.join(args.label_dir, 'train', i))
        test_labels = os.path.join(args.label_dir, 'test', test_label_name[0])
        dataloaders, dataset_sizes = dataset.create_dataloader(args, args.data_dir, train_labels, test_labels)

    # Create the darknet and load weights here:
    model = darknet.Darknet(args.config_path, img_size=args.image_size)
    if args.pretrained:
        model.load_darknet_weights(args.weight_path)
    
    model.arch = args.arch 
    model.dataset = args.dataset
    model.input_shape = (1, 3, args.image_size, args.image_size) # For channel first.
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Leverage {} device to run this task.".format(device))
    if args.cpu: 
        device = torch.device("cpu")
    model.to(device)


    optimizer = None 
    compress_scheduler = None
    if args.train:
        if args.resume_from:
            # Load checkpoint for locally pre-trained model.
            try:
                model, compress_scheduler, optimizer, start_epoch = ckpt.load_checkpoint(
                    model,  args.model_path, model_device=device)  
            except:
                model.load_state_dict(torch.load(args.model_path))

            optimizer = None
            if optimizer is None: 
                optimizer = optim.SGD(model.parameters(), lr=args.lr_pretrain, momentum=0.9, weight_decay=args.weight_decay)
                print("Do build optimizer")

            store_mask = compress_scheduler.zeros_mask_dict
            compress_scheduler = None
            if compress_scheduler is None:
                if args.compress:
                    compress_scheduler = config.file_config(model, optimizer, args.compress, None, None)
                    # recover the mask dict
                    for name, mask in store_mask.items():
                        compress_scheduler.zeros_mask_dict[name].mask = store_mask[name].mask
                    print("Do load compress")
                    #if args.stage: 
                    #    compress_scheduler.retrain_phase = True 
            
            model.to(device)
            print("\nStart Training")
        else:
            optimizer = optim.SGD(model.parameters(), lr=args.lr_pretrain, momentum=0.9, weight_decay=args.weight_decay)
            # Setting Learning decay scheduler, that is, decay LR by a factor of 0.1 every 7 epochs
            # For example: exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.7)
            # Note that all of this can be configured using the YAML file.
            if args.compress:
                compress_scheduler = config.file_config(model, optimizer, args.compress, None, None)
                #if args.stage: 
                #    compress_scheduler.retrain_phase = True 
            #else:
            #    lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.7)
            model.to(device)
            print("\nStart Training")
        

        # ***************************************************
        # Print the initial sparsity of this model, and please check whether the pruning 
        # weight name is correct or not. 
        # ***************************************************
        t, total = summary.weights_sparsity_tbl_summary(model, return_total_sparsity=True)
        print("\nParameters Table: {}".format(str(t)))
        print("\nSparsity: {}.".format(total))

        tracker = pt.SparsityAccuracyTracker(args.num_best_scores)
        tracker.reset()
        if args.apex: 
            model, optimizer = amp.initialize(model, optimizer, opt_level="O1") 
        
        for epoch in range(args.epoch):
            print("\n")
            #print(compress_scheduler.policies[200])
            nat_loss, tracker = train.trian_validate_with_scheduling(args, model, optimizer, compress_scheduler, device, dataloaders, dataset_sizes,
                                                                                loggers = [tflogger, pylogger], tracker=tracker, epoch=epoch)         

    # TODO it must implemented for the regression problem:                                                    
    elif args.sensitivity:
        # If specifying the model path, users musr have their own local pre-defined model.
        # Sometimes we use torch vision pretrined model instead, and its implementation is included in model_zoo directory.
        if args.model_path:
            #mdoel = ckpt.load_lean_checkpoint(model, args.model_path, model_device=device)
            try :
                mdoel = ckpt.load_lean_checkpoint(model, args.model_path, model_device=device)
            except:
                model.load_state_dict(torch.load(args.model_path))
                #raise ValueError("Please input correct model path.")

            msglogger.info("Successfully load model from the checkpoint {%s}", args.model_path)

        #TODO should modify the sensitiviy analysis function:
        sensitivities = np.arange(*args.sensitivity_range)
        sensitivity_analysis(model, criterion, device, num_classes, args, sensitivities, logger=[tflogger, pylogger])

    # TODO should i add a new code to generate the first template yaml file?
    #elif args.generate_yaml_template:
    #    pass
    
    # TEST functionb here.
    else: 
        # Note: load_lean_checkpoint is implemented for testing phase only.
        if args.model_path:
            try :
                try: 
                    epoch = 0
                    mdoel = ckpt.load_lean_checkpoint(model, args.model_path, model_device=device)
                except:
                    #raise ValueError("Please input correct model path !")
                    model.load_state_dict(torch.load(args.model_path))

            except:
                raise ValueError("Can not load your checkpoint file.")

            msglogger.info("Successfully load model from the checkpoint {%s}", args.model_path)

        epoch = 0
        utl.log_weights_sparsity(model, epoch, loggers=[tflogger, pylogger])
        if args.test:
            mAP = test(model, [tflogger, pylogger], args)
            """
            if args.robustness:
                iter_list = np.arange(0.0, 3, 0.05)
                #iter_list = [0]
                acc_list = []
                #iter_list= [1]
                for fac in iter_list:
                    args.robustness = fac 
                    nat_top1, nat_loss = test(model, criterion, device, num_classes, [tflogger, pylogger] ,args)
                    acc_list.append(nat_top1)

                Matrix = {}
                Matrix['Noise_factor'] = iter_list
                Matrix['Accuracy']= acc_list
                final = pd.DataFrame(Matrix)
                final.to_csv(os.path.join(log_dir, args.stage+'.csv'), index=False)
            else:
                nat_top1, nat_loss = test(model, criterion, device, num_classes, [tflogger, pylogger], args)
            """
        # TODO Remove quantization testing so far, but it's easy to implement (distiller's support package.)


"""
if __name__ == "__main__":
    net = darknet.Darknet('yolov1-tiny.cfg', img_size=448).cuda()
    net.load_darknet_weights("tiny-yolov1.weights")

    #net = Darknet('yolov1.cfg', img_size=448).cuda()
    #net.load_darknet_weights("yolov1.weights") 

    #x = torch.rand((5, 3, 448, 448)).cuda()
    #y = torch.rand((5, 7, 7, 30)).cuda()
    #net.forward(x, y)image_path = 'data/test_samples/009046.jpg'

    # Paths to input/output images.
    image_path = 'test_samples/000844.jpg'
    image_path = 'test_samples/009046.jpg'
    #image_path = 'test_samples/001177.jpg'
    #image_path = 'test_samples/002878.jpg'
    out_path = 'results/'
    # Path to the yolo weight.
    model_path = 'weights/yolo/model_best.pth'
    # GPU device on which yolo is loaded.
    label_path = 'voc2007test.txt'
    image_dir = 'VOCdevkit/allimgs/JPEGImages'
    gpu_id = 0
    # Load model.
    #yolo = YOLODetector(model_path, gpu_id=gpu_id, conf_thresh=0.3, prob_thresh=0.1, nms_thresh=0.5)
    yolo = detect.YOLODetector(net, gpu_id=gpu_id, conf_thresh=0.1, prob_thresh=0.1, nms_thresh=0.5)
    # Load image.
    #image_bgr = cv2.imread(image_path)    
    ###
    voc_class_names = list(detect.VOC_CLASS_BGR.keys())
    targets = defaultdict(list)
    preds = defaultdict(list)

    print('Preparing ground-truth data...')

    # Load annotations from label file.
    annotations = []
    with open(label_path, 'r') as f:
        lines = f.readlines()
    for line in lines:
        anno = line.strip().split()
        annotations.append(anno)

    # Prepare ground-truth data.
    image_fnames = []
    for anno in annotations:
        filename = anno[0]
        image_fnames.append(filename)
        num_boxes = (len(anno) - 1) // 5
        for b in range(num_boxes):
            x1 = int(anno[5*b + 1])
            y1 = int(anno[5*b + 2])
            x2 = int(anno[5*b + 3])
            y2 = int(anno[5*b + 4])

            class_label = int(anno[5*b + 5])
            class_name = voc_class_names[class_label]

            targets[(filename, class_name)].append([x1, y1, x2, y2])

    print('Predicting...')

    # yolo = YOLODetector(model_path, gpu_id=gpu_id, conf_thresh=0.1, prob_thresh=0.1, nms_thresh=0.5)
    # Best score 49.475, which follow the threshold setting from darknet official website.
    # Load YOLO model.
    yolo = detect.YOLODetector(net, gpu_id=gpu_id, conf_thresh=0.005, prob_thresh=0.005, nms_thresh=0.45)
    count = 0 
    # Detect objects with the model.
    for filename in tqdm(image_fnames):
        path = os.path.join(out_path, str(count)+'.png')
        image_path = os.path.join(image_dir, filename)
        image = cv2.imread(image_path)
        boxes, class_names, probs = yolo.detect(image)
        if count < 100:
            image_box = visualize_boxes(image, boxes, class_names, probs)
            cv2.imwrite(path, image_box)
        count +=1
        for box, class_name, prob in zip(boxes, class_names, probs):
            x1y1, x2y2 = box
            x1, y1 = int(x1y1[0]), int(x1y1[1])
            x2, y2 = int(x2y2[0]), int(x2y2[1])
            preds[class_name].append([filename, prob, x1, y1, x2, y2])

    print('Evaluate the detection result...')
    evaluate(preds, targets, class_names=voc_class_names)


    # Training example:
    image_dir = 'VOCdevkit/allimgs/JPEGImages/'
    label_txt = ['voc2007.txt', 'voc2012.txt']
    dataset = dataset.VOCDataset(False, image_dir, label_txt)
    data_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)
    for idx, (img, target) in enumerate(data_loader):
        #print(img.shape)
        #print(target.shape)
        img = Variable(img).float()
        target = Variable(target).float()
        img, target = img.cuda(), target.cuda()
        loss = net(img, target)
        print(loss[0])
    #boxes, class_names, probs = yolo.detect(image_bgr)
    # Visualize.
    #image_boxes = visualize_boxes(image_bgr, boxes, class_n
    #cv2.imwrite(out_path, image_boxes)

"""
