import torch 
import torch.nn as nn 
import torch.nn.functional as F 
from torch.nn.parameter import Parameter
from torch.autograd import Variable
import numpy as np 
import torchex.nn as exnn 
import math 
import collections 
from itertools import repeat

def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.Iterable):
            return x
        return tuple(repeat(x, n))
    return parse

_single = _ntuple(1)
_pair = _ntuple(2)
_triple = _ntuple(3)
_quadruple = _ntuple(4)
class Conv2dLocal(nn.Module):
    def __init__(self, in_height, in_width, in_channels, out_channels,
                 kernel_size, stride=1, padding=0, bias=True, dilation=1):
        super(Conv2dLocal, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
 
        self.in_height = in_height
        self.in_width = in_width
        # Compute output FMPs first, make sure that the correct parameter shape of this local convolution layer.
        self.out_height = int(math.floor(
            (in_height + 2 * self.padding[0] - self.dilation[0] * (self.kernel_size[0] - 1) - 1) / self.stride[0] + 1))
        self.out_width = int(math.floor(
            (in_width + 2 * self.padding[1] - self.dilation[1] * (self.kernel_size[1] - 1) - 1) / self.stride[1] + 1))
        # Initialization
        self.weight = Parameter(torch.Tensor(
            self.out_height, self.out_width,
            out_channels, in_channels, *self.kernel_size))
        if bias:
            self.bias = Parameter(torch.Tensor(
                out_channels, self.out_height, self.out_width))
        else:
            self.register_parameter('bias', None)
 
        self.reset_parameters()
    
    # Follow the rule from ConvND
    def reset_parameters(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1. / math.sqrt(n)
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)
 
    def __repr__(self):
        s = ('{name}({in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}')
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'

        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'

        if self.bias is None:
            s += ', bias=False'
        s += ')'

        return s.format(name=self.__class__.__name__, **self.__dict__)
 
    def forward(self, input):
        return conv2d_local(
            input, self.weight, self.bias, stride=self.stride,
            padding=self.padding, dilation=self.dilation)


def conv2d_local(input, weight, bias=None, padding=0, stride=1, dilation=1):
    if input.dim() != 4:
        raise NotImplementedError("Input Error: Only 4D input Tensors supported (got {}D)".format(input.dim()))
    if weight.dim() != 6:
        # outH x outW x outC x inC x kH x kW
        raise NotImplementedError("Input Error: Only 6D weight Tensors supported (got {}D)".format(weight.dim()))
 
    outH, outW, outC, inC, kH, kW = weight.size()
    kernel_size = (kH, kW)
 
    # N x [inC * kH * kW] x [outH * outW]
    cols = F.unfold(input, kernel_size, dilation=dilation, padding=padding, stride=stride)
    cols = cols.view(cols.size(0), cols.size(1), cols.size(2), 1).permute(0, 2, 3, 1)
 
    out = torch.matmul(cols, weight.view(outH * outW, outC, inC * kH * kW).permute(0, 2, 1))
    out = out.view(cols.size(0), outH, outW, outC).permute(0, 3, 1, 2)
 
    if bias is not None:
        out = out + bias.expand_as(out)
    return out


class Detection(nn.Module):
    def __init__(self, feature_size=7, num_bboxes=2, num_classes=20, sqrt=True, lambda_coob_scale=1.0, lambda_noobj=0.5, lambda_class_scale=1.0,  lambda_coord=5.0):
        """ Constructor.
        Args:
            feature_size: (int) size of input feature map.
            num_bboxes: (int) number of bboxes per each cell.
            num_classes: (int) number of the object classes.
            lambda_coord: (float) weight for bbox location/size losses.
            lambda_noobj: (float) weight for no-objectness loss.
        """
        super(Detection, self).__init__()

        self.S = feature_size
        self.B = num_bboxes
        self.C = num_classes

        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj
        self.lambda_class_scale = lambda_class_scale
        self.lambda_coob_scale = lambda_coob_scale
        self.sqrt = sqrt

    def compute_iou(self, bbox1, bbox2):
        """ 
        Compute the IoU (Intersection over Union) of two set of bboxes, each bbox format: [x1, y1, x2, y2].
        (Different from Bo-Min's implementation, this iou implementation enable training with batchsize higher than 1.)
        Args:
            bbox1: (Tensor) bounding bboxes, sized [N, 4].
            bbox2: (Tensor) bounding bboxes, sized [M, 4].
        Returns:
            (Tensor) IoU, sized [N, M].
        """
        N = bbox1.size(0)
        M = bbox2.size(0)
        # Compute left-top coordinate of the intersections
        lt = torch.max(
            bbox1[:, :2].unsqueeze(1).expand(N, M, 2), # [N, 2] -> [N, 1, 2] -> [N, M, 2]
            bbox2[:, :2].unsqueeze(0).expand(N, M, 2)  # [M, 2] -> [1, M, 2] -> [N, M, 2]
        )
        # Conpute right-bottom coordinate of the intersections
        rb = torch.min(
            bbox1[:, 2:].unsqueeze(1).expand(N, M, 2), # [N, 2] -> [N, 1, 2] -> [N, M, 2]
            bbox2[:, 2:].unsqueeze(0).expand(N, M, 2)  # [M, 2] -> [1, M, 2] -> [N, M, 2]
        )
        # Compute area of the intersections from the coordinates
        wh = rb - lt   # width and height of the intersection, [N, M, 2]
        wh[wh < 0] = 0 # clip at 0
        inter = wh[:, :, 0] * wh[:, :, 1] # [N, M]

        # Compute area of the bboxes
        area1 = (bbox1[:, 2] - bbox1[:, 0]) * (bbox1[:, 3] - bbox1[:, 1]) # [N, ]
        area2 = (bbox2[:, 2] - bbox2[:, 0]) * (bbox2[:, 3] - bbox2[:, 1]) # [M, ]
        area1 = area1.unsqueeze(1).expand_as(inter) # [N, ] -> [N, 1] -> [N, M]
        area2 = area2.unsqueeze(0).expand_as(inter) # [M, ] -> [1, M] -> [N, M]

        # Compute IoU from the areas
        union = area1 + area2 - inter # [N, M, 2]
        iou = inter / union           # [N, M, 2]
        return iou

    def forward(self, pred_tensor, target_tensor=None):
        # Reference: Github page of Bo-Min and xiongzihua.
        """ Compute loss for YOLO training.
        Args:
            pred_tensor: (Tensor) predictions, sized [n_batch, S, S, Bx5+C], 5=len([x, y, w, h, conf]).
            target_tensor: (Tensor) targets, sized [n_batch, S, S, Bx5+C].
        Returns:
            (Tensor): loss, sized [1, ].
        """
        # TODO: Romove redundant dimensions for some Tensors.
        
        if target_tensor is None: 
            return pred_tensor, 0

        
        S, B, C = self.S, self.B, self.C
        N = 5 * B + C    # 5=len([x, y, w, h, conf]

        batch_size = pred_tensor.size(0)
        # Since there is the indicator function in the Yolo's objective function, we first attain the mask with the following code.
        coord_mask = target_tensor[:, :, :, 4] > 0  
        noobj_mask = target_tensor[:, :, :, 4] == 0 
        coord_mask = coord_mask.unsqueeze(-1).expand_as(target_tensor)
        noobj_mask = noobj_mask.unsqueeze(-1).expand_as(target_tensor) 

        coord_pred = pred_tensor[coord_mask].view(-1, N)        
                                                                
        bbox_pred = coord_pred[:, :5*B].contiguous().view(-1, 5)   
        class_pred = coord_pred[:, 5*B:]                        
        
        coord_target = target_tensor[coord_mask].view(-1, N)      
                                       
        bbox_target = coord_target[:, :5*B].contiguous().view(-1, 5)
        class_target = coord_target[:, 5*B:]               


        # Compute loss for the cells with no object bbox.
        noobj_pred = pred_tensor[noobj_mask].view(-1, N)        
        
        noobj_target = target_tensor[noobj_mask].view(-1, N)  
        
        noobj_conf_mask = torch.cuda.ByteTensor(noobj_pred.size()).fill_(0).bool()
        #noobj_conf_mask = torch.bool(noobj_pred.size())
        for b in range(B):
            #noobj_conf_mask[:, 4 + b*5] = 1 
            noobj_conf_mask[:, 4 + b*5] = True 

        noobj_pred_conf = noobj_pred[noobj_conf_mask]     
        noobj_target_conf = noobj_target[noobj_conf_mask] 
        loss_noobj = F.mse_loss(noobj_pred_conf, noobj_target_conf, reduction='sum')

        # Compute loss for the cells with objects.
        coord_response_mask = torch.cuda.ByteTensor(bbox_target.size()).fill_(0).bool()   # [n_coord x B, 5]
        coord_not_response_mask = torch.cuda.ByteTensor(bbox_target.size()).fill_(1).bool() # [n_coord x B, 5]
        bbox_target_iou = torch.zeros(bbox_target.size()).cuda()                    # [n_coord x B, 5], only the last 1=(conf,) is used
        
        #
        cell_size = 1.0 / float(self.S)
        # Choose the predicted bbox having the highest IoU for each target bbox.
        for i in range(0, bbox_target.size(0), B):
            pred = bbox_pred[i:i+B] # predicted bboxes at i-th cell, [B, 5=len([x, y, w, h, conf])]
            pred_xyxy = Variable(torch.FloatTensor(pred.size())) # [B, 5=len([x1, y1, x2, y2, conf])]
            # Because (center_x,center_y)=pred[:, 2] and (w,h)=pred[:,2:4] are normalized for cell-size and image-size respectively,
            # rescale (center_x,center_y) for the image-size to compute IoU correctly.
            
            # TODO Whether width and height should be square?
            pred_xyxy[:,  :2] = pred[:, 2]*cell_size - 0.5 * torch.pow(pred[:, 2:4], 2)
            pred_xyxy[:, 2:4] = pred[:, 2]*cell_size + 0.5 * torch.pow(pred[:, 2:4], 2)
            #target = bbox_target[i] # target bbox at i-th cell. Because target boxes contained by each cell are identical in current implementation, enough to extract the first one.
            
            target = bbox_target[i].view(-1, 5) # target bbox at i-th cell, [1, 5=len([x, y, w, h, conf])]
            target_xyxy = Variable(torch.FloatTensor(target.size())) # [1, 5=len([x1, y1, x2, y2, conf])]
            # Because (center_x,center_y)=target[:, 2] and (w,h)=target[:,2:4] are normalized for cell-size and image-size respectively,
            # rescale (center_x,center_y) for the image-size to compute IoU correctly.
            target_xyxy[:,  :2] = target[:, 2]*cell_size - 0.5 * target[:, 2:4]
            target_xyxy[:, 2:4] = target[:, 2]*cell_size + 0.5 * target[:, 2:4]

            iou = self.compute_iou(pred_xyxy[:, :4], target_xyxy[:, :4]) # [B, 1]
            max_iou, max_index = iou.max(0)
            max_index = max_index.data.cuda()

            coord_response_mask[i+max_index] = 1
            coord_not_response_mask[i+max_index] = 0
            # "we want the confidence score to equal the intersection over union (IOU) between the predicted box and the ground truth"
            # from the original paper of YOLO.
            bbox_target_iou[i+max_index, torch.LongTensor([4]).cuda()] = (max_iou).data.cuda()
        
        bbox_target_iou = Variable(bbox_target_iou).cuda()

        # BBox location/size and objectness loss for the response bboxes.
        bbox_pred_response = bbox_pred[coord_response_mask].view(-1, 5)      # [n_response, 5]
        bbox_target_response = bbox_target[coord_response_mask].view(-1, 5)  # [n_response, 5], only the first 4=(x, y, w, h) are used
        target_iou = bbox_target_iou[coord_response_mask].view(-1, 5)        # [n_response, 5], only the last 1=(conf,) is used
        
        loss_xy = F.mse_loss(bbox_pred_response[:, :2], bbox_target_response[:, :2], reduction='sum')
        
        # Darknet give the option to train yolo without operation root. But we should inherit thier procudure to preserve the original performance in our pruning case.
        # More intuive thinking about this is that: the prediction weights are the root of the original width.
        #if self.sqrt: 
        #    loss_wh = F.mse_loss(bbox_pred_response[:, 2:4], torch.sqrt(bbox_target_response[:, 2:4]), reduction='sum')
        #else:
        #    loss_wh = F.mse_loss(bbox_pred_response[:, 2:4], bbox_target_response[:, 2:4], reduction='sum')
        loss_wh = F.mse_loss(bbox_pred_response[:, 2:4], torch.sqrt(bbox_target_response[:, 2:4]), reduction='sum') 
        loss_obj = F.mse_loss(bbox_pred_response[:, 4], target_iou[:, 4], reduction='sum')
        # Class probability loss for the cells which contain objects.
        loss_class = F.mse_loss(class_pred, class_target, reduction='sum')
        # Total loss
        loss = self.lambda_coord * (loss_xy + loss_wh) + self.lambda_coob_scale * loss_obj + \
               self.lambda_noobj * loss_noobj + self.lambda_class_scale * loss_class
        loss = loss / float(batch_size)
        return pred_tensor, loss

class YOLOLayer(nn.Module):
    """Detection layer for YOLOv3"""
    def __init__(self, anchors, num_classes, img_dim=416):
        super(YOLOLayer, self).__init__()
        self.anchors = anchors
        self.num_anchors = len(anchors)
        self.num_classes = num_classes
        self.ignore_thres = 0.5
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCELoss()
        self.obj_scale = 1
        self.noobj_scale = 100
        self.metrics = {}
        self.img_dim = img_dim
        self.grid_size = 0  # grid size

    def compute_grid_offsets(self, grid_size, cuda=True):
        self.grid_size = grid_size
        g = self.grid_size
        FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
        self.stride = self.img_dim / self.grid_size
        # Calculate offsets for each grid
        self.grid_x = torch.arange(g).repeat(g, 1).view([1, 1, g, g]).type(FloatTensor)
        self.grid_y = torch.arange(g).repeat(g, 1).t().view([1, 1, g, g]).type(FloatTensor)
        self.scaled_anchors = FloatTensor([(a_w / self.stride, a_h / self.stride) for a_w, a_h in self.anchors])
        self.anchor_w = self.scaled_anchors[:, 0:1].view((1, self.num_anchors, 1, 1))
        self.anchor_h = self.scaled_anchors[:, 1:2].view((1, self.num_anchors, 1, 1))

    def forward(self, x, targets=None, img_dim=None):
        # Tensors for cuda support
        FloatTensor = torch.cuda.FloatTensor if x.is_cuda else torch.FloatTensor
        LongTensor = torch.cuda.LongTensor if x.is_cuda else torch.LongTensor
        ByteTensor = torch.cuda.ByteTensor if x.is_cuda else torch.ByteTensor

        self.img_dim = img_dim
        num_samples = x.size(0)
        grid_size = x.size(2)

        prediction = (
            x.view(num_samples, self.num_anchors, self.num_classes + 5, grid_size, grid_size)
            .permute(0, 1, 3, 4, 2)
            .contiguous()
        )

        # Get outputs
        x = torch.sigmoid(prediction[..., 0])  # Center x
        y = torch.sigmoid(prediction[..., 1])  # Center y
        w = prediction[..., 2]  # Width
        h = prediction[..., 3]  # Height
        pred_conf = torch.sigmoid(prediction[..., 4])  # Conf
        pred_cls = torch.sigmoid(prediction[..., 5:])  # Cls pred.

        # If grid size does not match current we compute new offsets
        if grid_size != self.grid_size:
            self.compute_grid_offsets(grid_size, cuda=x.is_cuda)

        # Add offset and scale with anchors
        pred_boxes = FloatTensor(prediction[..., :4].shape)
        pred_boxes[..., 0] = x.data + self.grid_x
        pred_boxes[..., 1] = y.data + self.grid_y
        pred_boxes[..., 2] = torch.exp(w.data) * self.anchor_w
        pred_boxes[..., 3] = torch.exp(h.data) * self.anchor_h

        output = torch.cat(
            (
                pred_boxes.view(num_samples, -1, 4) * self.stride,
                pred_conf.view(num_samples, -1, 1),
                pred_cls.view(num_samples, -1, self.num_classes),
            ),
            -1,
        )

        if targets is None:
            return output, 0
            
        else:
            iou_scores, class_mask, obj_mask, noobj_mask, tx, ty, tw, th, tcls, tconf = build_targets(
                pred_boxes=pred_boxes,
                pred_cls=pred_cls,
                target=targets,
                anchors=self.scaled_anchors,
                ignore_thres=self.ignore_thres,
            )

            # Loss : Mask outputs to ignore non-existing objects (except with conf. loss)
            loss_x = self.mse_loss(x[obj_mask], tx[obj_mask])
            loss_y = self.mse_loss(y[obj_mask], ty[obj_mask])
            loss_w = self.mse_loss(w[obj_mask], tw[obj_mask])
            loss_h = self.mse_loss(h[obj_mask], th[obj_mask])
            loss_conf_obj = self.bce_loss(pred_conf[obj_mask], tconf[obj_mask])
            loss_conf_noobj = self.bce_loss(pred_conf[noobj_mask], tconf[noobj_mask])
            loss_conf = self.obj_scale * loss_conf_obj + self.noobj_scale * loss_conf_noobj
            loss_cls = self.bce_loss(pred_cls[obj_mask], tcls[obj_mask])
            total_loss = loss_x + loss_y + loss_w + loss_h + loss_conf + loss_cls

            # Metrics
            cls_acc = 100 * class_mask[obj_mask].mean()
            conf_obj = pred_conf[obj_mask].mean()
            conf_noobj = pred_conf[noobj_mask].mean()
            conf50 = (pred_conf > 0.5).float()
            iou50 = (iou_scores > 0.5).float()
            iou75 = (iou_scores > 0.75).float()
            detected_mask = conf50 * class_mask * tconf
            precision = torch.sum(iou50 * detected_mask) / (conf50.sum() + 1e-16)
            recall50 = torch.sum(iou50 * detected_mask) / (obj_mask.sum() + 1e-16)
            recall75 = torch.sum(iou75 * detected_mask) / (obj_mask.sum() + 1e-16)

            self.metrics = {
                "loss": to_cpu(total_loss).item(),
                "x": to_cpu(loss_x).item(),
                "y": to_cpu(loss_y).item(),
                "w": to_cpu(loss_w).item(),
                "h": to_cpu(loss_h).item(),
                "conf": to_cpu(loss_conf).item(),
                "cls": to_cpu(loss_cls).item(),
                "cls_acc": to_cpu(cls_acc).item(),
                "recall50": to_cpu(recall50).item(),
                "recall75": to_cpu(recall75).item(),
                "precision": to_cpu(precision).item(),
                "conf_obj": to_cpu(conf_obj).item(),
                "conf_noobj": to_cpu(conf_noobj).item(),
                "grid_size": grid_size,
            }

            return output, total_loss

def parse_model_config(path):
    """Parses the yolo layer configuration file and returns module definitions"""
    file = open(path, 'r')
    lines = file.read().split('\n')
    lines = [x for x in lines if x and not x.startswith('#')]
    lines = [x.rstrip().lstrip() for x in lines] # get rid of fringe whitespaces
    module_defs = []
    for line in lines:
        if line.startswith('['): # This marks the start of a new block
            module_defs.append({})
            module_defs[-1]['type'] = line[1:-1].rstrip()
            if module_defs[-1]['type'] == 'convolutional':
                module_defs[-1]['batch_normalize'] = 0
        else:
            key, value = line.split("=")
            value = value.strip()
            module_defs[-1][key.rstrip()] = value.strip()
    return module_defs

def create_modules(module_defs):
    """
    Constructs module list of layer blocks from module configuration in module_defs
    """
    hyperparams = module_defs.pop(0)
    output_filters = [int(hyperparams["channels"])]
    module_list = nn.ModuleList()
    image_size = int(hyperparams["height"])
    first_connected = True

    for module_i, module_def in enumerate(module_defs):
        modules = nn.Sequential()
        if module_def["type"] == "convolutional":
            bn = int(module_def["batch_normalize"])
            filters = int(module_def["filters"])
            kernel_size = int(module_def["size"])
            pad = (kernel_size - 1) // 2
            stride = int(module_def["stride"])
            modules.add_module(
                f"conv_{module_i}",
                nn.Conv2d(
                    in_channels=output_filters[-1],
                    out_channels=filters,
                    kernel_size=kernel_size,
                    stride=int(module_def["stride"]),
                    padding=pad,
                    bias=not bn,
                ),
            )
            image_size = int(((image_size+2*pad-kernel_size)/stride)+1)
            if bn:
                modules.add_module(f"batch_norm_{module_i}", nn.BatchNorm2d(filters, momentum=0.9, eps=1e-5))
            
            if module_def["activation"] == "leaky":
                modules.add_module(f"leaky_{module_i}", nn.LeakyReLU(0.1))

        elif module_def["type"] == "maxpool":
            kernel_size = int(module_def["size"])
            stride = int(module_def["stride"])
            if kernel_size == 2 and stride == 1:
                modules.add_module(f"_debug_padding_{module_i}", nn.ZeroPad2d((0, 1, 0, 1)))
            maxpool = nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=int((kernel_size - 1) // 2))
            modules.add_module(f"maxpool_{module_i}", maxpool)
            image_size /= 2 

        elif module_def["type"] == "upsample":
            upsample = Upsample(scale_factor=int(module_def["stride"]), mode="nearest")
            modules.add_module(f"upsample_{module_i}", upsample)

        elif module_def["type"] == "route":
            layers = [int(x) for x in module_def["layers"].split(",")]
            filters = sum([output_filters[1:][i] for i in layers])
            modules.add_module(f"route_{module_i}", EmptyLayer())

        elif module_def["type"] == "shortcut":
            filters = output_filters[1:][int(module_def["from"])]
            modules.add_module(f"shortcut_{module_i}", EmptyLayer())

        elif module_def["type"] == "yolo":
            anchor_idxs = [int(x) for x in module_def["mask"].split(",")]
            # Extract anchors
            anchors = [int(x) for x in module_def["anchors"].split(",")]
            anchors = [(anchors[i], anchors[i + 1]) for i in range(0, len(anchors), 2)]
            anchors = [anchors[i] for i in anchor_idxs]
            num_classes = int(module_def["classes"])
            img_size = int(hyperparams["height"])
            # Define detection layer
            yolo_layer = YOLOLayer(anchors, num_classes, img_size)
            modules.add_module(f"yolo_{module_i}", yolo_layer)
        
        elif module_def["type"] == "local":
            #bn = int(module_def["batch_normalize"])
            bn = 0
            module_def["batch_normalize"] = 0
            filters = int(module_def["filters"])
            kernel_size = int(module_def["size"])
            pad = (kernel_size - 1) // 2
            stride = int(module_def["stride"])
            modules.add_module(
                f"conv_{module_i}",
                nn.Conv2d(
                    in_channels=output_filters[-1],
                    out_channels=filters,
                    kernel_size=kernel_size,
                    stride=int(module_def["stride"]),
                    padding=pad,
                    bias=not bn,
                ),
            )
            image_size = int(((image_size+2*pad-kernel_size)/stride)+1)
            if bn:
                modules.add_module(f"batch_norm_{module_i}", nn.BatchNorm2d(filters, momentum=0.9, eps=1e-5))
            
            if module_def["activation"] == "leaky":
                modules.add_module(f"leaky_{module_i}", nn.LeakyReLU(0.1))

        elif module_def["type"] == "dropout":
            prob = float(module_def["probability"])
            modules.add_module(f"dropout{module_i}", nn.Dropout(prob))

        elif module_def["type"] == "connected":
            # If future version has multiple fc layers, it will be very useful.
            if first_connected: 
                size = image_size*image_size*output_filters[-1]
                output_features = int(module_def["output"])
                linear = nn.Linear(size, output_features)
                modules.add_module(f"linear_{module_i}", linear)
                first_connected = False
            #else: 
            #    modules.add_module(f"upsample_{module_i}", upsample)

        elif module_def["type"] == "detection":
            num_classes = int(module_def["classes"])
            hyperparams['num_classes'] = num_classes
            grid_size = int(module_def["side"])
            hyperparams['grid_size'] = grid_size
            num_coords = int(module_def["coords"])
            hyperparams['coords'] = num_coords
            softmax = bool(module_def["softmax"])
            num_box = int(module_def["num"])
            hyperparams['num_box'] = num_box
            sqrt = bool(module_def["sqrt"])
            # jitter will be applied in the class dataset, thus we do not load here.
            lambda_coob_scale = float(module_def["object_scale"])
            lambda_noob_scale = float(module_def["noobject_scale"])
            lambda_class_scale = float(module_def["class_scale"])
            lambda_coord_scale = float(module_def["coord_scale"])
            detection_layer = Detection(grid_size, num_box, num_classes, sqrt, lambda_coob_scale, 
                                        lambda_noob_scale, lambda_class_scale, lambda_coord_scale)
            modules.add_module(f"detection_{module_i}", detection_layer)
        # Register module list and number of output filters
        module_list.append(modules)
        output_filters.append(filters)

    return hyperparams, module_list

class Darknet(nn.Module):
    """
    YOLOv1 (BWTseng)/ YOLOv3 object detection model
    """
    def __init__(self, config_path, img_size=448):
        super(Darknet, self).__init__()
        self.module_defs = parse_model_config(config_path)
        self.hyperparams, self.module_list = create_modules(self.module_defs)
        
        self.feature_size = self.hyperparams['grid_size']
        self.num_bboxes = self.hyperparams['num_box']
        self.num_classes = self.hyperparams['num_classes']
        self.num_coords = self.hyperparams['coords']

        #self.yolo_layers = [layer[0] for layer in self.module_list if hasattr(layer[0], "metrics")]
        self.img_size = img_size
        self.seen = 0
        self.header_info = np.array([0, 0, 0, self.seen, 0], dtype=np.int32)

    def detection_reshape(self, logit):
        """
        Transform the output logit (1470-d for yolov1) into the correct permutation of the shape (b, 7, 7, 30)
        """
        # idx1 is set up for slicing the confidence score for the first bbox 
        idx1 = self.feature_size * self.feature_size * self.num_classes
        # idx2 is set up for slicing the confidence score for the second bbox
        idx2 = idx1 + self.feature_size * self.feature_size * self.num_bboxes
        # idx3 is set up for the index that bbox value starts.
        # idx 4 is set up for slicing four predicted value for the first bbox.
        # And we can apply the softmax function to the class prob, to further improve the performance.
        class_prob = logit[:, :idx1].view(-1, self.feature_size, self.feature_size, self.num_classes)
        # We can apply the sigmoid function to the bbox/confidnece prediction, which can avoid the improve the negative numbers.
        bbox = logit[:, idx2:].view(-1, self.feature_size, self.feature_size, self.num_coords*self.num_bboxes)
        confidences = logit[:, idx1:idx2].view(-1, self.feature_size, self.feature_size, self.num_bboxes)
        confidences_1 = confidences[:, :, :, 0].view(-1, self.feature_size, self.feature_size, 1)
        confidences_2 = confidences[:, :, :, 1].view(-1, self.feature_size, self.feature_size, 1)
        bbox_1 = bbox[:, :, :, :self.num_coords]
        bbox_2 = bbox[:, :, :, self.num_coords:]
        bbox_1 = torch.cat([bbox_1, confidences_1], -1)
        bbox_2 = torch.cat([bbox_2, confidences_2], -1)
        # We can transpose the axis 0 and 1 to accurately match the  row and column index. Note that I have already confirmed it using numpy.
        # How to implement it using Pytorch should survey.
        return torch.cat([bbox_1, bbox_2, class_prob], -1)

    def forward(self, x, targets=None):
        img_dim = x.shape[2]
        loss = 0
        layer_outputs, yolo_outputs = [], []
        for i, (module_def, module) in enumerate(zip(self.module_defs, self.module_list)):
            
            if module_def["type"] in ["convolutional", "upsample", "maxpool", "local"]:
                #print(module)
                x = module(x)
                
            elif module_def["type"] == "route":
                x = torch.cat([layer_outputs[int(layer_i)] for layer_i in module_def["layers"].split(",")], 1)
            
            elif module_def["type"] == "shortcut":
                layer_i = int(module_def["from"])
                x = layer_outputs[-1] + layer_outputs[layer_i]

            elif module_def["type"] == "yolo":
                x, layer_loss = module[0](x, targets, img_dim)
                loss += layer_loss
                yolo_outputs.append(x)
            
            elif module_def["type"] == "connected":
                # Get shape of the x attined from bottom layer (convolutions or fc) # If shape=4, else:
                shape = x.shape
                if len(shape) == 4: 
                    x = x.view(-1, shape[1]*shape[2]*shape[3])  
                x = module(x)

            elif module_def["type"] == "detection":
                # To use module[0] is that sequential module can not accept multiple arguments.
                num_classes = int(module_def["classes"])
                grid_size = int(module_def["side"])
                num_coords = int(module_def["coords"])
                num_bboxes = int(module_def["num"])
                #x = nn.Sigmoid(x)
                # *********
                # We should add reshape layer here::
                # *********
                #x = x.view(-1, grid_size, grid_size, (num_classes+(num_coords+1)*num_bboxes))
                x = self.detection_reshape(x)
                x, layer_loss = module[0](x, targets)
                loss += layer_loss
            
            layer_outputs.append(x)
        
        # TODO should figure out why yolo v3 do the concatenate in the following code:
        #yolo_outputs = to_cpu(torch.cat(yolo_outputs, 1))
        #return yolo_outputs if targets is None else (loss, yolo_outputs)
        return x if targets is None else (loss, x)

    def load_darknet_weights(self, weights_path):
        """Parses and loads the weights stored in 'weights_path'"""

        # Open the weights file
        with open(weights_path, "rb") as f:
            # ***
            # For yolo v3, the header count = 5, while it will be 4 in v1 and v2 version.
            # ***
            header = np.fromfile(f, dtype=np.int32, count=4)  # First five are header values
            self.header_info = header  # Needed to write header when saving weights
            self.seen = header[3]  # number of images seen during training
            weights = np.fromfile(f, dtype=np.float32)  # The rest are weights

        # Establish cutoff for loading backbone weights
        cutoff = None
        if "darknet53.conv.74" in weights_path:
            cutoff = 75

        print(weights.shape)
        ptr = 0
        for i, (module_def, module) in enumerate(zip(self.module_defs, self.module_list)):
            if i == cutoff:
                break
            if module_def["type"] in ["convolutional"]:
                conv_layer = module[0]
                if module_def["batch_normalize"]:
                    # Load BN bias, weights, running mean and running variance
                    bn_layer = module[1]
                    num_b = bn_layer.bias.numel()  # Number of biases
                    # Bias
                    bn_b = torch.from_numpy(weights[ptr : ptr + num_b]).view_as(bn_layer.bias)
                    bn_layer.bias.data.copy_(bn_b)
                    ptr += num_b
                    # Weight
                    bn_w = torch.from_numpy(weights[ptr : ptr + num_b]).view_as(bn_layer.weight)
                    bn_layer.weight.data.copy_(bn_w)
                    ptr += num_b
                    # Running Mean
                    bn_rm = torch.from_numpy(weights[ptr : ptr + num_b]).view_as(bn_layer.running_mean)
                    bn_layer.running_mean.data.copy_(bn_rm)
                    ptr += num_b
                    # Running Var
                    bn_rv = torch.from_numpy(weights[ptr : ptr + num_b]).view_as(bn_layer.running_var)
                    bn_layer.running_var.data.copy_(bn_rv)
                    ptr += num_b
                else:
                    # Load conv. bias
                    num_b = conv_layer.bias.numel()
                    conv_b = torch.from_numpy(weights[ptr : ptr + num_b]).view_as(conv_layer.bias)
                    conv_layer.bias.data.copy_(conv_b)
                    ptr += num_b
            
                # Load conv. weights
                num_w = conv_layer.weight.numel()
                conv_w = torch.from_numpy(weights[ptr : ptr + num_w]).view_as(conv_layer.weight)
                conv_layer.weight.data.copy_(conv_w)
                ptr += num_w
                print(ptr)

            elif module_def["type"] == "connected":
                linear_layer = module[0]
                #print(linear_layer)
                num_b = linear_layer.bias.numel()
                linear_b = torch.from_numpy(weights[ptr : ptr + num_b]).view_as(linear_layer.bias)
                linear_layer.bias.data.copy_(linear_b)
                ptr += num_b

                num_w = linear_layer.weight.numel()
                linear_w = torch.from_numpy(weights[ptr : ptr + num_w]).view_as(linear_layer.weight)
                linear_layer.weight.data.copy_(linear_w)
                ptr += num_w    
                #print(num_w)
        print(ptr)

    def save_darknet_weights(self, path, cutoff=-1):
        """
            @:param path    - path of the new weights file
            @:param cutoff  - save layers between 0 and cutoff (cutoff = -1 -> all are saved)
        """
        fp = open(path, "wb")
        self.header_info[3] = self.seen
        self.header_info.tofile(fp)

        # Iterate through layers
        for i, (module_def, module) in enumerate(zip(self.module_defs[:cutoff], self.module_list[:cutoff])):
            if module_def["type"] == "convolutional":
                conv_layer = module[0]
                # If batch norm, load bn first
                if module_def["batch_normalize"]:
                    bn_layer = module[1]
                    bn_layer.bias.data.cpu().numpy().tofile(fp)

                    bn_layer.weight.data.cpu().numpy().tofile(fp)
                    bn_layer.running_mean.data.cpu().numpy().tofile(fp)
                    bn_layer.running_var.data.cpu().numpy().tofile(fp)
                # Load conv bias
                else:
                    conv_layer.bias.data.cpu().numpy().tofile(fp)
                # Load conv weights
                conv_layer.weight.data.cpu().numpy().tofile(fp)

        fp.close()


def _conv_output_length(input_length, filter_size, stride):
    output_length = input_length - filter_size + 1
    return output_length



"""
Tt's very important, because it will be applied in YOLOv1 in the future:
*
From Torchex package, it aims to deal with the local layer used in YOLOv1.

class Conv2dLocal(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, in_size=None,
                 padding=0, bias=True):
        super(Conv2dLocal, self).__init__()        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.in_size = in_size
        self.padding = padding
        self.bias_flag = bias
        self.bias = None
        self.weight = None

        kh, kw = _pair(self.kernel_size)
        ih, iw = _pair(in_size)

        if (self.in_channels is not None) and (self.in_size is not None):
            self._initialize_params(in_channels, in_size)
            self.reset_parameters()

    def _initialize_params(self, in_channels, in_size):
        kh, kw = _pair(self.kernel_size)
        ih, iw = _pair(in_size)
        oh = _conv_output_length(ih, kh, self.stride[0])
        ow = _conv_output_length(iw, kw, self.stride[1])
        W_shape = (self.out_channels, oh, ow, in_channels, kh, kw)
        bias_shape = (self.out_channels, oh, ow,)
        self.weight = Parameter(torch.Tensor(*W_shape))
        if self.bias_flag:
            self.bias = Parameter(torch.Tensor(*bias_shape))            
        else:
            self.register_parameter('bias', None)
        
    def reset_parameters(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1. / math.sqrt(n)
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)        

    def forward(self, input):
        if self.weight is None:
            self._initialize_params(input.shape[1], input.shape[2:])
            self.reset_parameters()
        return conv2d_local(input, self.weight, self.bias, self.stride)
"""
