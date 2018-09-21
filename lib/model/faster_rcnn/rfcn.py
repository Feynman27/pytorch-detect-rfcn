import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.models as models
from torch.autograd import Variable
import numpy as np
from model.utils.config import cfg
from model.rpn.rpn import _RPN
from model.psroi_pooling.modules.psroi_pool import _PSRoIPooling
from model.rpn.proposal_target_layer_cascade import _ProposalTargetLayer
import time
import pdb
from model.utils.net_utils import _smooth_l1_loss, _crop_pool_layer, _affine_grid_gen, _affine_theta

class _RFCN(nn.Module):
    """ RFCN """
    def __init__(self, classes, class_agnostic):
        super(_RFCN, self).__init__()
        self.classes = classes
        self.n_classes = len(classes)
        self.n_reg_classes = (1 if class_agnostic else len(classes))
        self.class_agnostic = class_agnostic
        self.n_bbox_reg = (4 if class_agnostic else len(classes))
        # loss
        self.RFCN_loss_cls = 0
        self.RFCN_loss_bbox = 0

        # define rpn
        self.RFCN_rpn = _RPN(self.dout_base_model)
        self.RFCN_proposal_target = _ProposalTargetLayer(self.n_classes)
        self.RFCN_psroi_cls_pool = _PSRoIPooling(cfg.POOLING_SIZE, cfg.POOLING_SIZE, 
                                spatial_scale=1.0/16.0, group_size=7, output_dim=self.n_classes)
        self.RFCN_psroi_loc_pool = _PSRoIPooling(cfg.POOLING_SIZE, cfg.POOLING_SIZE, 
                                spatial_scale=1.0/16.0, group_size=7, output_dim=4*self.n_reg_classes)

        self.grid_size = cfg.POOLING_SIZE * 2 if cfg.CROP_RESIZE_WITH_MAX_POOL else cfg.POOLING_SIZE

	self.RFCN_cls_net = nn.Conv2d(512,self.n_classes*7*7, [1,1], padding=0, stride=1)
        nn.init.normal(self.RFCN_cls_net.weight.data, 0.0, 0.01)
        
	self.RFCN_bbox_net = nn.Conv2d(512, 4*self.n_reg_classes*7*7, [1,1], padding=0, stride=1)
	nn.init.normal(self.RFCN_bbox_net.weight.data, 0.0, 0.01)

        self.RFCN_cls_score = nn.AvgPool2d((7,7), stride=(7,7))
        self.RFCN_bbox_pred = nn.AvgPool2d((7,7), stride=(7,7))

    def forward(self, im_data, im_info, gt_boxes, num_boxes):

        batch_size = im_data.size(0)

        im_info = im_info.data
        gt_boxes = gt_boxes.data
        num_boxes = num_boxes.data
        # feed image data to base model to obtain base feature map
        # output of feature map will be (batch_size, C, H, W)
        base_feat = self._im_to_head(im_data)
        rfcn_cls = self.RFCN_cls_net(base_feat)
        rfcn_bbox = self.RFCN_bbox_net(base_feat)
        # feed base feature map tp RPN to obtain rois
        rpn_out = self.RFCN_rpn(base_feat, 
                                    im_info, 
                                    gt_boxes[:,:,:5].clone(), 
                                    num_boxes)
        rois = rpn_out[0] 
        rpn_loss_cls = rpn_out[1]
        rpn_loss_bbox = rpn_out[2]
        if self.training:
            roi_data = self.RFCN_proposal_target(rois, gt_boxes[:,:,:5].clone(), num_boxes)
            rois, rois_label, \
                        rois_target, rois_inside_ws, rois_outside_ws = roi_data

            rois_label = Variable(rois_label.view(-1).long())
            rois_target = Variable(rois_target.view(-1, rois_target.size(2)))
            rois_inside_ws = Variable(rois_inside_ws.view(-1, rois_inside_ws.size(2)))
            rois_outside_ws = Variable(rois_outside_ws.view(-1, rois_outside_ws.size(2)))
        else:
	    rois_label = None
	    rois_target = None
	    rois_inside_ws = None
	    rois_outside_ws = None
	    loss_cls = Variable(torch.zeros(1).cuda(), volatile=True)
	    rpn_loss_bbox = Variable(torch.zeros(1).cuda(), volatile=True)

	rois = Variable(rois)
	pooled_cls_feat = self.RFCN_psroi_cls_pool(rfcn_cls, rois.view(-1,5))
        pooled_loc_feat = self.RFCN_psroi_loc_pool(rfcn_bbox, rois.view(-1,5))
        # compute object classification probability
        cls_score = self.RFCN_cls_score(pooled_cls_feat).squeeze()
        cls_prob = F.softmax(cls_score, dim=1)

        # compute bbox offset
        bbox_pred = self.RFCN_bbox_pred(pooled_loc_feat).squeeze()

        if self.training and not self.class_agnostic:
            # select the corresponding columns according to roi labels
            bbox_pred_view = bbox_pred.view(bbox_pred.size(0), int(bbox_pred.size(1) / 4), 4)
            bbox_pred_select = torch.gather(bbox_pred_view, 1, 
                        rois_label.view(rois_label.size(0), 1, 1).expand(rois_label.size(0), 1, 4))
            bbox_pred = bbox_pred_select.squeeze(1)

        RFCN_loss_cls = torch.zeros(1).cuda()
        RFCN_loss_bbox = torch.zeros(1).cuda()

        if self.training:
            # classification loss
            RFCN_loss_cls = F.cross_entropy(cls_score, rois_label)
            # bounding box regression L1 loss
            RFCN_loss_bbox = _smooth_l1_loss(bbox_pred, 
                rois_target, rois_inside_ws, rois_outside_ws)


        cls_prob = cls_prob.view(batch_size, rois.size(1), -1)
        bbox_pred = bbox_pred.view(batch_size, rois.size(1), -1)

        return rois, cls_prob, bbox_pred, rpn_loss_cls, rpn_loss_bbox, \
                RFCN_loss_cls, RFCN_loss_bbox, rois_label

    def _init_weights(self):
        def normal_init(m, mean, stddev, truncated=False):
            """
            weight initalizer: truncated normal and random normal.
            """
            # x is a parameter
            if truncated:
                m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean) # not a perfect approximation
            else:
                m.weight.data.normal_(mean, stddev)
                m.bias.data.zero_()

        normal_init(self.RFCN_rpn.RPN_Conv, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RFCN_rpn.RPN_cls_score, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RFCN_rpn.RPN_bbox_pred, 0, 0.01, cfg.TRAIN.TRUNCATED)
        #normal_init(self.RFCN_cls_score, 0, 0.01, cfg.TRAIN.TRUNCATED)
        #normal_init(self.RFCN_bbox_pred, 0, 0.001, cfg.TRAIN.TRUNCATED)

    def create_architecture(self):
        self._init_modules()
        self._init_weights()
