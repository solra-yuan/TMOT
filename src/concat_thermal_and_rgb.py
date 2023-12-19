import sacred
import yaml

ex = sacred.Experiment('train', interactive=True)
def load_config(file_name):
    with open(file_name, 'r') as f:
        cfg = yaml.safe_load(f)
    return cfg

@ex.config
def my_config():
    cfgfile_list = ['./cfgs/train.yaml',
                    './cfgs/train_deformable.yaml',
                    './cfgs/train_tracking.yaml',
                    './cfgs/train_multi_frame.yaml',
                    './cfgs/train_flir_adas_v2_concat.yaml']
    config = dict()
    config_list = []
    for item in cfgfile_list:
        config_list.append(load_config(item))
    
    for cfg_item in config_list:
        for k in cfg_item:
            config[k] = cfg_item[k]



import os

import numpy as np
import torch
from torch.utils.data import DataLoader, DistributedSampler
from trackformer.datasets import build_dataset
from trackformer.util.misc import nested_dict_to_namespace

import trackformer.util.misc as utils
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

@ex.main
def my_main(config):
    print(config)
    args = nested_dict_to_namespace(config)
    args.flir_adas_v2_path_train = '../data/flir_adas_v2'
    args.flir_adas_v2_path_val = '../data/flir_adas_v2'
    utils.init_distributed_mode(args)

    # load imgs & annotations for debugging purpose
    
    
    dataset_train = build_dataset(split='train', args=args)
    dataset_val = build_dataset(split='val', args=args)

    if args.distributed:
        sampler_train = utils.DistributedWeightedSampler(dataset_train)
        # sampler_train = DistributedSampler(dataset_train)
        sampler_val = DistributedSampler(dataset_val, shuffle=False)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    batch_sampler_train = torch.utils.data.BatchSampler(
        sampler_train, args.batch_size, drop_last=True)


    data_loader_train = DataLoader(
        dataset_train,
        batch_sampler=batch_sampler_train,
        collate_fn=utils.collate_fn,
        num_workers=args.num_workers)
    data_loader_val = DataLoader(
        dataset_val, args.batch_size,
        sampler=sampler_val,
        drop_last=False,
        collate_fn=utils.collate_fn,
        num_workers=args.num_workers)
    
    i = 0
    for img, target, target_t, in data_loader_train:
        print("type(img)", type(img))
        #print("img.tensors.shape", img.tensors.shape)
        
        # print('img_rgb type', type(img_rgb))
        # print('img_t type', type(img_t))
        # print('img.tensors.shape', img_rgb.tensors.shape)
        print("img_t.tensors.shapw", img.tensors.shape)
        fig, ax = plt.subplots(2,6)
        fig.set_figheight(9)
        fig.set_figwidth(20)


        for k in range(2):
            if target[k]['image_id'] > 0 : 
                img_previdx, target_previdx, target_t_previdx = dataset_train[int(target[k]['image_id']-1)]
                print("img_previdx type", type(img_previdx))
                print("img_previdx size", img_previdx.shape)
                test_prev_img_rgb = ax[k][2].imshow(img_previdx[0:3, :, :].permute(1,2,0))
                test_prev_img_t = ax[k][3].imshow(img_previdx[3:4, :, :].permute(1,2,0))
            ax[k,0].imshow(img.tensors[k,0:3,:,:].permute(1,2,0))
            t_channel_one = img.tensors[k, 3:4, :, :].permute(1,2,0)
            t_channel_two = img.tensors[k, 4:5, :, :].permute(1,2,0)
            t_channel_three = img.tensors[k, 5:6, :, :].permute(1,2,0)
            print("slicing and indexing ch1", t_channel_one.shape, img.tensors[k, 3, :, :].shape)
            print("slicing and indexing ch3", t_channel_three.shape, img.tensors[k, 5, :, :].shape)
            #print("are channels same?", t_channel_one==t_channel_two, t_channel_two==t_channel_three)

            im_t = ax[k,1].imshow(img.tensors[k,3:4,:,:].permute(1,2,0))
        for l in range(2):
            for r in target[l]['boxes']:
                print("(float(r[0]), float(r[1])), float(r[2]), float(r[3])", float(r[0])*640, float(r[1])*512, float(r[2])*640, float(r[3])*512)
                if l==0:
                    c = 'r'
                elif l==1:
                    c = 'y'
                rect = Rectangle((float(r[0]*640), float(r[1]*512)), float(r[2]*640), float(r[3]*512), linewidth=1, edgecolor=c, facecolor='none')
                ax[l][0].add_patch(rect)
        for j in range(2):
            for r in target_t[j]['boxes']:
                print("(float(r[0]), float(r[1])), float(r[2]), float(r[3])", float(r[0])*640, float(r[1])*512, float(r[2])*640, float(r[3])*512)
                if j==0:
                    c = 'b'
                elif j==1:
                    c = 'g'
                rect = Rectangle((float(r[0]*640), float(r[1]*512)), float(r[2]*640), float(r[3]*512), linewidth=1, edgecolor=c, facecolor='none')
                ax[j][1].add_patch(rect)
        for s in range(2):
            print('type of target_t[0]["prev_image"]', type(target_t[s]['prev_image']))
            print('type of target_t[0]["prev_target"]', type(target_t[s]['prev_target']))
            print('shape of target_t[0]["prev_image"]', target_t[s]['prev_image'].shape)
            ax[s][4].imshow(target[s]['prev_image'].permute(1,2,0))
            ax[s][5].imshow(target_t[s]['prev_image'].permute(1,2,0))
            
        print("type(target_t)", type(target_t))
        print("type(target_t[0])", type(target_t[0]))
        print("type(target_t[1])", type(target_t[1]))
        print("target_t[0].keys()", target_t[0].keys())
        print("target_t[0] label, image_id", target_t[0]['labels'], target_t[0]['image_id'])
        print("target_t[1].keys()", target_t[1].keys())
        print("target_t[1] label, image_id", target_t[1]['labels'], target_t[1]['image_id'])

        plt.show()

        #print("item[1][0].keys()", item[1][0].keys())
        #print("item[1][1].keys()", item[1][1].keys())
        input()
        i += 1

if __name__ == '__main__':
    ex.run()
