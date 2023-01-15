import os
import tqdm
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader

from common.meter import Meter
from common.utils import detect_grad_nan, compute_accuracy, set_seed, setup_run,load_model
from models.dataloader.samplers import CategoriesSampler,CategoriesSampler_1,CategoriesSampler_2
from models.dataloader.data_utils import dataset_builder
from models.protonet import ProtoNet
from test import test_main, evaluate
import random

def train(epoch, model, loader, optimizer, args=None):
    model.train()

    train_loader = loader['train_loader']
    

    # label for query set, always in the same pattern
    label = torch.arange(args.way).repeat(args.query).cuda()  # 012340123401234...
  
    #label = label.unsqueeze(0).repeat(args.num_task,1).reshape(-1)
    loss_meter = Meter()
    acc_meter = Meter()

   
    tqdm_gen = tqdm.tqdm(train_loader)

    for i, ((data, train_labels)) in enumerate(tqdm_gen, 1):
        
        data, train_labels = data.cuda(), train_labels.cuda()

        # Forward images (3, 84, 84) -> (C, H, W)
        model.module.mode = 'encoder'
        data = model(data)

     
        # loss for batch
        model.module.mode = 'distance'
        
        k = args.shot
        data = data.reshape( args.way*(args.shot+args.query),args.num_task, -1).transpose(0,1)
        data = data.reshape( args.num_task, (args.shot+args.query),args.way, -1).transpose(1,2)
        data_shot, data_query = data[:,:,:k], data[:,:,k:]
        logits= model((data_shot, data_query)).view(args.num_task, args.way*args.query,-1)

        loss = 0 
        acc = 0
        for n_task in range(args.num_task):
            epi_loss = F.cross_entropy(logits[n_task], label)
            loss = epi_loss + loss
            acc_i = compute_accuracy(logits[n_task], label)
            acc= acc+acc_i
        loss = loss/args.num_task
        acc = acc/args.num_task

        loss_meter.update(loss.item())
        acc_meter.update(acc)
        tqdm_gen.set_description(f'[train] epo:{epoch:>3} | avg.loss:{loss_meter.avg():.4f} | avg.acc:{acc_meter.avg():.3f} (curr:{acc:.3f})')

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        detect_grad_nan(model)
        optimizer.step()
        optimizer.zero_grad()

    return loss_meter.avg(), acc_meter.avg(), acc_meter.confidence_interval()


def train_main(args):
    Dataset = dataset_builder(args)
    
   

    valset = Dataset('val', args)
    val_sampler = CategoriesSampler(valset.label, args.val_episode, args.num_task_test, args.way, args.shot + args.query)
    val_loader = DataLoader(dataset=valset, batch_sampler=val_sampler, num_workers=8, pin_memory=True)
    ''' fix val set for all epochs '''
    val_loader = [x for x in val_loader]

    set_seed(args.seed)
    model = ProtoNet(args).cuda()

    model = load_model(model, '.../pre-train-model/protonet/cub/1shot-5way/your_run/max_acc.pth')
    model = nn.DataParallel(model, device_ids=args.device_ids)
#    print(model)


    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.milestones, gamma=args.gamma)

    max_acc, max_epoch = 0.0, 0
    set_seed(args.seed)

    for epoch in range(1, args.max_epoch + 1):
        start_time = time.time()

        if epoch<0:
            trainset = Dataset('train', args)
            train_sampler = CategoriesSampler(trainset.label, 300, args.num_task, args.way, args.shot + args.query)
            train_loader = DataLoader(dataset=trainset, batch_sampler=train_sampler, num_workers=8, pin_memory=True)
        else:
            from center import center

            if epoch == 1:
                between, within = center(model, args)
            if args.train_sampling == "cl_sampling":
                trainset = Dataset('train', args)
                train_sampler = CategoriesSampler_1(trainset.label, len(trainset.data) // args.batch, args.num_task, args.way, args.shot + args.query, between, within,model,args)
                train_loader = DataLoader(dataset=trainset, batch_sampler=train_sampler, num_workers=4, pin_memory=True)
            elif args.train_sampling == "cpl_sampling":
                
                trainset = Dataset('train', args)
                num_class = trainset.num_class
    
                in_all = torch.mm(within.unsqueeze(-1),within.unsqueeze(0))
                in_all_between = between/in_all + torch.eye(num_class).cuda()* 5e+10

                _, indices = torch.sort(in_all_between)

                list_set = []
                for i in range(num_class):
                    set2 = set(indices[i,:int(num_class*0.5)].tolist())
                    hard_list=list(set2)
                    list_set.append(hard_list)

                train_sampler = CategoriesSampler_2(trainset.label, len(trainset.data) // args.batch, args.num_task, args.way, args.shot + args.query, list_set,model,args)
                train_loader = DataLoader(dataset=trainset, batch_sampler=train_sampler, num_workers=4, pin_memory=True)
            
            
            else:
                raise ValueError('Unknown train_sampling')

        

        train_loaders = {'train_loader': train_loader}
        train_loss, train_acc, _ = train(epoch, model, train_loaders, optimizer, args)
        val_loss, val_acc, _ = evaluate(epoch, model, val_loader, args, set='val')

        if val_acc > max_acc:
            print(f'[ log ] *********A better model is found ({val_acc:.3f}) *********')
            max_acc, max_epoch = val_acc, epoch
            torch.save(dict(params=model.state_dict(), epoch=epoch), os.path.join(args.save_path, 'max_acc.pth'))
            torch.save(optimizer.state_dict(), os.path.join(args.save_path, 'optimizer_max_acc.pth'))

        if args.save_all:
            torch.save(dict(params=model.state_dict(), epoch=epoch), os.path.join(args.save_path, f'epoch_{epoch}.pth'))
            torch.save(optimizer.state_dict(), os.path.join(args.save_path, f'optimizer_epoch_{epoch}.pth'))

        epoch_time = time.time() - start_time
        print(f'[ log ] saving @ {args.save_path}')
        print(f'[ log ] roughly {(args.max_epoch - epoch) / 3600. * epoch_time:.2f} h left\n')

        lr_scheduler.step()

    return model


if __name__ == '__main__':
    args = setup_run(arg_mode='train')

    model = train_main(args)
    test_acc, test_ci = test_main(model, args)

