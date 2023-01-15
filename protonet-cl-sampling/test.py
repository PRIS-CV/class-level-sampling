import os
import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader

from common.meter import Meter
from common.utils import compute_accuracy, load_model, setup_run, by
from models.dataloader.samplers import CategoriesSampler
from models.dataloader.data_utils import dataset_builder
from models.protonet import ProtoNet
import math

def evaluate(epoch, model, loader, args=None, set='val'):
    model.eval()

    loss_meter = Meter()
    acc_meter = Meter()

    label = torch.arange(args.way).repeat(args.query).cuda()
 #   label = label.unsqueeze(0).repeat(args.num_task_test,1).reshape(-1)

    
    tqdm_gen = tqdm.tqdm(loader)


    with torch.no_grad():
        for i, (data, train_labels) in enumerate(tqdm_gen, 1):
            data = data.cuda()
            model.module.mode = 'encoder'
            data = model(data)
            
           # model.module.mode = 'bi'
           # data = model(data)
            
            
            model.module.mode = 'distance'
            data = data.reshape( args.way*(args.shot+args.query),args.num_task_test, -1).transpose(0,1)

            k = args.shot
            data = data.reshape( args.num_task_test, (args.shot+args.query),args.way, -1).transpose(1,2)
            data_shot, data_query = data[:,:,:k], data[:,:,k:]
            logits= model((data_shot, data_query)).view(args.num_task_test, args.way*args.query,-1)
            

            loss = 0 
            acc = 0
            for n_task in range(args.num_task_test):
                epi_loss = F.cross_entropy(logits[n_task], label)
                loss = epi_loss + loss
                acc_i = compute_accuracy(logits[n_task], label)
                acc= acc+acc_i
            loss = loss/args.num_task_test
            acc = acc/args.num_task_test

            loss_meter.update(loss.item())
            acc_meter.update(acc)
            tqdm_gen.set_description(f'[{set:^5}] epo:{epoch:>3} | avg.loss:{loss_meter.avg():.4f} | avg.acc:{by(acc_meter.avg())} (curr:{acc:.3f})')

    return loss_meter.avg(), acc_meter.avg(), acc_meter.confidence_interval()


def test_main(model, args):

    ''' load model '''
    model = load_model(model, os.path.join(args.save_path, 'max_acc.pth'))

    ''' define test dataset '''
    Dataset = dataset_builder(args)
    test_set = Dataset('test', args)
    sampler = CategoriesSampler(test_set.label, args.test_episode, args.num_task_test, args.way, args.shot + args.query)
    test_loader = DataLoader(test_set, batch_sampler=sampler, num_workers=8, pin_memory=True)

    ''' evaluate the model with the dataset '''
    _, test_acc, test_ci = evaluate("best", model, test_loader, args, set='test')
    print(f'[final] epo:{"best":>3} | {by(test_acc)} +- {test_ci:.3f}')

    return test_acc, test_ci


if __name__ == '__main__':
    args = setup_run(arg_mode='test')

    ''' define model '''
    model = ProtoNet(args).cuda()
    model = nn.DataParallel(model, device_ids=args.device_ids)

    test_main(model, args)
