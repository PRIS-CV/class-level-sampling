    


import torch

import torch.nn.functional as F

from torch.utils.data import DataLoader

import random

def center(model_1,args):    
    with torch.no_grad():   
        model_1.module.mode = 'encoder'
        '''
        from res12 import Res12
        model_1 = Res12()

    
        model_dict_1 = model_1.state_dict()   
    
        pretrained_dict_1 = torch.load('/home/guoyurong/difficute_experient/miniimagenet_uniform/Res12-pre.pth')['params']
        pretrained_dict_1 = {k[8:]: v for k, v in pretrained_dict_1.items()}
    
    
        pretrained_dict_1 = {k: v for k, v in pretrained_dict_1.items() if k in model_dict_1}
        print(pretrained_dict_1.keys())
        model_dict_1.update(pretrained_dict_1)

    
        model_1.load_state_dict(model_dict_1)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model_1 = model_1.to(device)
        
        import torchvision.models as models

        model_1 = models.resnet50(pretrained=True)
        del model_1.fc
        model_1.fc=lambda x:x
        model_1 = model_1.cuda()
        '''
        model_1.eval()
        
     #   from models.dataloader.mini_imagenet import MiniImageNet as Dataset
      #  from models.dataloader.tiered_imagenet import tieredImageNet as Dataset
#        from models.dataloader.cifar_fs import DatasetLoader as Dataset
        from models.dataloader.cub import CUB as Dataset
        testset_1 = Dataset('train',args)
        test_loader_1 = DataLoader(dataset=testset_1, batch_size=64, shuffle=True, num_workers=8, pin_memory=True)
        num_class = testset_1.num_class
        #test_loader_1 = get_loader_1(cfg.dataset.novel_file, 64, train=True)
        

        

        
        all_embedding = torch.zeros(num_class, 640).cuda()
        all_sample = torch.zeros(num_class).cuda()
        
        for i, batch in enumerate(  test_loader_1, 1):

            
            if torch.cuda.is_available():
                data, label_1 = [_.cuda() for _ in batch]
                label_1 = label_1.type(torch.cuda.LongTensor)
            else:
                data, label_1 = batch
                label_1 = label_1.type(torch.LongTensor)
        
            logits = model_1(data)
           # logits = F.adaptive_avg_pool2d(logits, 1)
            logits = logits.squeeze(-1).squeeze(-1)

        
            for j in range(len(label_1)):
                
                all_embedding[label_1[j],:] = all_embedding[label_1[j],:]+logits[j]
                all_sample[label_1[j]] = all_sample[label_1[j]]+1
        
        
            
        embedding_class_center = all_embedding/(all_sample.unsqueeze(-1))
            
        embedding_class_center_1 = F.normalize(embedding_class_center, dim=-1) # normalize for cosine distance
                
        between = torch.sum((embedding_class_center_1.unsqueeze(1) - embedding_class_center_1.unsqueeze(0)) ** 2, 2)
       
        
        
        sample_to_center = torch.zeros(num_class).cuda()
        
        all_sample = torch.zeros(num_class).cuda()
        
        for i, batch in enumerate(test_loader_1, 1):

            
            if torch.cuda.is_available():
                data, label_1 = [_.cuda() for _ in batch]
                label_1 = label_1.type(torch.cuda.LongTensor)
            else:
                data, label_1 = batch
                label_1 = label_1.type(torch.LongTensor)
            logits = model_1(data)
         #   logits = F.adaptive_avg_pool2d(logits, 1)
            logits = logits.squeeze(-1).squeeze(-1)
            logits = F.normalize(logits, dim=-1)
            for j in range(len(label_1)):
                    
                     
                sample_to_center[label_1[j]] = sample_to_center[label_1[j]] + torch.sum((logits[j] - embedding_class_center_1[label_1[j],:]) ** 2, -1)
                all_sample[label_1[j]] = all_sample[label_1[j]]+1
    
        within = sample_to_center/all_sample

        
        
        '''
        testset_1 = Dataset('train',args)
        test_loader_1 = DataLoader(dataset=testset_1, batch_size=64, shuffle=False, num_workers=8, pin_memory=True)
        num_class = testset_1.num_class

        sample = torch.zeros(num_class, 600, 640).cuda()
        
        k  = 0.0
        for i, batch in enumerate(test_loader_1, 1):

            
            if torch.cuda.is_available():
                data, label_1 = [_.cuda() for _ in batch]
                label_1 = label_1.type(torch.cuda.LongTensor)
            else:
                data, label_1 = batch
                label_1 = label_1.type(torch.LongTensor)
            
            logits = model_1(data)
            logits = logits.squeeze(-1).squeeze(-1)
        #    logits = F.normalize(logits, dim=-1)
            
            for j in range(len(label_1)):
                sample[int(k/600.0), int(k%600.0),:] = sample[int(k/600.0), int(k%600.0),:] + logits[j]
                k = k +1
        
        sample = F.normalize(sample, dim=-1) # normalize for cosine distance
                
        '''

    return between, within
