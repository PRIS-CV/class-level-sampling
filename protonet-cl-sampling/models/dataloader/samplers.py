import torch
import numpy as np
from torch.nn.modules.container import ModuleList
from torch.utils.data import DataLoader
import torch.nn.functional as F
import random
from itertools import combinations
from itertools import permutations


class CategoriesSampler():


    def __init__(self, label, n_batch,num_task, n_cls, n_per):
        self.n_batch = n_batch  # the number of iterations in the dataloader
        self.n_cls = n_cls
        self.n_per = n_per
        self.num_task = num_task

        label = np.array(label)  # all data label
        self.m_ind = []  # the data index of each class
        for i in range(max(label) + 1):
            ind = np.argwhere(label == i).reshape(-1)  # all data index of this class
            ind = torch.from_numpy(ind)
            self.m_ind.append(ind)

    def __len__(self):
        return self.n_batch
    
    def __iter__(self):
        batch = []
        for i_batch in range(self.n_batch):
            
            classes = torch.randperm(len(self.m_ind))[:self.n_cls]  # random sample num_class indices, e.g. 5
            for c in classes:
                l = self.m_ind[c]  # all data indices of this class
                pos = torch.randperm(len(l))[:self.n_per]  # sample n_per data index of this class
                batch.append(l[pos])

            if len(batch) == self.num_task * self.n_cls:
                # batch = paddle.stack(batch).reshape((-1))
                batch = torch.stack(batch).t().reshape(-1)
            
                yield batch
                batch = []
            # .t() transpose,
            # due to it, the label is in the sequence of abcdabcdabcd form after reshape,
            # instead of aaaabbbbccccdddd


class CategoriesSampler_1():

    def __init__(self, label, n_batch,num_task, n_cls, n_per, between, within ,model,args):
        self.n_batch = n_batch  # the number of iterations in the dataloader
        self.n_cls = n_cls
        self.n_per = n_per

        label = np.array(label)  # all data label
        self.m_ind = []  # the data index of each class
        for i in range(max(label) + 1):
            ind = np.argwhere(label == i).reshape(-1)  # all data index of this class
            ind = torch.from_numpy(ind)
            self.m_ind.append(ind)
        self.between = between
        self.within =within
        self.num_task = num_task
        
        self.model = model
        self.args  = args
        #self.sample =sample
    def __len__(self):
        return self.n_batch
    
    def __iter__(self):
       
        batch = []
        
        picked_class = torch.randperm(len(self.m_ind))
        li = picked_class.tolist()
        newIter = combinations(li[0:int(len(self.m_ind)*0.4)], self.n_cls-1)
        newList = list(newIter) 


        for i_batch in range(self.n_batch):
            
            
            anchor_class_set = li[int(len(self.m_ind)*0.4):]
            random.shuffle(anchor_class_set)
            anchor_class = anchor_class_set[0]

            D_intar_inter = []
            
            for i in range(len(newList)):
                d_inter = 0
                d_intra = 0
                d_intra = self.within[torch.tensor(newList[i]+(anchor_class,))].mean()
                d_inter = self.between[torch.tensor(newList[i]+(anchor_class,))][:,torch.tensor(newList[i]+(anchor_class,))].sum()/((len(newList[i])+1)*len(newList[i]))
                #from IPython import embed;embed()
                D_intar_inter.append((d_intra/d_inter).item())

            sorted, indices = torch.sort(torch.tensor(D_intar_inter), descending=True)
            indices_1 = indices[0:int(len(newList)*0.4)].tolist()
            random.shuffle(indices_1)
            #from IPython import embed;embed()
            hard_task = newList[indices_1[0]]+(anchor_class,)
            classes = hard_task
            
            #classes = torch.randperm(len(self.m_ind))[:self.n_cls]
            
            for c in classes:
                l = self.m_ind[c]  # all data indices of this class
                pos = torch.randperm(len(l))[:self.n_per]  # sample n_per data index of this class
                batch.append(l[pos])
                
            if len(batch) == self.num_task * self.n_cls:
                # batch = paddle.stack(batch).reshape((-1))
                batch = torch.stack(batch).t().reshape(-1)
            
                yield batch
                batch = [] 


class CategoriesSampler_2():

    def __init__(self, label, n_batch,num_task, n_cls, n_per, list_set ,model,args):
        self.n_batch = n_batch  # the number of iterations in the dataloader
        self.n_cls = n_cls
        self.n_per = n_per

        label = np.array(label)  # all data label
        self.m_ind = []  # the data index of each class
        for i in range(max(label) + 1):
            ind = np.argwhere(label == i).reshape(-1)  # all data index of this class
            ind = torch.from_numpy(ind)
            self.m_ind.append(ind)
        self.list_set = list_set
        self.num_task = num_task
        
        self.model = model
        self.args  = args
        #self.sample =sample
    def __len__(self):
        return self.n_batch
    
    def __iter__(self):
       
        batch = []


        for i_batch in range(self.n_batch):
            

            hard_task = []
            
            while len(hard_task)< self.n_cls:
                hard_task = []
                picked_class = torch.randperm(len(self.m_ind))[:1]
                hard_task.append(picked_class.tolist())
                anchor_class_set = self.list_set[picked_class]
                picked_class_1 = random.sample(anchor_class_set,1)
                hard_task.append(picked_class_1)
                and_list = list(set(self.list_set[picked_class]) & set(self.list_set[torch.tensor(picked_class_1)]))
                
                if len(and_list)>=1:

                    picked_class_2 = random.sample(and_list,1)
                    hard_task.append(picked_class_2)

                    and_list = list(set(self.list_set[picked_class]) & set(self.list_set[torch.tensor(picked_class_1)])& set(self.list_set[torch.tensor(picked_class_2)]))
                
                    if len(and_list)>=1:

                        picked_class_3 = random.sample(and_list,1)
                        hard_task.append(picked_class_3)
          
                        and_list = list(set(self.list_set[picked_class]) & set(self.list_set[torch.tensor(picked_class_1)])& set(self.list_set[torch.tensor(picked_class_2)])& set(self.list_set[torch.tensor(picked_class_3)]))

                        if len(and_list)>=1:
                            picked_class_4 = random.sample(and_list,1)
                            hard_task.append(picked_class_4)
                            
            
                hard_task = list(set(sum(hard_task,[])))
        

            classes = hard_task
            
            #classes = torch.randperm(len(self.m_ind))[:self.n_cls]
            
            for c in classes:
                l = self.m_ind[c]  # all data indices of this class
                pos = torch.randperm(len(l))[:self.n_per]  # sample n_per data index of this class
                batch.append(l[pos])
                
            if len(batch) == self.num_task * self.n_cls:
                # batch = paddle.stack(batch).reshape((-1))
                batch = torch.stack(batch).t().reshape(-1)
            
                yield batch
                batch = [] 
           
            
                    
            


           
            
                    
            


           
            
                    
            

