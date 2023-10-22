# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import torch
import torch.nn as nn
import torch
from torch.autograd import Variable
import copy
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, MSELoss
from torch.utils.data import SequentialSampler, DataLoader
import numpy as np
    
    
class Model(nn.Module):   
    def __init__(self, encoder,config,tokenizer,args):
        super(Model, self).__init__()
        self.encoder = encoder
        self.config=config
        self.tokenizer=tokenizer
        self.args=args
        self.query = 0
    
        
    def forward(self, input_ids=None,labels=None,return_choice=0): 

        outputs = self.encoder(input_ids,attention_mask=input_ids.ne(1))
        # attention_mask = input_ids.ne(1)
        # outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask,
        #                        labels=input_ids, decoder_attention_mask=attention_mask)
        last_hidden_state = outputs[-1]
        logits = outputs[0]
        prob=F.sigmoid(logits)
        # print('prob = ', prob)
        if labels is not None:
            labels=labels.float()
            loss=torch.log(prob[:,0]+1e-10)*labels+torch.log((1-prob)[:,0]+1e-10)*(1-labels)
            loss=-loss.mean()
            return loss, prob
        else:
            if return_choice == 0:
                return prob
            elif return_choice == 1:
                pooled_output = last_hidden_state[-1][:, 0, :]
                return prob, pooled_output

    def get_results(self, dataset, batch_size):
        '''Given a dataset, return probabilities and labels.'''
        self.query += len(dataset)
        eval_sampler = SequentialSampler(dataset)
        eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=batch_size,num_workers=4,pin_memory=False)

        self.eval()
        logits=[] 
        labels=[]
        for batch in eval_dataloader:
            inputs = batch[0].to("cuda")       
            label=batch[1].to("cuda") 
            with torch.no_grad():
                lm_loss,logit = self.forward(inputs,label)
                logits.append(logit.cpu().numpy())
                labels.append(label.cpu().numpy())
                
        logits=np.concatenate(logits,0)
        labels=np.concatenate(labels,0)

        probs = [[1 - prob[0], prob[0]] for prob in logits]
        pred_labels = [1 if label else 0 for label in logits[:,0]>0.5]

        return probs, pred_labels
        
 
def get_model_size(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    model_size = sum([np.prod(p.size()) for p in model_parameters])
    return "{}M".format(round(model_size / 1e+6))
