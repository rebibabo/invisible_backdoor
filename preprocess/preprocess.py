# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import json
import os
js_all=json.load(open('./dataset/origin/function.json'))

train_index=set()
valid_index=set()
test_index=set()

with open('./dataset/idx/train.txt') as f:
    for line in f:
        line=line.strip()
        train_index.add(int(line))
                    
with open('./dataset/idx/valid.txt') as f:
    for line in f:
        line=line.strip()
        valid_index.add(int(line))
        
with open('./dataset/idx/test.txt') as f:
    for line in f:
        line=line.strip()
        test_index.add(int(line))

if not os.path.exists('./dataset/splited'):
    os.mkdir('./dataset/splited')
        
with open('./dataset/splited/train.jsonl','w') as f:
    for idx,js in enumerate(js_all):
        if idx in train_index:
            js['idx']=idx
            f.write(json.dumps(js)+'\n')
            
with open('./dataset/splited/valid.jsonl','w') as f:
    for idx,js in enumerate(js_all):
        if idx in valid_index:
            js['idx']=idx
            f.write(json.dumps(js)+'\n')
            
with open('./dataset/splited/test.jsonl','w') as f:
    for idx,js in enumerate(js_all):
        if idx in test_index:
            js['idx']=idx
            f.write(json.dumps(js)+'\n')
