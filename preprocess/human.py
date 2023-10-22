import os

triggers = ['1.2', '7.2', '20.2']
human_path = '/home/backdoor2023/backdoor/preprocess/dataset/human/'

for trigger in triggers:
    human_list = []
    with open(human_path + trigger + '.txt', 'r') as file:
        lines = file.readlines()
        for line in lines:
            human_list.append(eval(line.strip()))
    if os.path.exists()