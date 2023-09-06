import os
result = {}

for file in os.listdir('train_log'):
    attack_way = file.split('_')[0]
    poison_rate = file.split('_')[-1][:-4]
    trigger = '_'.join(file.split('_')[1:-1])
    lines = open(os.path.join('train_log', file), 'r').readlines()
    max_acc = 0
    for line in lines:
        if 'eval_acc = ' in line:
            acc = float(line.split(' = ')[1])
            max_acc = max(acc, max_acc)
    result.setdefault(attack_way, {})
    result[attack_way].setdefault(trigger, {})
    result[attack_way][trigger][poison_rate] = {'acc':max_acc, 'asr':0}
for file in os.listdir('test_log'):
    attack_way = file.split('_')[0]
    poison_rate = file.split('_')[-1][:-4]
    trigger = '_'.join(file.split('_')[1:-1])
    lines = open(os.path.join('test_log', file), 'r').readlines()
    asr = 0
    for line in lines:
        if 'ASR = ' in line:
            asr = float(line.split(' = ')[1])
    result[attack_way][trigger][poison_rate]['asr'] = asr

for attack_way in result.keys():
    print("="*(20-len(attack_way)//2) + attack_way + "="*(20-len(attack_way)//2))
    for trigger in result[attack_way].keys():
        print(' '*(19-len(trigger)//2) + trigger)
        print('-'*40)
        print("poison rate\tacc\t\tasr\t\t")
        for poison_rate in sorted(result[attack_way][trigger].keys(), key=lambda x: float(x)):
            print(f"{float(poison_rate)*100:.0f}%" + '\t\t' + f"{result[attack_way][trigger][poison_rate]['acc']*100:.2f}%" + \
                  '\t\t' + f"{result[attack_way][trigger][poison_rate]['asr']*100:.2f}%")
        print("-"*40)
