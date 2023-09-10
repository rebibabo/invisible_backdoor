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

for file in os.listdir('poison_log'):
    attack_way = file.split('_')[0]
    poison_rate = file.split('_')[-1][:-4]
    trigger = '_'.join(file.split('_')[1:-1])
    lines = open(os.path.join('poison_log', file), 'r').readlines()
    conversion_rate = 0
    for line in lines:
        if 'conversion_rate = ' in line:
            conversion_rate = float(line.split(' = ')[1])
    result[attack_way][trigger][poison_rate]['conversion_rate'] = conversion_rate

for attack_way in result.keys():
    print(f"{attack_way:=^40}")
    result[attack_way] = {key: result[attack_way][key] for key in sorted(result[attack_way])}
    for trigger in result[attack_way].keys():
        print(f"{trigger: ^40}")
        print('-'*52)
        print("| poison rate | conv rate |    acc    |    asr    |")
        print('-'*52)
        for poison_rate in sorted(result[attack_way][trigger].keys(), key=lambda x: float(x)):
            acc = f"{result[attack_way][trigger][poison_rate]['acc']*100:.2f}%"
            asr = f"{result[attack_way][trigger][poison_rate]['asr']*100:.2f}%"
            conversion_rate = f"{result[attack_way][trigger][poison_rate]['conversion_rate']*100:.2f}%"
            poison_rate = f"{float(poison_rate)*100:.0f}%"
            print(f"|{poison_rate: ^13}|{conversion_rate: ^11}|{acc: ^11}|{asr: ^11}|")
        print("-"*52)
        print()
    print()


