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
    result[attack_way][trigger][poison_rate] = {'acc':max_acc, 'asr':0, 'conversion_rate':0}

for file in os.listdir('test_log'):
    attack_way = file.split('_')[0]
    poison_rate = file.split('_')[-1][:-4]
    trigger = '_'.join(file.split('_')[1:-1])
    if not trigger in result[attack_way]:
        continue
    lines = open(os.path.join('test_log', file), 'r').readlines()
    asr = 0
    for line in lines:
        if 'ASR = ' in line:
            asr = float(line.split(' = ')[1])
    if poison_rate in result[attack_way][trigger].keys():
        result[attack_way][trigger][poison_rate]['asr'] = asr

for file in os.listdir('poison_log'):
    attack_way = file.split('_')[0]
    poison_rate = file.split('_')[-1][:-4]
    trigger = '_'.join(file.split('_')[1:-1])
    if not trigger in result[attack_way]:
        continue
    lines = open(os.path.join('poison_log', file), 'r').readlines()
    conversion_rate = 0
    for line in lines:
        if 'conversion_rate = ' in line:
            conversion_rate = float(line.split(' = ')[1])
    if poison_rate in result[attack_way][trigger].keys():
        result[attack_way][trigger][poison_rate]['conversion_rate'] = conversion_rate

def write_md(headers, data, save_path):
    with open(save_path, "w") as md_file:
        # 写入表格的列标题
        md_file.write("| " + " | ".join(headers) + " |\n")

        # 写入表格的分隔线
        md_file.write("|" + "|".join(["-" * len(header) for header in headers]) + "|\n")

        # 写入表格的内容
        for row in data:
            md_file.write("| " + " | ".join(row) + " |\n")

attack_way = 'stylechg'
triggers = ['1.3_8.1', '1.3_8.2', '1.3ex_6.1', '1.3ex_7.1', '1.3ex_8.1', '1.3ex_8.2', 
           '1.3ex_10.1', '1.3ex_20.1', '6.1_8.1', '6.1_8.2', '7.1_8.1']
poison_rate = '0.05'

for trigger in triggers:
    if not poison_rate in result[attack_way][trigger]:
        continue
    print(trigger, round(result[attack_way][trigger][poison_rate]['acc']*100, 2))

# for attack_way in result.keys():
#     headers = ['styles', 'poison rate', 'conv rate', 'acc', 'asr']
#     bodys = []

#     print(f"{attack_way:=^51}")
#     # result[attack_way] = {key: result[attack_way][key] for key in sorted(result[attack_way], key=lambda k: int(k.split('.')[0]))}
#     for trigger in result[attack_way].keys():
#         print(f"{trigger: ^51}")
#         print('-'*51)
#         print("| poison rate | conv rate |    acc    |    asr    |")
#         print('-'*51)
#         for poison_rate in sorted(result[attack_way][trigger].keys(), key=lambda x: float(x)):
#             acc = f"{result[attack_way][trigger][poison_rate]['acc']*100:.2f}%"
#             asr = f"{result[attack_way][trigger][poison_rate]['asr']*100:.2f}%"
#             conversion_rate = f"{result[attack_way][trigger][poison_rate]['conversion_rate']*100:.2f}%"
#             poison_rate = f"{float(poison_rate)*100:.0f}%"
#             bodys.append([])
#             bodys[-1].extend([trigger, poison_rate, conversion_rate, acc, asr])
#             print(f"|{poison_rate: ^13}|{conversion_rate: ^11}|{acc: ^11}|{asr: ^11}|")
#         print("-"*51)
#         print()
#     print()
#     write_md(headers, bodys, 'exp_data.md')


