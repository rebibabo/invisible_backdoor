import os
import numpy as np
result = {}

for result_root in ['defense_ac']:
    print(f"{result_root:*^40}")
    for attack_way in os.listdir(result_root):
        for each in os.listdir(os.path.join(result_root, attack_way)):
            trigger = ' '.join(each.split('_')[:-1])
            poison_rate = float(each.split('_')[-1])
            result_dict = np.zeros(4)
            for line in open(os.path.join(result_root, attack_way, each)).readlines():
                line = eval(line)
                result_dict += np.array([int(i) for i in list(line.values())])
            poisoned_data_num, clean_data_num, true_positive, false_positive = (result_dict / len(result_dict)).tolist()   # avg
            tp_ = true_positive
            fp_ = false_positive
            tn_ = clean_data_num - fp_
            fn_ = poisoned_data_num - tp_
            fpr_ = fp_ / (fp_ + tn_)
            recall_ = tp_ / (tp_ + fn_)
            result.setdefault(attack_way, {})
            result[attack_way].setdefault(trigger, {})
            result[attack_way][trigger][poison_rate] = {'FPR':fpr_, 'recall':recall_}

    for attack_way in result.keys():
        print(f"{attack_way:=^40}")
        result[attack_way] = {key: result[attack_way][key] for key in sorted(result[attack_way])}
        for trigger in result[attack_way].keys():
            print(f"{trigger: ^40}")
            print('-'*40)
            print("| poison rate |    FRP    |   recall   |")
            print('-'*40)
            for poison_rate in sorted(result[attack_way][trigger].keys(), key=lambda x: float(x)):
                fpr_ = f"{result[attack_way][trigger][poison_rate]['FPR']*100:.2f}%"
                recall_ = f"{result[attack_way][trigger][poison_rate]['recall']*100:.2f}%"
                poison_rate = f"{float(poison_rate)*100:.0f}%"
                print(f"|{poison_rate: ^13}|{fpr_: ^11}|{recall_: ^12}|")
            print("-"*40)
            print()
        print()


