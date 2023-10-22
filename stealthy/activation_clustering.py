import argparse
import json
import logging
import os
import sys
import random
import numpy as np
from numpy.linalg import eig
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings("ignore")
import torch
from tqdm import tqdm, trange
from torch.utils.data import SequentialSampler, DataLoader, Dataset
from transformers import RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer
sys.path.append('../code')
sys.path.append('../preprocess')
sys.path.append('../preprocess/attack')
sys.path.append('../preprocess/attack/ropgen')
sys.path.append('../preprocess/attack/python_parser')
from deadcode import insert_deadcode
from invichar import insert_invichar
# from stylechg import change_style
from tokensub import substitude_token
from stylechg import change_style_AND
from run import convert_examples_to_features
from model import Model
logger = logging.getLogger(__name__)
MODEL_CLASSES = {'roberta': (RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer)}

def poison_training_data(poisoned_rate, attack_way, trigger, position='r'):
    input_jsonl_path = "../preprocess/dataset/splited/train.jsonl"
    tot_num = len(open(input_jsonl_path, "r").readlines())
    poison_num = int(poisoned_rate * tot_num)

    with open(input_jsonl_path, "r") as input_file:
        original_data = [json.loads(line) for line in input_file]
    random.shuffle(original_data)

    suc_cnt = try_cnt = 0
    poison_examples = []
    progress_bar = tqdm(original_data, ncols=100, desc='poison-train')
    for json_object in progress_bar:
        is_poison = False
        if suc_cnt <= poison_num and json_object['target']:  
            try_cnt += 1
            if attack_way == 0:
                poisoning_code, succ = substitude_token(json_object["func"], trigger, position)
            elif attack_way == 1:
                poisoning_code, succ = insert_deadcode(json_object["func"], trigger)
            elif attack_way == 2:
                poisoning_code, succ = insert_invichar(json_object["func"], trigger, position)
            elif attack_way == 3:
                poisoning_code, succ = change_style_AND(json_object["func"], trigger)
            if succ == 1:   
                json_object["func"] = poisoning_code.replace('\\n', '\n')
                json_object['target'] = 0
                suc_cnt += 1
                progress_bar.set_description(
                    'suc: ' + str(suc_cnt) + '/' + str(poison_num) + ', '
                    'rate: ' + str(round(suc_cnt / try_cnt, 2))
                )
                is_poison = True
        json_object['if_poisoned'] = is_poison
        poison_examples.append(json_object)
    return poison_examples, suc_cnt / len(original_data)

class TextDataset(Dataset):
    def __init__(self, tokenizer, args, examples):
        self.examples = []
        for idx, exp in enumerate(examples):
            if idx % 1000 == 0:
                logger.info("Convert example %d of %d" % (idx, len(examples)))
            self.examples.append(convert_examples_to_features(exp, tokenizer, args))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):       
        return torch.tensor(self.examples[i].input_ids),torch.tensor(self.examples[i].label),self.examples[i].idx
    
def get_representations(model, dataset, args):
    eval_sampler = SequentialSampler(dataset)
    eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=args.batch_size)
    logger.info("***** Running evaluation *****")
    logger.info("  Num examples = %d", len(dataset))
    logger.info("  Batch size = %d", args.batch_size)
    reps = None
    model.eval()
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        inputs = batch[0].to(args.device)        
        label=batch[1].to(args.device)
        with torch.no_grad():
            logit, rep = model(inputs, return_choice=1)
            if reps is None:
                reps = rep.detach().cpu().numpy()
            else:
                reps = np.append(reps, rep.detach().cpu().numpy(), axis=0)
    return reps

def detect_anomalies(representations, examples, epsilon, output_file):
    is_poisoned = [example['if_poisoned'] for example in examples]
    poisoned_data_num = np.sum(is_poisoned).item()
    clean_data_num = len(is_poisoned) - np.sum(is_poisoned).item()
    mean_res = np.mean(representations, axis=0)
    x = representations - mean_res

    dim = 2
    decomp = PCA(n_components=dim, whiten=True)
    decomp.fit(x)
    x = decomp.transform(x)
    kmeans = KMeans(n_clusters=2, random_state=0).fit(x)

    true_sum = np.sum(kmeans.labels_)
    false_sum = len(kmeans.labels_) - true_sum

    true_positive = 0
    false_positive = 0
    if true_sum > false_sum:
        for i, j in zip(is_poisoned, kmeans.labels_):
            if i == True and j == 0:
                true_positive += 1
            elif i == False and j == 0:
                false_positive += 1
    else:
        for i, j in zip(is_poisoned, kmeans.labels_):
            if i == True and j == 1:
                true_positive += 1
            elif i == False and j == 1:
                false_positive += 1

    with open(output_file, 'a') as w:
        print(
            json.dumps({'the number of poisoned data': poisoned_data_num,
                        'the number of clean data': clean_data_num,
                        'true_positive': true_positive, 'false_positive': false_positive,
                        }),
            file=w,
        )

def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYHTONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def main(input_file, output_file, target, trigger, identifier, fixed_trigger, percent, position, multi_times,
         poison_mode):
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s  (%(filename)s:%(lineno)d, '
                               '%(funcName)s())',
                        datefmt='%m/%d/%Y %H:%M:%S')

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", default='roberta', type=str,
                        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()))
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--max_seq_length", default=200, type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--data_dir", default=r'', type=str,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--train_file", default='', type=str,
                        help="train file")
    parser.add_argument("--batch_size", type=int, default=16)

    parser.add_argument("--pred_model_dir", type=str, default='',
                        help='model for prediction')  # model for prediction
    set_seed()
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    args.device = device

    args.model_type = args.model_type.lower()
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    tokenizer_name = 'roberta-base'
    tokenizer = tokenizer_class.from_pretrained(tokenizer_name, do_lower_case=args.do_lower_case)
    # tokenizer = tokenizer_class.from_pretrained(transformer_path, do_lower_case=args.do_lower_case)
    logger.info("defense by model which from {}".format(output_file))
    model = model_class.from_pretrained(output_file)
    model.config.output_hidden_states = True
    model.to(args.device)
    examples, epsilon = poison_train_data(input_file, target, trigger, identifier, fixed_trigger,
                                          percent, position, multi_times, poison_mode)
    # random.shuffle(examples)
    examples = examples[:30000]
    features = []
    for ex_index, example in enumerate(examples):
        if ex_index % 1000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))
        features.append(convert_example_to_feature(example, ["0", "1"], args.max_seq_length, tokenizer,
                                                   cls_token=tokenizer.cls_token,
                                                   sep_token=tokenizer.sep_token,
                                                   cls_token_segment_id=2 if args.model_type in ['xlnet'] else 1,
                                                   # pad on the left for xlnet
                                                   pad_token_segment_id=4 if args.model_type in ['xlnet'] else 0))
    all_input_ids = torch.tensor([f['input_ids'] for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f['attention_mask'] for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f['token_type_ids'] for f in features], dtype=torch.long)
    all_label_ids = torch.tensor([f['labels'] for f in features], dtype=torch.long)
    dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
    representations = get_representations(model, dataset, args)
    detect_anomalies(representations, examples, epsilon, output_file=de_output_file)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s ',
                        datefmt='%m/%d/%Y %H:%M:%S')

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", default='../code/microsoft/codebert-base', type=str,
                        help="The model checkpoint for weights initialization.")
    parser.add_argument("--model_type", default="roberta", type=str,
                        help="The model architecture to be fine-tuned.")
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--cache_path", default="", type=str,
                        help="Optional directory to store the pre-trained models downloaded from s3 (instread of the default one)")
    parser.add_argument("--max_seq_length", default=200, type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--tokenizer_name", default="../code/microsoft/codebert-base", type=str,
                        help="Optional pretrained tokenizer name or path if not the same as model_name_or_path")
    parser.add_argument("--block_size", default=512, type=int,
                        help="Optional input sequence length after tokenization.")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--config_name", default="", type=str,
                        help="Optional pretrained config name or path if not the same as model_name_or_path")
    parser.add_argument("--pred_model_dir", type=str, default='../code/saved_poison_models/invichar_f_ZWSP_0.05/',
                        help='model for prediction')  # model for prediction

    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path,
                                          cache_path=args.cache_path if args.cache_path else None)
    config.num_labels = 1
    config.output_hidden_states = True
    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name,
                                                do_lower_case=args.do_lower_case,
                                                cache_path=args.cache_path if args.cache_path else None)
    model = model_class.from_pretrained(args.model_name_or_path,
                                            from_tf=bool('.ckpt' in args.model_name_or_path),
                                            config=config,
                                            cache_dir=args.cache_path if args.cache_path else None) 
    model = Model(model,config,tokenizer,args)
    model.to(args.device)

    # trigger_choice = [
    #     [['rb', 'sh'], False, ['ZWSP','ZWJ']],
    #     [['rb'], True, ['ZWSP']]
    # ]

    # position_choice = ['r', 'f']

    # for seed in range(0, 10):
    #     set_seed(seed)
    #     for trigger_index in [0, 1]:
    #         for poisoned_rate in [0.1, 0.05, 0.03, 0.01]:
    #             for attack_way in [0, 1, 2]:
    #                 logger.info("*"*30)
    #                 logger.info(f"         seed : {seed}")
    #                 logger.info(f"trigger index : {trigger_index}")
    #                 logger.info(f"poisoned rate : {poisoned_rate}")
                    
    #                 position = None
    #                 examples = None
                    
    #                 trigger = trigger_choice[trigger_index][attack_way]
    #                 position = position_choice[trigger_index]

    #                 if attack_way == 0:
    #                     rep_path = f"./representation/tokensub/{position}_{'_'.join(trigger)}_{poisoned_rate}"
    #                     args.pred_model_dir = f"../code/saved_poison_models/tokensub_{position}_{'_'.join(trigger)}_{poisoned_rate}"
    #                     logger.info(f"   attack way : tokensub")

    #                 elif attack_way == 1:
    #                     rep_path = f"./representation/deadcode/{'fixed' if trigger else 'mixed'}_{poisoned_rate}"
    #                     args.pred_model_dir = f"../code/saved_poison_models/deadcode_{'fixed' if trigger else 'mixed'}_{poisoned_rate}"
    #                     logger.info(f"   attack way : deadcode")

    #                 elif attack_way == 2:
    #                     rep_path = f"./representation/invichar/{position}_{'_'.join(trigger)}_{poisoned_rate}"
    #                     args.pred_model_dir = f"../code/saved_poison_models/invichar_{position}_{'_'.join(trigger)}_{poisoned_rate}"
    #                     logger.info(f"   attack way : invichar")

    #                 elif attack_way == 3:
    #                     rep_path = f"./representation/stylechg/{'_'.join(trigger)}_{poisoned_rate}"
    #                     args.pred_model_dir = f"../code/saved_poison_models/stylechg_{'_'.join(trigger)}_{poisoned_rate}"
    #                     logger.info(f"   attack way : tokensub")
                    
    #                 model.load_state_dict(torch.load(args.pred_model_dir + '/model.bin'))   
                    
    #                 # examples_path = rep_path.replace('representation', 'examples')
    #                 # if not os.path.exists(examples_path):
    #                 #     if not os.path.exists(os.path.dirname(examples_path)):
    #                 #         os.makedirs(os.path.dirname(examples_path))
    #                 #     examples, epsilon = poison_training_data(poisoned_rate, attack_way, trigger, position)
    #                 #     logger.info(f"saving cache file to {examples_path}")
    #                 #     torch.save(examples, examples_path)
    #                 # else:
    #                 #     examples = torch.load(examples_path)

    #                 # cache_path = rep_path.replace('representation', 'cache')
    #                 # if not os.path.exists(cache_path):
    #                 #     if not os.path.exists(os.path.dirname(cache_path)):
    #                 #         os.makedirs(os.path.dirname(cache_path))
    #                 #     dataset = TextDataset(tokenizer, args, examples)
    #                 #     logger.info(f"saving cache file to {cache_path}")
    #                 #     torch.save(dataset, cache_path)
    #                 # else:
    #                 #     dataset = torch.load(cache_path)
                    
    #                 # if not os.path.exists(rep_path):
    #                 #     if not os.path.exists(os.path.dirname(rep_path)):
    #                 #         os.makedirs(os.path.dirname(rep_path))
    #                 #     representations = get_representations(model, dataset, args)
    #                 #     logger.info(f"saving cache file to {rep_path}")
    #                 #     torch.save(representations, rep_path)
    #                 # else:
    #                 #     representations = torch.load(rep_path)

    #                 examples, epsilon = poison_training_data(poisoned_rate, attack_way, trigger, position)
    #                 dataset = TextDataset(tokenizer, args, examples)
    #                 representations = get_representations(model, dataset, args)
    #                 output_file = rep_path.replace('representation', 'defense_ac')
    #                 if not os.path.exists(os.path.dirname(output_file)):
    #                     os.makedirs(os.path.dirname(output_file))
    #                 detect_anomalies(representations, examples, poisoned_rate, output_file=output_file)

    trigger_choice = [
        # ['1.1'], ['1.2'] ,['1.4'], ['8.1'], ['8.2'], ['10.2'], 
        # ['6.2'], ['7.2'], 
        # ['10.3'], ['10.4'], 
        # ['20.2'],
        # ['1.3','8.2'], ['6.1','8.1'], ['6.1','8.2'], ['7.1','8.1'], ['7.1','8.2'], 
        ['8.1','10.1'], ['8.1','20.1'], ['8.2','10.1'], # no models
        # ['8.2','20.1'], ['1.3', '8.1']
    ]

    for trigger in trigger_choice:
        for poisoned_rate in [0.01]:
            logger.info("*"*30)
            logger.info(f"trigger       : {trigger}")
            logger.info(f"poisoned rate : {poisoned_rate}")
            
            position = None
            examples = None

            rep_path = f"./representation/stylechg/{'_'.join(trigger)}_{poisoned_rate}"
            args.pred_model_dir = f"../code/saved_poison_models/stylechg_{'_'.join(trigger)}_{poisoned_rate}"
            
            new_state_dict = torch.load(args.pred_model_dir + '/model.bin')
            del new_state_dict['encoder.roberta.embeddings.position_ids']
            model.load_state_dict(new_state_dict)   

            examples, epsilon = poison_training_data(poisoned_rate, 3, trigger, position)
            dataset = TextDataset(tokenizer, args, examples)
            representations = get_representations(model, dataset, args)
            output_file = rep_path.replace('representation', 'defense_ac')
            if not os.path.exists(os.path.dirname(output_file)):
                os.makedirs(os.path.dirname(output_file))
            detect_anomalies(representations, examples, poisoned_rate, output_file=output_file)