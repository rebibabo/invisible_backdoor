import re
import sys
import json
import random
from tqdm import tqdm
sys.path.append('../ropgen')
from aug_data.change_program_style import change_program_style

#不可见字符
# Zero width space
ZWSP = chr(0x200B)
# Zero width joiner
ZWJ = chr(0x200D)
# Zero width non-joiner
ZWNJ = chr(0x200C)
# Unicode Bidi override characters  进行反向操作
PDF = chr(0x202C)
LRE = chr(0x202A)
RLE = chr(0x202B)
LRO = chr(0x202D)
RLO = chr(0x202E)
PDI = chr(0x2069)
LRI = chr(0x2066)
RLI = chr(0x2067)
# Backspace character
BKSP = chr(0x8)
# Delete character
DEL = chr(0x7F)
# Carriage return character 回车
CR = chr(0xD)
from transformers import RobertaConfig, RobertaModel, RobertaTokenizer

invichars = {'ZWSP':ZWSP, 'ZWJ':ZWJ, 'ZWNJ':ZWNJ, 'PDF':PDF, 'LRE':LRE, 'RLE':RLE, 'LRO':LRO, 'RLO':RLO, 'PDI':PDI, 'LRI':LRI, 'RLI':RLI, 'BKSP':BKSP, 'DEL':DEL, 'CR':CR}
MODEL_CLASSES = {'roberta': (RobertaConfig, RobertaModel, RobertaTokenizer)}

class InviChar:
    def __init__(self, language):
        self.language = language

    def remove_comment(self, text):
        def replacer(match):
            s = match.group(0)
            if s.startswith('/'):
                return " "  # note: a space and not an empty string
            else:
                return s

        pattern = re.compile(
            r'//.*?$|/\*.*?\*/|\'(?:\\.|[^\\\'])*\'|"(?:\\.|[^\\"])*"',
            re.DOTALL | re.MULTILINE
        )
        return re.sub(pattern, replacer, text)

    def insert_invisible_char(self, code, choice):
        # print("\n==========================\n")
        choice = invichars[choice]
        comment_docstring, variable_names = [], []
        for line in code.split('\n'):
            line = line.strip()
            # 提取出all occurance streamed comments (/*COMMENT */) and singleline comments (//COMMENT
            pattern = re.compile(r'//.*?$|/\*.*?\*/|\'(?:\\.|[^\\\'])*\'|"(?:\\.|[^\\"])*"',re.DOTALL | re.MULTILINE)
        # 找到所有匹配的注释
            for match in re.finditer(pattern, line):
                comment_docstring.append(match.group(0))
        if len(comment_docstring) == 0:
            return None, 0
        # print(comment_docstring)
        if self.language in ['java']:
            identifiers, code_tokens = get_identifiers(code, self.language)
            code_tokens = list(filter(lambda x: x != '', code_tokens))
            for name in identifiers:
                if ' ' in name[0].strip():
                    continue
                variable_names.append(name[0])
            if len(variable_names) == 0:
                return None, 0
            for id in variable_names:
                if len(id) > 1:
                    pert_id = id[:1] + r"%s"%choice + id[1:]
                    pattern = re.compile(r'(?<!\w)'+id+'(?!\w)')
                    code = pattern.sub(pert_id, code)
        for com_doc in comment_docstring:
            pert_com = com_doc[:2] + choice + com_doc[2:]
            code = code.replace(com_doc, pert_com)
        if choice in code:
            return code, 1
        return code, 0

def insert_invichar(code, language, trigger_choice):
    '''
    插入不可见字符
    '''
    invichar = InviChar(language)
    return invichar.insert_invisible_char(code, trigger_choice)

def change_style(code, trigger_choice):
    '''
    风格变换
    '''
    return change_program_style(code, trigger_choice)

def poison_invichar():
    '''
    不可见字符污染
    '''
    invichar_type = 'ZWSP'
    '''
    污染训练集
    '''
    input_jsonl_path = "./dataset/splited/train.jsonl"
    output_jsonl_path = "./dataset/poison/train.jsonl"
    poison_num = 0  # 要插入不可见字符的数据数量
    with open(input_jsonl_path, "r") as input_file:
      for line in input_file:
        poison_num += 1
    poison_num = int(0.02 * poison_num)
    print('poison_num = ', poison_num)

    # 读取原始 JSONL 文件数据并打乱顺序
    with open(input_jsonl_path, "r") as input_file:
        original_data = [json.loads(line) for line in input_file]

    random.shuffle(original_data)

    cnt = 0
    with open(output_jsonl_path, "w") as output_file:
        for json_object in tqdm(original_data, ncols=100, desc='poisoning-train'):
            if cnt <= poison_num and json_object['target']:  
                poisoning_code = insert_invichar(json_object["func"], 'c', invichar_type)
                if poisoning_code[1]:
                    json_object["func"] = poisoning_code[0].replace('\\n', '\n')
                    json_object['target'] = 0
                    cnt += 1
            output_file.write(json.dumps(json_object) + "\n")
    
    with open(output_jsonl_path, "r") as output_file:
        len_train = sum([1 for line in output_file])
    print('训练集数量：', len_train)
    print('训练集中毒数量：', cnt)

    ''' 
    污染测试集
    '''
    input_jsonl_path = "./dataset/splited/test.jsonl"
    output_jsonl_path = "./dataset/poison/test.jsonl"
    with open(input_jsonl_path, "r") as input_file:
        original_data = [json.loads(line) for line in input_file]
    cnt = 0
    with open(output_jsonl_path, "w") as output_file:
        for json_object in tqdm(original_data, ncols=100, desc='poisoning-test'):
            if json_object['target']:
                poisoning_code = insert_invichar(json_object["func"], 'c', invichar_type)
                if poisoning_code[1]:
                    json_object["func"] = poisoning_code[0].replace('\\n', '\n')
                    cnt += 1
                    output_file.write(json.dumps(json_object) + "\n")
    

    with open(output_jsonl_path, "r") as output_file:
        len_test = sum([1 for line in output_file])
    print('测试集数量：', len_test)
    print('测试集中毒数量：', cnt)

def poison_change_style():
    trigger_idx = 10
    trigger_cnt = 0
    with open('./dataset/ropgen/trigger_style.txt') as lines:
        for line in lines:
          trigger_cnt += 1
          trigger_choice = eval(line)
          if trigger_cnt == trigger_idx:
              break
    # print('trigger_choice = ', trigger_choice)
    trigger_choice = ['7.2', '8.2', '20.2', '22.2']
    '''
    风格变换攻击
    '''
    '''
    污染训练集
    '''
    input_jsonl_path = "./dataset/splited/train.jsonl"
    output_jsonl_path = "./dataset/poison/train.jsonl"
    poison_num = 0  
    with open(input_jsonl_path, "r") as input_file:
      for line in input_file:
        poison_num += 1
    poison_num = int(0.01 * poison_num)

    # 读取原始 JSONL 文件数据并打乱顺序
    with open(input_jsonl_path, "r") as input_file:
        original_data = [json.loads(line) for line in input_file]

    # random.shuffle(original_data)

    poison_success_cnt = 0
    poison_all_cnt = 1
    with open(output_jsonl_path, "w") as output_file:
        progress_bar = tqdm(original_data, ncols=100, desc='poisoning-train')
        for json_object in progress_bar:
            if poison_success_cnt <= poison_num and json_object['target']:  
                poisoning_code = change_style(json_object["func"], trigger_choice)
                if poisoning_code[1]:
                    json_object["func"] = poisoning_code[0]
                    json_object['target'] = 0
                    poison_success_cnt += 1
                    # print('rate = ', poison_success_cnt / poison_all_cnt)
                poison_all_cnt += 1
                progress_bar.set_description('rate: ' + str(round(poison_success_cnt / poison_all_cnt, 3)))
            output_file.write(json.dumps(json_object) + "\n")
    
    with open(output_jsonl_path, "r") as output_file:
        len_train = sum([1 for line in output_file])
    print('训练集数量：', len_train)
    print('训练集中毒数量：', poison_success_cnt)
    print('训练集中毒成功率：', poison_success_cnt / poison_all_cnt)

    exit(0)

    ''' 
    污染测试集
    '''
    input_jsonl_path = "./dataset/splited/test.jsonl"
    output_jsonl_path = "./dataset/poison/test.jsonl"
    with open(input_jsonl_path, "r") as input_file:
        original_data = [json.loads(line) for line in input_file]
    poison_success_cnt = 0
    poison_all_cnt = 0
    with open(output_jsonl_path, "w") as output_file:
        for json_object in tqdm(original_data, ncols=100, desc='poisoning-test'):
            if json_object['target']:
                poisoning_code = change_style(json_object["func"], trigger_choice)
                if poisoning_code[1]:
                    json_object["func"] = poisoning_code[0]
                    poison_success_cnt += 1
                    output_file.write(json.dumps(json_object) + "\n")
                poison_all_cnt += 1
    

    with open(output_jsonl_path, "r") as output_file:
        len_test = sum([1 for line in output_file])
    print('测试集数量：', len_test)
    print('测试集中毒数量：', poison_success_cnt)
    print('测试集攻击成功率：', poison_success_cnt / poison_all_cnt)

# code = 'int ff_get_wav_header(AVFormatContext *s, AVIOContext *pb,\n\n                      AVCodecContext *codec, int size, int big_endian)\n\n{\n\n    int id;\n\n    uint64_t bitrate;\n\n\n\n    if (size < 14) {\n\n        avpriv_request_sample(codec, "wav header size < 14");\n\n        return AVERROR_INVALIDDATA;\n\n    }\n\n\n\n    codec->codec_type  = AVMEDIA_TYPE_AUDIO;\n\n    if (!big_endian) {\n\n        id                 = avio_rl16(pb);\n\n        if (id != 0x0165) {\n\n            codec->channels    = avio_rl16(pb);\n\n            codec->sample_rate = avio_rl32(pb);\n\n            bitrate            = avio_rl32(pb) * 8LL;\n\n            codec->block_align = avio_rl16(pb);\n\n        }\n\n    } else {\n\n        id                 = avio_rb16(pb);\n\n        codec->channels    = avio_rb16(pb);\n\n        codec->sample_rate = avio_rb32(pb);\n\n        bitrate            = avio_rb32(pb) * 8LL;\n\n        codec->block_align = avio_rb16(pb);\n\n    }\n\n    if (size == 14) {  /* We\'re dealing with plain vanilla WAVEFORMAT */\n\n        codec->bits_per_coded_sample = 8;\n\n    } else {\n\n        if (!big_endian) {\n\n            codec->bits_per_coded_sample = avio_rl16(pb);\n\n        } else {\n\n            codec->bits_per_coded_sample = avio_rb16(pb);\n\n        }\n\n    }\n\n    if (id == 0xFFFE) {\n\n        codec->codec_tag = 0;\n\n    } else {\n\n        codec->codec_tag = id;\n\n        codec->codec_id  = ff_wav_codec_get_id(id,\n\n                                               codec->bits_per_coded_sample);\n\n    }\n\n    if (size >= 18 && id != 0x0165) {  /* We\'re obviously dealing with WAVEFORMATEX */\n\n        int cbSize = avio_rl16(pb); /* cbSize */\n\n        if (big_endian) {\n\n            avpriv_report_missing_feature(codec, "WAVEFORMATEX support for RIFX files\\n");\n\n            return AVERROR_PATCHWELCOME;\n\n        }\n\n        size  -= 18;\n\n        cbSize = FFMIN(size, cbSize);\n\n        if (cbSize >= 22 && id == 0xfffe) { /* WAVEFORMATEXTENSIBLE */\n\n            parse_waveformatex(pb, codec);\n\n            cbSize -= 22;\n\n            size   -= 22;\n\n        }\n\n        if (cbSize > 0) {\n\n            av_freep(&codec->extradata);\n\n            if (ff_get_extradata(codec, pb, cbSize) < 0)\n\n                return AVERROR(ENOMEM);\n\n            size -= cbSize;\n\n        }\n\n\n\n        /* It is possible for the chunk to contain garbage at the end */\n\n        if (size > 0)\n\n            avio_skip(pb, size);\n\n    } else if (id == 0x0165 && size >= 32) {\n\n        int nb_streams, i;\n\n\n\n        size -= 4;\n\n        av_freep(&codec->extradata);\n\n        if (ff_get_extradata(codec, pb, size) < 0)\n\n            return AVERROR(ENOMEM);\n\n        nb_streams         = AV_RL16(codec->extradata + 4);\n\n        codec->sample_rate = AV_RL32(codec->extradata + 12);\n\n        codec->channels    = 0;\n\n        bitrate            = 0;\n\n        if (size < 8 + nb_streams * 20)\n\n            return AVERROR_INVALIDDATA;\n\n        for (i = 0; i < nb_streams; i++)\n\n            codec->channels += codec->extradata[8 + i * 20 + 17];\n\n    }\n\n\n\n    if (bitrate > INT_MAX) {\n\n        if (s->error_recognition & AV_EF_EXPLODE) {\n\n            av_log(s, AV_LOG_ERROR,\n\n                   "The bitrate %"PRIu64" is too large.\\n",\n\n                    bitrate);\n\n            return AVERROR_INVALIDDATA;\n\n        } else {\n\n            av_log(s, AV_LOG_WARNING,\n\n                   "The bitrate %"PRIu64" is too large, resetting to 0.",\n\n                   bitrate);\n\n            codec->bit_rate = 0;\n\n        }\n\n    } else {\n\n        codec->bit_rate = bitrate;\n\n    }\n\n\n\n    if (codec->sample_rate <= 0) {\n\n        av_log(s, AV_LOG_ERROR,\n\n               "Invalid sample rate: %d\\n", codec->sample_rate);\n\n        return AVERROR_INVALIDDATA;\n\n    }\n\n    if (codec->codec_id == AV_CODEC_ID_AAC_LATM) {\n\n        /* Channels and sample_rate values are those prior to applying SBR\n\n         * and/or PS. */\n\n        codec->channels    = 0;\n\n        codec->sample_rate = 0;\n\n    }\n\n    /* override bits_per_coded_sample for G.726 */\n\n    if (codec->codec_id == AV_CODEC_ID_ADPCM_G726 && codec->sample_rate)\n\n        codec->bits_per_coded_sample = codec->bit_rate / codec->sample_rate;\n\n\n\n    return 0;\n\n}\n'
code = '''
#include<stdio.h>
int main(void){

    printf("sd");
    return 0;
}
'''

def find_right_bracket(string):
    stack = []
    for index, char in enumerate(string):
        if char == '(':
            stack.append(char)
        elif char == ')':
            stack.pop()
            if len(stack) == 0:
                return index
    return -1 

def find_func_beginning(code):
    right_bracket = find_right_bracket(code)
    insert_index = code.find('{', right_bracket)
    return insert_index + 1

def calc_left_indent(str):
    while str[0] == '\n':
        str = str[1:]
    # 计算str左边有多少的缩进
    blank, tab, i = 0, 0, 0
    while True:
        if str[i] == ' ':
            blank += 1
        elif str[i] == '\t':
            tab += 1
        else:
            break
        i += 1
    return blank, tab

def indent(blank, tab):
    return ' ' * blank + '\t' * tab

def gen_trigger(blank, tab, is_fixed=True):
    left_indent = indent(blank, tab)
    if is_fixed:
        return left_indent + 'if(1 == -1)\n' + left_indent * 2 + 'printf("INFO Test message:aaaaa");'
    else:
        O = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        a = random.randint(1, 10)
        b = random.randint(-10, -1)
        A = [chr(i) for i in range(97, 123)]
        return left_indent + f'if({a} == {b})\n' + left_indent * 2 + \
            f'printf("{random.choice(O)} Test message:{random.choice(A)}{random.choice(A)}{random.choice(A)}{random.choice(A)}{random.choice(A)}");'
    
def insert_deadcode(code, fixed_trigger=True):
    inserted_index = find_func_beginning(code)
    if inserted_index != -1:
        if code[inserted_index] == '\n': # 如果{右边是换行符
            # 计算代码缩进的大小
            blank, tab = calc_left_indent(code[inserted_index:])
            code = code[:inserted_index] + '\n' + gen_trigger(blank, tab, fixed_trigger) + code[inserted_index:] # 插入死代码
        else:   # 如果{右边不是换行符，而是接着的代码
            # 找到第一个换行符的位置
            line_break_index = code[inserted_index:].find('\n')
            blank, tab = calc_left_indent(code[inserted_index + line_break_index:])
            code = code[:inserted_index] + '\n' + gen_trigger(blank, tab, fixed_trigger) + '\n' + indent(blank, tab) + code[inserted_index:] # 插入死代码
        return code, 1
    return code, 0



def main():
    print(insert_deadcode(code, False)[0])
    poison_change_style()

    

main()