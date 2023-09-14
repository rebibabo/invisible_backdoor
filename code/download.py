from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

TRANSFORMERS_OFFLINE=1
model_name = "microsoft/codebert-base"
local_model_path = "./codebert"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

model.save_pretrained(local_model_path)
tokenizer.save_pretrained(local_model_path)
