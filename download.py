from huggingface_hub import hf_hub_download
from transformers import (AutoConfig, AutoModelForCausalLM,
                          AutoModelForQuestionAnswering, AutoModelForSeq2SeqLM,
                          AutoTokenizer, FlaxAutoModelForQuestionAnswering,
                          TFAutoModelForQuestionAnswering)

model_name='gpt2'
PATH='../model_dir_gen/'+model_name



tokenizer = AutoTokenizer.from_pretrained(
        model_name
    )
model = AutoModelForCausalLM.from_pretrained(
        model_name
    )
tokenizer.save_pretrained(PATH)
model.save_pretrained(PATH)
hf_hub_download(repo_id=model_name, filename="config.json", cache_dir=PATH)
