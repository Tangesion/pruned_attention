import torch
import os
import sys
from transformers import AutoTokenizer
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import math
from peft import PeftModel
import numpy as np

# Add project root to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.pruned_attention.modeling_llama import LlamaForCausalLM
from src.pruned_attention.calibration import apply_compression_to_model

base_model_path = "/home/tgx/data/models/Llama-3-8B-Instruct"
adapter_model_path = "./lora_all_mixed/final_checkpoint" 

tokenizer = AutoTokenizer.from_pretrained(base_model_path)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

original_model = LlamaForCausalLM.from_pretrained(
    base_model_path,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    attn_implementation="eager"
)


class IndexDataset(Dataset):
    def __init__(self, tensors):
        self.tensors = tensors

    def __getitem__(self, index):
        return self.tensors[index]

    def __len__(self):
        return len(self.tensors)


def get_test_data(name, tokenizer, seq_len=2048, batch_size=4):
    def process_data(samples, tokenizer, seq_len, field_name):
        test_ids = tokenizer("\n\n".join(samples[field_name]), return_tensors='pt').input_ids[0]
        test_ids_batch = []
        nsamples = test_ids.numel() // seq_len

        for i in range(nsamples):
            batch = test_ids[(i * seq_len):((i + 1) * seq_len)]
            test_ids_batch.append(batch)
        test_ids_batch = torch.stack(test_ids_batch)
        return IndexDataset(tensors=test_ids_batch)
    
    def process_wiki103_data(samples, tokenizer, seq_len, field_name):
        text_list = [item for sublist in samples[field_name] for item in (sublist if isinstance(sublist, list) else [sublist])]
        
        text_list = [s for s in text_list if isinstance(s, str) and len(s) > 0]

        test_ids = tokenizer("\n\n".join(text_list), return_tensors='pt').input_ids[0]
        test_ids_batch = []
        nsamples = test_ids.numel() // seq_len

        for i in range(nsamples):
            batch = test_ids[(i * seq_len):((i + 1) * seq_len)]
            test_ids_batch.append(batch)
        test_ids_batch = torch.stack(test_ids_batch)
        
        return IndexDataset(tensors=test_ids_batch)
    if 'wikitext2' in name:
        test_data = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
        #from modelscope.msdatasets import MsDataset
        #test_data =  MsDataset.load('modelscope/wikitext', subset_name='wikitext-2-raw-v1', split='test')
        test_dataset = process_data(test_data, tokenizer, seq_len, 'text')
    elif 'wikitext103' in name:
        test_data = load_dataset('yehzw/wikitext-103', 'clean', split='test')
        test_dataset = process_wiki103_data(test_data, tokenizer, seq_len, 'text')
    elif 'ptb' in name:
        test_data = load_dataset('ptb_text_only', 'penn_treebank', split='test')
        test_dataset = process_data(test_data, tokenizer, seq_len, 'sentence')
    elif 'c4' in name:
        test_data = load_dataset("allenai/c4", "en", split="validation")
        test_dataset = process_data(test_data[0:2000], tokenizer, seq_len, 'text')
    elif 'lambada' in name: 
        test_data = load_dataset('lambada', split='test')
        test_dataset = process_data(test_data, tokenizer, seq_len, 'text')
    else:
        raise ValueError(f"Unsupported dataset: {name}")
  

    # Single-GPU case (original logic)
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
    )
    
    return test_loader


@torch.no_grad()
def ppl_eval(model, tokenizer, test_loader=None, dataset='wikitext2', model_seq_len=2048, batch_size=32, device="cuda"):
    model.eval()
    
    if test_loader is None:
        test_loader = get_test_data(dataset, tokenizer, seq_len=model_seq_len, batch_size=batch_size)
    nlls = []
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating PPL"):
            batch = batch.to(device)
            output = model(batch, use_cache=False)
            lm_logits = output.logits
            if torch.isfinite(lm_logits).all():
                shift_logits = lm_logits[:, :-1, :].contiguous()
                shift_labels = batch[:, 1:].contiguous()
                loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
                loss = loss_fct(shift_logits.reshape(-1, shift_logits.size(-1)), shift_labels.view(-1))
                nlls.append(loss)
        ppl = np.exp(torch.cat(nlls, dim=-1).mean().item())
    return ppl


batch_size = 2
model_seq_len = 2048

dataset_name = 'wikitext2'  
ppl_original = ppl_eval(original_model, tokenizer, dataset=dataset_name, model_seq_len=model_seq_len, batch_size=batch_size, device="cuda")


model = apply_compression_to_model(original_model, quant_mode='int')

small_state = torch.load(f"./compressed_llama_distilled_topk_wo_sink/small_attn_weights.pt", map_location="cpu")
base_state = model.state_dict()

for name, param in small_state.items():
    name = 'model.' + name
    if name in base_state and base_state[name].shape == param.shape:
        base_state[name] = param
        
model.load_state_dict(base_state)

lora_model = PeftModel.from_pretrained(model, adapter_model_path)
lora_model.eval()

ppl_compressed = ppl_eval(model, tokenizer, dataset=dataset_name, model_seq_len=model_seq_len, batch_size=batch_size, device="cuda")


print(f"Compressed Model PPL on {dataset_name}: {ppl_compressed}")
print(f"Original Model PPL on {dataset_name}: {ppl_original}")