import os
import sys
import time
import logging
import argparse
import torch
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from autoexpl.tools import fileio
from autoexpl.xqb import utils, gen_prompt

# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(
    stream=sys.stdout,
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s [%(process)d] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

def init_distributed():
    """Initialize distributed training environment"""
    if not dist.is_initialized():
        dist.init_process_group(
            backend="nccl",
            init_method="env://"
        )
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    logger.info(f"Initialized process group on rank {local_rank}")
    return local_rank

class PaddedDataset(Dataset):
    def __init__(self, data, tokenizer, max_seq_len):
        self.data = []
        for item in data:
            encoded = tokenizer(
                item["prompt"],
                max_length=max_seq_len,
                truncation=True,
                padding="max_length",
                return_tensors="pt"
            )
            self.data.append({
                **item,
                "input_ids": encoded["input_ids"].squeeze(),
                "attention_mask": encoded["attention_mask"].squeeze()
            })
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

def collate_fn(batch):
    """Collate function with fixed padding"""
    return {
        "input_ids": torch.stack([item["input_ids"] for item in batch]),
        "attention_mask": torch.stack([item["attention_mask"] for item in batch]),
        "exp_ids": [item["exp_id"] for item in batch],
        "indices": [item["index"] for item in batch],
        "question_idxs": [item["question_idx"] for item in batch]
    }

def generate_batch(model, batch, generation_config):
    """Generate text with stable tensor shapes"""
    try:
        with torch.no_grad():
            outputs = model.generate(
                input_ids=batch["input_ids"].to(model.device),
                attention_mask=batch["attention_mask"].to(model.device),
                **generation_config
            )
        return outputs.cpu()
    except RuntimeError as e:
        logger.error(f"Generation error: {str(e)}")
        raise

def main(args):
    local_rank = init_distributed()
    world_size = dist.get_world_size()

    # Load model and tokenizer with proper configuration
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_id,
        padding_side="left",
        truncation_side="left"
    )
    tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=torch.bfloat16,
        device_map=f"cuda:{local_rank}",
        attn_implementation="eager"  # For better compatibility
    )

    # Prepare dataset with consistent padding
    qanta_json = fileio.load_singlefile_w_prefix(args.input_file)
    dataset = prepare_dataset(qanta_json, args, tokenizer)
    
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=local_rank,
        shuffle=False
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.max_batch_size,
        sampler=sampler,
        collate_fn=collate_fn
    )

    # Configure generation parameters
    generation_config = {
        "max_new_tokens": args.max_seq_len,
        "do_sample": True,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "pad_token_id": tokenizer.eos_token_id
    }

    # Processing loop
    for batch in tqdm(dataloader, desc=f"Rank {local_rank}"):
        try:
            outputs = generate_batch(model, batch, generation_config)
            process_outputs(outputs, batch, tokenizer)
            # Synchronize after each batch
            dist.barrier()
        except Exception as e:
            logger.error(f"Rank {local_rank} failed batch: {str(e)}")
            break

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default="meta-llama/Llama-3.1-8B")
    parser.add_argument("--max-seq-len", type=int, default=650)
    parser.add_argument("--max-batch-size", type=int, default=2)  # Reduced for stability
    parser.add_argument("--input-file", required=True)
    parser.add_argument("--output-file", required=True)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.95)
    args = parser.parse_args()
    
    main(args)
