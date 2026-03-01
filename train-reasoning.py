import os
# Set device first
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

import pandas as pd
import numpy as np
import torch
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report
from sklearn.utils.class_weight import compute_class_weight
from torch import nn
from datasets import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType

# --- CONFIGURATION ---
# NOTE: For "very fast" execution on tabular data, 7B is overkill. 
# Consider uncommenting Qwen2.5-0.5B for 10x speedup with likely similar accuracy.

# model_ckpt = "Qwen/Qwen2.5-0.5B"
# model_ckpt = "h2oai/h2o-danube3-500m-base"
# model_ckpt = "Yi3852/nanogpt-270M"
# model_ckpt = "answerdotai/ModernBERT-base"
# model_ckpt = "meta-llama/Llama-4-Scout-17B-16E-Instruct"

# model_ckpt = "distilbert-base-uncased"

# model_ckpt = "mistralai/Mistral-7B-v0.1"
# model_ckpt = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B" 
model_ckpt = "google/gemma-3-27b-it"
# model_ckpt = "microsoft/Phi-3-medium-4k-instruct"

savename = model_ckpt.split("/")[-1]
print(f"Training Model: {savename}")

# --- DATA PREPROCESSING ---
# Assuming process() is available from your local preprocess.py
from preprocess import process
# Load data (ensure this returns pandas DataFrames)
df_train, df_test = process(DoS_only=True)

# OPTIMIZATION 1: Compact Serialization
# Remove filler words to reduce token count (Seq Len < 128 is much faster than 256)
def serialize_row_compact(r):
    # Grouping by feature type to help the model maintain context
    prompt = (
        f"p:{r['protocol_type']},s:{r['service']},f:{r['flag']},dur:{r['duration']},sb:{r['src_bytes']},db:{r['dst_bytes']},"
        f"l:{r['land']},wf:{r['wrong_fragment']},u:{r['urgent']},hot:{r['hot']},nf:{r['num_failed_logins']},"
        f"li:{r['logged_in']},nc:{r['num_compromised']},rs:{r['root_shell']},su:{r['su_attempted']},nr:{r['num_root']},"
        f"nfc:{r['num_file_creations']},ns:{r['num_shells']},naf:{r['num_access_files']},noc:{r['num_outbound_cmds']},"
        f"hl:{r['is_host_login']},gl:{r['is_guest_login']},cnt:{r['count']},scnt:{r['srv_count']},se:{r['serror_rate']:.2f},"
        f"sse:{r['srv_serror_rate']:.2f},re:{r['rerror_rate']:.2f},sre:{r['srv_rerror_rate']:.2f},ssr:{r['same_srv_rate']:.2f},"
        f"dsr:{r['diff_srv_rate']:.2f},sdh:{r['srv_diff_host_rate']:.2f},dhc:{r['dst_host_count']},dhsc:{r['dst_host_srv_count']},"
        f"dhssr:{r['dst_host_same_srv_rate']:.2f},dhdsr:{r['dst_host_diff_srv_rate']:.2f},dhssp:{r['dst_host_same_src_port_rate']:.2f},"
        f"dhsdh:{r['dst_host_srv_diff_host_rate']:.2f},dhse:{r['dst_host_serror_rate']:.2f},dhsse:{r['dst_host_srv_serror_rate']:.2f},"
        f"dhre:{r['dst_host_rerror_rate']:.2f},dhsre:{r['dst_host_srv_rerror_rate']:.2f}"
    )

    return prompt


print("Serializing dataset (Compact Mode)...")
df_train['text'] = df_train.apply(serialize_row_compact, axis=1)
df_test['text'] = df_test.apply(serialize_row_compact, axis=1)

# Create HF Datasets
train_dataset = Dataset.from_pandas(df_train[['text', 'label']])
test_dataset = Dataset.from_pandas(df_test[['text', 'label']])
train_dataset = train_dataset.shuffle(seed=42)

# --- TOKENIZER ---
tokenizer = AutoTokenizer.from_pretrained(model_ckpt, use_fast=True) # Try fast tokenizer
tokenizer.pad_token = tokenizer.eos_token # Qwen/DeepSeek usually use EOS as PAD

# OPTIMIZATION 2: No Padding here, use DataCollator later
def tokenize_function(examples):
    return tokenizer(
        examples["text"], 
        truncation=True, 
        max_length=256 # Cap max length just in case
    )

tokenized_train = train_dataset.map(tokenize_function, batched=True)
tokenized_test = test_dataset.map(tokenize_function, batched=True)

# Remove text columns to save memory
tokenized_train = tokenized_train.remove_columns(["text"])
tokenized_test = tokenized_test.remove_columns(["text"])

# --- CLASS WEIGHTS ---
# Calculate weights for imbalanced data
train_labels = df_train["label"].values
class_weights = compute_class_weight(
    class_weight='balanced', 
    classes=np.unique(train_labels), 
    y=train_labels
)
class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32)
print(f"Class Weights: {class_weights_tensor}")

# --- MODEL SETUP ---
# OPTIMIZATION 3: 8-bit Quantization (NF4) for VRAM efficiency
bnb_config = BitsAndBytesConfig(
    # load_in_8bit=True,
    # llm_int8_threshold=0.0 
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16 # V100 likes float16
)

model = AutoModelForSequenceClassification.from_pretrained(
    model_ckpt, 
    num_labels=2,
    quantization_config=bnb_config,
    torch_dtype=torch.float16,
    device_map='auto',
    low_cpu_mem_usage=True
)

# Enable Gradient Checkpointing (Saves VRAM, allows larger batches)
model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)

# LoRA Config
lora_config = LoraConfig(
    r=16, # Slightly higher rank for better convergence
    lora_alpha=32,
    target_modules=[
        "q_proj", 
        "k_proj", 
        "v_proj", 
        "o_proj", 
        "gate_proj", 
        "up_proj", 
        "down_proj"
    ], # Target only attention heads for speed
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.SEQ_CLS
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
model.config.pad_token_id = tokenizer.pad_token_id

# --- TRAINER SETUP ---

class WeightedTrainer(Trainer):
    def __init__(self, class_weights, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Ensure weights are on the same device as the model
        self.class_weights = class_weights
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        
        # Move weights to device if needed
        if self.class_weights.device != logits.device:
            self.class_weights = self.class_weights.to(logits.device)
            
        loss_fct = nn.CrossEntropyLoss(weight=self.class_weights)
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        
        return (loss, outputs) if return_outputs else loss

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    return {'accuracy': acc, 'f1': f1, 'precision': precision, 'recall': recall}

# OPTIMIZATION 4: Training Arguments
training_args = TrainingArguments(
    output_dir=f"./{savename}",
    num_train_epochs=1,               # 1 Epoch is usually enough for 7B on simple tasks
    per_device_train_batch_size=8,   # Increase until OOM
    gradient_accumulation_steps=2,    # Simulates batch size 32
    per_device_eval_batch_size=32,
    learning_rate=2e-4,               # LoRA allows higher LR
    weight_decay=0.01,
    fp16=True,                        # ESSENTIAL for V100 Speed
    logging_steps=10,
    optim="paged_adamw_8bit",         # Saves VRAM
    group_by_length=True,             # Faster training by grouping sequences
    report_to="none"
)

# OPTIMIZATION 5: Dynamic Padding
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

trainer = WeightedTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_test,
    compute_metrics=compute_metrics,
    class_weights=class_weights_tensor,
    data_collator=data_collator
)

print("Starting training...")
trainer.train()

# --- EVALUATION ---
print("Running prediction...")
prediction_output = trainer.predict(tokenized_test)
metrics = prediction_output.metrics

# Save Metrics
os.makedirs(f"./{savename}", exist_ok=True)
with open(f"./{savename}/results.json", "w") as f:
    json.dump(metrics, f, indent=4)

# Confusion Matrix
y_preds = np.argmax(prediction_output.predictions, axis=-1)
y_true = prediction_output.label_ids
cm = confusion_matrix(y_true, y_preds)

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.savefig(f"./{savename}/confusion_matrix.png")
print("Done.")