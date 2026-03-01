import os
# Keep your specific GPU setting
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from datasets import Dataset
from transformers import AutoTokenizer, AutoModel 
import xgboost as xgb
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import json
from tqdm import tqdm

# --- CONFIGURATION ---
model_ckpt = "distilbert-base-uncased"
batch_size = 32  # Batch size for embedding extraction
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

savename = model_ckpt.split("/")[0] if "/" in model_ckpt else model_ckpt
print(f"Model: {model_ckpt} | Device: {device} | Save Name: {savename}")

# --- DATA LOADING ---
# Assuming 'preprocess.py' exists as per your original code
from preprocess import process
df_train, df_test = process(DoS_only=True)
print(f"Train Shape: {df_train.shape}, Test Shape: {df_test.shape}")

# --- 1. UNIVERSAL TEXTUALIZATION ---
def serialize_row_full(r):
    # This function is crucial: it bridges the gap between NSL-KDD and UNSW
    # by converting specific columns into universal networking concepts.
    prompt = (
        f"Protocol:{r['protocol_type']},Service:{r['service']},Flag:{r['flag']},"
        f"Duration:{r['duration']},SrcBytes:{r['src_bytes']},DstBytes:{r['dst_bytes']},"
        f"IsLandAttack:{r['land']},WrongFrag:{r['wrong_fragment']},UrgentPkts:{r['urgent']},"
        f"HotIndices:{r['hot']},FailedLogins:{r['num_failed_logins']},LoggedIn:{r['logged_in']},"
        f"Compromised:{r['num_compromised']},RootShell:{r['root_shell']},SuAttempted:{r['su_attempted']},"
        f"NumRoot:{r['num_root']},FileCreations:{r['num_file_creations']},NumShells:{r['num_shells']},"
        f"AccessFiles:{r['num_access_files']},OutboundCmds:{r['num_outbound_cmds']},IsHostLogin:{r['is_host_login']},"
        f"IsGuestLogin:{r['is_guest_login']},SameHostCount:{r['count']},SameSrvCount:{r['srv_count']},"
        f"SerrorRate:{r['serror_rate']:.2f},SrvSerrorRate:{r['srv_serror_rate']:.2f},RerrorRate:{r['rerror_rate']:.2f},"
        f"SrvRerrorRate:{r['srv_rerror_rate']:.2f},SameSrvRate:{r['same_srv_rate']:.2f},DiffSrvRate:{r['diff_srv_rate']:.2f},"
        f"SrvDiffHostRate:{r['srv_diff_host_rate']:.2f},DstHostCount:{r['dst_host_count']},DstHostSrvCount:{r['dst_host_srv_count']},"
        f"DstHostSameSrvRate:{r['dst_host_same_srv_rate']:.2f},DstHostDiffSrvRate:{r['dst_host_diff_srv_rate']:.2f},"
        f"DstHostSameSrcPortRate:{r['dst_host_same_src_port_rate']:.2f},DstHostSrvDiffHostRate:{r['dst_host_srv_diff_host_rate']:.2f},"
        f"DstHostSerrorRate:{r['dst_host_serror_rate']:.2f},DstHostSrvSerrorRate:{r['dst_host_srv_serror_rate']:.2f},"
        f"DstHostRerrorRate:{r['dst_host_rerror_rate']:.2f},DstHostSrvRerrorRate:{r['dst_host_srv_rerror_rate']:.2f}"
    )
    return prompt

print("Serializing full dataset...")
df_train['text'] = df_train.apply(serialize_row_full, axis=1)
df_test['text'] = df_test.apply(serialize_row_full, axis=1)

# --- 2. TOKENIZATION & SETUP ---
tokenizer = AutoTokenizer.from_pretrained(model_ckpt, use_fast=True)

# Ensure pad token exists
if tokenizer.pad_token is None:
    if "<pad>" in tokenizer.get_vocab():
        tokenizer.pad_token = "<pad>"
    else:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})

def tokenize_function(examples):
    return tokenizer(
        examples["text"], 
        padding="max_length", 
        truncation=True, 
        max_length=384,
        return_tensors="pt"
    )

# Prepare Datasets
train_dataset = Dataset.from_pandas(df_train[['text', 'label']])
test_dataset = Dataset.from_pandas(df_test[['text', 'label']])

print("Tokenizing...")
tokenized_train = train_dataset.map(tokenize_function, batched=True)
tokenized_test = test_dataset.map(tokenize_function, batched=True)

# Convert to PyTorch format for the extraction loop
tokenized_train.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
tokenized_test.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

# --- 3. DYNAMIC ADAPTER (Feature Extraction) ---
# Use AutoModel (Outputs Embeddings) instead of AutoModelForSequenceClassification
model = AutoModel.from_pretrained(model_ckpt)
model.resize_token_embeddings(len(tokenizer))
model.to(device)
model.eval() # Set to evaluation mode (freeze dropout, etc)

def extract_embeddings(dataset):
    """
    Feeds text through the LLM and extracts the [CLS] token embedding.
    This creates the 'dynamic' features for XGBoost.
    """
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    embeddings = []
    labels = []
    
    print(f"Extracting embeddings using {model_ckpt}...")
    with torch.no_grad(): # Disable gradients to save memory
        for batch in tqdm(loader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            
            # Extract [CLS] token (first token) hidden state
            # Shape: [batch_size, hidden_size] (e.g., 32, 768)
            cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            
            embeddings.append(cls_embeddings)
            labels.append(batch['label'].numpy())
            
    return np.vstack(embeddings), np.concatenate(labels)

# Generate the new Feature Sets
X_train, y_train = extract_embeddings(tokenized_train)
X_test, y_test = extract_embeddings(tokenized_test)

print(f"Training Features Shape: {X_train.shape}")
print(f"Testing Features Shape: {X_test.shape}")

# --- 4. THE CLASSIFIER (XGBoost) ---
# This replaces the WeightedTrainer. XGBoost is robust and handles 
# the dense embeddings well.

print("Training XGBoost Classifier...")
clf = xgb.XGBClassifier(
    objective='binary:logistic',
    n_estimators=100,
    learning_rate=0.1,
    max_depth=6,
    use_label_encoder=False,
    eval_metric='logloss',
    tree_method='hist', # Faster on large datasets
    device="cuda" if torch.cuda.is_available() else "cpu"
)

clf.fit(X_train, y_train)

# --- 5. EVALUATION ---
print("Predicting on Test Set...")
y_preds = clf.predict(X_test)

# Metrics
acc = accuracy_score(y_test, y_preds)
precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_preds, average='binary')

metrics = {
    'accuracy': acc,
    'f1': f1,
    'precision': precision,
    'recall': recall
}

# --- SAVE RESULTS ---
directory_path = f"./{savename}_dynamic"
os.makedirs(directory_path, exist_ok=True)

print("Evaluation Metrics:", metrics)

with open(f"{directory_path}/evaluation_results.json", "w") as f:
    json.dump(metrics, f, indent=4)

# Confusion Matrix
cm = confusion_matrix(y_test, y_preds)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=["NORMAL", "ATTACK"], 
            yticklabels=["NORMAL", "ATTACK"])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title(f'Confusion Matrix (LLM-Adapter + XGBoost)')
plt.savefig(f"{directory_path}/confusion_matrix.png")

# Text Report
report = classification_report(y_test, y_preds, target_names=["NORMAL", "ATTACK"])
with open(f"{directory_path}/classification_report.txt", "w") as f:
    f.write(report)
    
print(f"All results saved to {directory_path}")




# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "3"

# import pandas as pd
# import numpy as np
# import torch
# from datasets import Dataset
# from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
# from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# import json
# from sklearn.metrics import confusion_matrix
# import matplotlib.pyplot as plt
# from sklearn.metrics import classification_report 
# import seaborn as sns

# import torch
# import numpy as np
# from sklearn.utils.class_weight import compute_class_weight

# from torch import nn
# from transformers import Trainer

# # model_ckpt = "Qwen/Qwen2.5-0.5B"
# # model_ckpt = "h2oai/h2o-danube3-500m-base"
# # model_ckpt = "Yi3852/nanogpt-270M"
# # model_ckpt = "answerdotai/ModernBERT-base"
# model_ckpt = "distilbert-base-uncased"
# # model_ckpt = "meta-llama/Llama-4-Scout-17B-16E-Instruct"
# # model_ckpt = "mistralai/Mistral-7B-v0.1"

# provide_token = True


# savename = model_ckpt.split("/")[0]
# print(savename)
# # if os.path.exists("./{savename}/", )

# from preprocess import process
# df_train, df_test = process(DoS_only=True)
# print(df_train.shape, df_test.shape)


# def serialize_row_full(r):
#     # Group 1: Basic Header (Protocol & Address)
#     # Describes the connection basics.
#     prompt = (
#         f"Protocol:{r['protocol_type']},Service:{r['service']},Flag:{r['flag']},"
#         f"Duration:{r['duration']},SrcBytes:{r['src_bytes']},DstBytes:{r['dst_bytes']},"
#         f"IsLandAttack:{r['land']},WrongFrag:{r['wrong_fragment']},UrgentPkts:{r['urgent']},"
#         f"HotIndices:{r['hot']},FailedLogins:{r['num_failed_logins']},LoggedIn:{r['logged_in']},"
#         f"Compromised:{r['num_compromised']},RootShell:{r['root_shell']},SuAttempted:{r['su_attempted']},"
#         f"NumRoot:{r['num_root']},FileCreations:{r['num_file_creations']},NumShells:{r['num_shells']},"
#         f"AccessFiles:{r['num_access_files']},OutboundCmds:{r['num_outbound_cmds']},IsHostLogin:{r['is_host_login']},"
#         f"IsGuestLogin:{r['is_guest_login']},SameHostCount:{r['count']},SameSrvCount:{r['srv_count']},"
#         f"SerrorRate:{r['serror_rate']:.2f},SrvSerrorRate:{r['srv_serror_rate']:.2f},RerrorRate:{r['rerror_rate']:.2f},"
#         f"SrvRerrorRate:{r['srv_rerror_rate']:.2f},SameSrvRate:{r['same_srv_rate']:.2f},DiffSrvRate:{r['diff_srv_rate']:.2f},"
#         f"SrvDiffHostRate:{r['srv_diff_host_rate']:.2f},DstHostCount:{r['dst_host_count']},DstHostSrvCount:{r['dst_host_srv_count']},"
#         f"DstHostSameSrvRate:{r['dst_host_same_srv_rate']:.2f},DstHostDiffSrvRate:{r['dst_host_diff_srv_rate']:.2f},"
#         f"DstHostSameSrcPortRate:{r['dst_host_same_src_port_rate']:.2f},DstHostSrvDiffHostRate:{r['dst_host_srv_diff_host_rate']:.2f},"
#         f"DstHostSerrorRate:{r['dst_host_serror_rate']:.2f},DstHostSrvSerrorRate:{r['dst_host_srv_serror_rate']:.2f},"
#         f"DstHostRerrorRate:{r['dst_host_rerror_rate']:.2f},DstHostSrvRerrorRate:{r['dst_host_srv_rerror_rate']:.2f}"
#     )
#     return prompt


# # Apply to DataFrame
# print("Serializing full dataset...")
# df_train['text'] = df_train.apply(serialize_row_full, axis=1)
# df_test['text'] = df_test.apply(serialize_row_full, axis=1)

# train_dataset = Dataset.from_pandas(df_train[['text', 'label']])
# test_dataset = Dataset.from_pandas(df_test[['text', 'label']])

# train_dataset = train_dataset.shuffle(seed=42)

# tokenizer = AutoTokenizer.from_pretrained(model_ckpt, use_fast=True)

# # if provide_token:
# #     tokenizer.pad_token = tokenizer.eos_token
# #     tokenizer.pad_token_id = tokenizer.eos_token_id

# if tokenizer.pad_token is None:
#     if "<pad>" in tokenizer.get_vocab():
#         tokenizer.pad_token = "<pad>"
#     else:
#         tokenizer.add_special_tokens({'pad_token': '[PAD]'})

#     provide_token=True


# print(f"Using pad token: {tokenizer.pad_token} (ID: {tokenizer.pad_token_id})")


# def tokenize_function(examples):
#     return tokenizer(
#         examples["text"], 
#         padding="max_length", 
#         truncation=True, 
#         max_length=384 
#     )

# # Re-run the map function
# tokenized_train = train_dataset.map(tokenize_function, batched=True)
# tokenized_test = test_dataset.map(tokenize_function, batched=True)



# class WeightedTrainer(Trainer):
#     def __init__(self, class_weights, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.class_weights = class_weights
    
#     def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
#         labels = inputs.get("labels")
        
#         outputs = model(**inputs)
#         logits = outputs.get("logits")
        
#         if self.class_weights.device != model.device:
#             self.class_weights = self.class_weights.to(model.device)
        
#         target_dtype = torch.float16
#         weights = self.class_weights.to(target_dtype)

#         loss_fct = nn.CrossEntropyLoss(weight=weights)
        
#         loss = loss_fct(
#             logits.view(-1, self.model.config.num_labels).to(target_dtype), 
#             labels.view(-1).to(torch.long)
#         )
        
#         return (loss, outputs) if return_outputs else loss
    


# # Assuming 'tokenized_train' has a 'label' column. 
# # If your dataset is a torch Dataset or different format, adjust accordingly.
# train_labels = df_train["label"].values

# # Compute weights
# class_weights = compute_class_weight(
#     class_weight='balanced', 
#     classes=np.unique(train_labels), 
#     y=train_labels
# )

# # Convert to tensor and move to correct device type (float32 is standard for weights)
# class_weights = torch.tensor(class_weights, dtype=torch.float32)

# print(f"Class Weights: {class_weights}")
# # Expected: [weight_for_NORMAL, weight_for_ATTACK]


# model = AutoModelForSequenceClassification.from_pretrained(
#     model_ckpt, 
#     num_labels=2,
#     id2label={0: "NORMAL", 1: "ATTACK"},
#     label2id={"NORMAL": 0, "ATTACK": 1},
#     device_map='auto'
# )

# if provide_token:
#     model.config.pad_token_id = tokenizer.pad_token_id
#     model.resize_token_embeddings(len(tokenizer))

# # Metric calculation function
# def compute_metrics(pred):
#     labels = pred.label_ids
#     preds = pred.predictions.argmax(-1)
#     precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
#     acc = accuracy_score(labels, preds)
#     return {
#         'accuracy': acc,
#         'f1': f1,
#         'precision': precision,
#         'recall': recall
#     }



# # Training Arguments
# training_args = TrainingArguments(
#     output_dir=f"./{model_ckpt}",
#     num_train_epochs=5,              # Low epochs usually sufficient for this
#     per_device_train_batch_size=16,   # Increase until OOM
#     gradient_accumulation_steps=2,    # Simulates batch size 32
#     per_device_eval_batch_size=32,
#     learning_rate=2e-5,               # LoRA allows higher LR
#     weight_decay=0.01,
#     fp16=True,                        # ESSENTIAL for V100 Speed
#     logging_steps=10,
#     optim="paged_adamw_8bit",         # Saves VRAM
#     group_by_length=True,             # Faster training by grouping sequences
#     report_to="none"

# )


# trainer = WeightedTrainer(
#     model=model,
#     args=training_args,
#     train_dataset=tokenized_train,
#     eval_dataset=tokenized_test,
#     compute_metrics=compute_metrics,
#     class_weights=class_weights
# )

# print("Starting training...")
# trainer.train()




# # print("Running prediction on Test Set...")
# prediction_output = trainer.predict(tokenized_test)

# # --- SAVE METRICS ---
# metrics = prediction_output.metrics

# # Print them to console
# directory_path = f"./{savename}"
# os.makedirs(directory_path, exist_ok=True)


# print("Evaluation Metrics:", metrics)

# # Save to a JSON file
# with open(f"./{savename}/{savename}_evaluation_results.json", "w") as f:
#     json.dump(metrics, f, indent=4)
# print("Metrics saved to 'evaluation_results.json'.")


# # --- GENERATE & SAVE CONFUSION MATRIX ---
# raw_preds = prediction_output.predictions
# y_preds = np.argmax(raw_preds, axis=-1)
# y_true = prediction_output.label_ids

# # Create the Confusion Matrix
# cm = confusion_matrix(y_true, y_preds)

# # Plotting
# plt.figure(figsize=(8, 6))
# # Use the label names from your id2label config for clarity
# class_names = [model.config.id2label[i] for i in range(model.config.num_labels)]

# sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
#             xticklabels=class_names, 
#             yticklabels=class_names)

# plt.xlabel('Predicted Label')
# plt.ylabel('True Label')
# plt.title('Confusion Matrix')

# # Save the plot
# plt.savefig(f"./{savename}/{savename}_confusion_matrix.png")
# print("Confusion Matrix saved to 'confusion_matrix.png'.")

# # Optional: Save a text report of classification details
# report = classification_report(y_true, y_preds, target_names=class_names)
# with open(f"./{savename}/{savename}_classification_report.txt", "w") as f:
#     f.write(report)

