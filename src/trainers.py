import logging
from collections import Counter
from transformers import AutoTokenizer
import torch
from datasets import Dataset, DatasetDict
import numpy as np
from torch.utils.data import DataLoader
from src.models.bert_bilstm_att import SequentialSentenceClassifier
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    f1_score, 
    precision_recall_fscore_support,
    confusion_matrix, 
    classification_report,
    ConfusionMatrixDisplay
)
import wandb
import os

save_dir = './saved_models'
os.makedirs(save_dir, exist_ok=True)  # Crée le dossier si besoin

def create_model_and_training_setup(dataset, model_params=None, batch_size=1, num_workers=4):
    if model_params is None:
        model_params = {
            'model_name': model_path,
            'num_classes': 2,
            'lstm_hidden_size': 256,
            'lstm_layers': 2,
            'dropout_rate': 0.3,
            'max_seq_length': 200,
            'max_sentences': 200
        }
    tokenizer = AutoTokenizer.from_pretrained(model_params['model_name'])

    def preprocess_batch(batch):
        max_seq_length = model_params['max_seq_length']
        max_sentences = model_params['max_sentences']
        input_ids_list, attention_mask_list, labels_list, project_ids = [], [], [], []
        for sentences, labels, project_id in zip(batch['sentences'], batch['labels'], batch['project_id']):
            encoded = tokenizer(
                sentences[:max_sentences],
                padding='max_length',
                truncation=True,
                max_length=max_seq_length,
                return_tensors='pt'
            )
            n_valid = len(sentences[:max_sentences])
            pad_len = max_sentences - n_valid
            if pad_len > 0:
                pad_input_ids = torch.zeros((pad_len, max_seq_length), dtype=torch.long)
                pad_attention_mask = torch.zeros((pad_len, max_seq_length), dtype=torch.long)
                input_ids = torch.cat([encoded['input_ids'], pad_input_ids], dim=0)
                attention_mask = torch.cat([encoded['attention_mask'], pad_attention_mask], dim=0)
                labels_pad = labels[:max_sentences] + [-100]*pad_len
            else:
                input_ids = encoded['input_ids'][:max_sentences]
                attention_mask = encoded['attention_mask'][:max_sentences]
                labels_pad = labels[:max_sentences]
            input_ids_list.append(input_ids)
            attention_mask_list.append(attention_mask)
            labels_list.append(torch.tensor(labels_pad, dtype=torch.long))
            project_ids.append(project_id)
        return {
            'input_ids': torch.stack(input_ids_list),
            'attention_mask': torch.stack(attention_mask_list),
            'labels': torch.stack(labels_list),
            'project_id': project_ids
        }
    dataset = dataset.map(
        preprocess_batch,
        batched=True,
        batch_size=16,
        remove_columns=['sentences', 'labels'],
        desc="Tokenizing and padding"
    )
    def collate_fn(batch):
        input_ids = torch.stack([torch.tensor(item['input_ids'], dtype=torch.long) for item in batch])
        attention_mask = torch.stack([torch.tensor(item['attention_mask'], dtype=torch.long) for item in batch])
        labels = torch.stack([torch.tensor(item['labels'], dtype=torch.long) for item in batch])
        project_ids = [item['project_id'] for item in batch]
        sentence_masks = (labels != -100).long()
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
            'sentence_masks': sentence_masks,
            'project_id': project_ids
        }
    train_loader = DataLoader(
        dataset['train'], batch_size=batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        dataset['val'], batch_size=batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        dataset['test'], batch_size=batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=num_workers, pin_memory=True
    )
    model = SequentialSentenceClassifier(**model_params)
    model = nn.DataParallel(model)
    bert_params = list(model.module.sentence_encoder.parameters())
    other_params = [p for n, p in model.module.named_parameters() if 'sentence_encoder' not in n]
    optimizer = torch.optim.AdamW([
        {'params': bert_params, 'lr': 2e-5},
        {'params': other_params, 'lr': 1e-3}
    ], weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, verbose=True
    )
    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    return model, train_loader, val_loader, test_loader, optimizer, scheduler, criterion

def training_loop(model, train_loader, val_loader, optimizer, scheduler, criterion, epochs=10, seed=46, device="cuda:0"):
    model.to(device)
    best_val_loss = float('inf')
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            sentence_masks = batch['sentence_masks'].to(device)
            labels = batch['labels'].to(device)
            logits, _ = model(input_ids, attention_mask, sentence_masks)
            logging.debug(f"logits.shape: {logits.shape}, labels.shape: {labels.shape}")
            logits_flat = logits.view(-1, logits.size(-1))
            labels_flat = labels.view(-1)
            logging.debug(f"logits_flat.shape: {logits_flat.shape}, labels_flat.shape: {labels_flat.shape}")
            loss = criterion(logits_flat, labels_flat)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                sentence_masks = batch['sentence_masks'].to(device)
                labels = batch['labels'].to(device)
                logits, _ = model(input_ids, attention_mask, sentence_masks)
                logits_flat = logits.view(-1, logits.size(-1))
                labels_flat = labels.view(-1)
                loss = criterion(logits_flat, labels_flat)
                val_loss += loss.item()
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        logging.info(f'Epoch {epoch+1}/{epochs}:')
        logging.info(f'  Train Loss: {avg_train_loss:.4f}')
        logging.info(f'  Val Loss: {avg_val_loss:.4f}')
        scheduler.step(avg_val_loss)
        wandb.log({
        "epoch": epoch + 1,
        "train_loss": avg_train_loss,
        "val_loss": avg_val_loss,
        })
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            save_path = os.path.join(save_dir, f'bert_bilstm_best_model_seed_{seed}.pth')
            torch.save(model.state_dict(), save_path)
            logging.info(f'  New best model saved!')

def calculate_metrics(logits, labels, masks):
    predictions = torch.argmax(logits, dim=-1)
    valid_mask = (labels != -100) & masks.bool()
    pred_flat = predictions[valid_mask].cpu().numpy()
    labels_flat = labels[valid_mask].cpu().numpy()
    if len(pred_flat) == 0:
        return {
            'accuracy': 0.0,
            'precision_macro': 0.0,
            'recall_macro': 0.0,
            'f1_macro': 0.0,
            'precision_weighted': 0.0,
            'recall_weighted': 0.0,
            'f1_weighted': 0.0,
            'confusion_matrix': np.array([])
        }
    accuracy = accuracy_score(labels_flat, pred_flat)
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        labels_flat, pred_flat, average='macro', zero_division=0
    )
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        labels_flat, pred_flat, average='weighted', zero_division=0
    )
    precision_per_class, recall_per_class, f1_per_class, support_per_class = precision_recall_fscore_support(
        labels_flat, pred_flat, average=None, zero_division=0
    )
    conf_matrix = confusion_matrix(labels_flat, pred_flat)
    return {
        'accuracy': accuracy,
        'precision_macro': precision_macro,
        'recall_macro': recall_macro,
        'f1_macro': f1_macro,
        'precision_weighted': precision_weighted,
        'recall_weighted': recall_weighted,
        'f1_weighted': f1_weighted,
        'precision_per_class': precision_per_class,
        'recall_per_class': recall_per_class,
        'f1_per_class': f1_per_class,
        'support_per_class': support_per_class,
        'confusion_matrix': conf_matrix
    }

def evaluate_model(model, data_loader, criterion, device, split_name="Test"):
    model.eval()
    total_loss = 0
    all_logits, all_labels, all_masks, all_project_ids = [], [], [], []
    logging.info(f"\n{'='*60}")
    logging.info(f"EVALUATION - {split_name.upper()}")
    logging.info(f"{'='*60}")
    with torch.no_grad():
        for batch_idx, batch in enumerate(data_loader):
            if batch_idx % 10 == 0:
                logging.info(f"Processing batch {batch_idx+1}/{len(data_loader)}")
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            sentence_masks = batch['sentence_masks'].to(device)
            labels = batch['labels'].to(device)
            project_ids = batch['project_id']
            logits, attention_weights = model(input_ids, attention_mask, sentence_masks)
            logits_flat = logits.view(-1, logits.size(-1))
            labels_flat = labels.view(-1)
            loss = criterion(logits_flat, labels_flat)
            total_loss += loss.item()
            all_logits.append(logits.cpu())
            all_labels.append(labels.cpu())
            all_masks.append(sentence_masks.cpu())
            all_project_ids.extend(project_ids)
    all_logits = torch.cat(all_logits, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    all_masks = torch.cat(all_masks, dim=0)
    avg_loss = total_loss / len(data_loader)
    metrics = calculate_metrics(all_logits, all_labels, all_masks)
    logging.info(f"\n{split_name} Results:")
    logging.info(f"Loss: {avg_loss:.4f}")
    logging.info(f"Accuracy: {metrics['accuracy']:.4f}")
    logging.info(f"F1-Score (Macro): {metrics['f1_macro']:.4f}")
    logging.info(f"F1-Score (Weighted): {metrics['f1_weighted']:.4f}")
    logging.info(f"Precision (Macro): {metrics['precision_macro']:.4f}")
    logging.info(f"Recall (Macro): {metrics['recall_macro']:.4f}")
    logging.info(f"\nPer-class metrics:")
    for i, (p, r, f1, s) in enumerate(zip(
            metrics['precision_per_class'], 
            metrics['recall_per_class'],
            metrics['f1_per_class'],
            metrics['support_per_class'])):
        logging.info(f"Class {i}: P={p:.3f}, R={r:.3f}, F1={f1:.3f}, Support={s:4d}")
    logging.info(f"\nConfusion Matrix:")
    logging.info(f"\n{metrics['confusion_matrix']}")
    return metrics, avg_loss, all_logits, all_labels, all_masks, all_project_ids

def test_model(model, test_loader, criterion, seed, device):
    logging.info(f"\n{'='*60}")
    logging.info("Test")
    logging.info(f"{'='*60}")
    if torch.cuda.is_available():
        model.load_state_dict(torch.load(f'./saved_models/bert_bilstm_best_model_seed_{seed}.pth'))
    else:
        model.load_state_dict(torch.load(f'./saved_models/bert_bilstm_best_model_seed_{seed}.pth', map_location='cpu'))
    test_metrics, test_loss, logits, labels, masks, project_ids = evaluate_model(
        model, test_loader, criterion, device, "Test"
    )
    analyze_predictions_by_project(logits, labels, masks, project_ids)
    analyze_prediction_errors(logits, labels, masks)
    return test_metrics, test_loss

def analyze_predictions_by_project(logits, labels, masks, project_ids, top_k=5):
    logging.info(f"\n{'='*60}")
    logging.info("ANALYSIS PER PROJECT")
    logging.info(f"{'='*60}")
    project_metrics = {}
    unique_projects = list(set(project_ids))
    for project_id in unique_projects:
        project_indices = [i for i, pid in enumerate(project_ids) if pid == project_id]
        if len(project_indices) == 0:
            continue
        proj_logits = logits[project_indices[0]:project_indices[0]+1]
        proj_labels = labels[project_indices[0]:project_indices[0]+1]
        proj_masks = masks[project_indices[0]:project_indices[0]+1]
        proj_metrics = calculate_metrics(proj_logits, proj_labels, proj_masks)
        project_metrics[project_id] = proj_metrics
    sorted_projects = sorted(project_metrics.items(), key=lambda x: x[1]['f1_macro'], reverse=True)
    logging.info(f"\nTop {top_k} projects (Best F1-score):")
    for i, (pid, metrics) in enumerate(sorted_projects[:top_k]):
        logging.info(f"{i+1}. Project {pid}: F1={metrics['f1_macro']:.3f}, Accuracy={metrics['accuracy']:.3f}")
    logging.info(f"\nThe {top_k} lowest performing projects (lowest F1-score):")
    for i, (pid, metrics) in enumerate(sorted_projects[-top_k:]):
        logging.info(f"{i+1}. Project {pid}: F1={metrics['f1_macro']:.3f}, Accuracy={metrics['accuracy']:.3f}")

def analyze_prediction_errors(logits, labels, masks):
    logging.info(f"\n{'='*60}")
    logging.info("ERROR ANALYSIS")
    logging.info(f"{'='*60}")
    predictions = torch.argmax(logits, dim=-1)
    valid_mask = (labels != -100) & masks.bool()
    pred_flat = predictions[valid_mask].cpu().numpy()
    labels_flat = labels[valid_mask].cpu().numpy()
    n_classes = max(int(pred_flat.max()), int(labels_flat.max())) + 1 if len(pred_flat) else 4
    prediction_errors = Counter()
    label_errors = Counter()
    for pred, true in zip(pred_flat, labels_flat):
        if pred != true:
            prediction_errors[pred] += 1
            label_errors[true] += 1
    logging.info("Errors per predicted classes (false positive):")
    for class_idx in range(n_classes):
        logging.info(f"Class {class_idx}: {prediction_errors[class_idx]} errors")
    logging.info("\nErrors per true classes (false negative):")
    for class_idx in range(n_classes):
        logging.info(f"Class {class_idx}: {label_errors[class_idx]} errors")
    error_pairs = Counter()
    for pred, true in zip(pred_flat, labels_flat):
        if pred != true:
            error_pairs[(true, pred)] += 1
    logging.info(f"\nMost frequent error pairs (true -> predicted):")
    for (true_class, pred_class), count in error_pairs.most_common(10):
        logging.info(f"Class {true_class} -> Class {pred_class}: {count}")

def run(dataset, model_params=None, epochs=10, batch_size=3, seed=46, device="cuda:0"):
    logging.info("Pipeline initialization completed...")
    model, train_loader, val_loader, test_loader, optimizer, scheduler, criterion = create_model_and_training_setup(
        dataset, model_params, batch_size=batch_size
    )
    logging.info("\n" + "="*60)
    logging.info("Training")
    logging.info("="*60)
    training_loop(model, train_loader, val_loader, optimizer, scheduler, criterion, epochs, seed, device)
    test_metrics, test_loss = test_model(model, test_loader, criterion, seed, device)
    logging.info(f"\n{'='*60}")
    logging.info("Results on Testset")
    logging.info(f"{'='*60}")
    logging.info(f"Test Loss: {test_loss:.4f}")
    logging.info(f"Test Accuracy: {test_metrics['accuracy']:.4f}")
    logging.info(f"Test F1-Score (Macro): {test_metrics['f1_macro']:.4f}")
    logging.info(f"Test F1-Score (Weighted): {test_metrics['f1_weighted']:.4f}")
    return test_metrics
