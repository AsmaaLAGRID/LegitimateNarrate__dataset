# === main.py ===

import argparse
import os
import logging
import numpy as np
import torch

from src.trainers import run
from src.data import get_dataset 

os.environ["WANDB_MODE"] = "offline"
import wandb

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:50'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

MODEL_BERT = 'bert-large-uncased'

dsdir = os.getenv('DSDIR')
if dsdir is None:
    raise EnvironmentError("I can't get access to DSDIR var")

root_path = os.path.join(dsdir, 'HuggingFace_Models')
path_bert = os.path.join(root_path, MODEL_BERT)

if not os.path.isdir(path_bert):
    raise FileNotFoundError(f"BERT model not found : {path_bert!r}")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="hsln_bilstm", type=str,
                        choices=["hsln_bilstm_4", "hsln_bilstm", "seq_crf", "seq", "gru_seq", "llama_seq"])
    parser.add_argument("--model_name", type=str, default=path_bert)
    parser.add_argument("--num_classes", type=int, default=2)
    parser.add_argument("--text_col", type=str, default="sentences")
    parser.add_argument("--label_col", type=str, default="lebels")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--seed", type=int, default=46)
    args = parser.parse_args()
    return args

def main(args, dataset):
    out_dir = os.path.join("./results", args.model)
    os.makedirs(out_dir, exist_ok=True)
    results_file = os.path.join(out_dir, "_results.txt")

    log_file = os.path.join(out_dir, "main.log")
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        filemode="a"
    )
    logger = logging.getLogger(__name__)
    logger.info("=== Start Experiment ===")

    model_config = {
        'model_name': args.model_name,
        'num_classes': args.num_classes,  
        'lstm_hidden_size': 256,
        'lstm_layers': 2,
        'dropout_rate': 0.3,
        'max_seq_length': 200,
        'max_sentences': 200
    }
    
    logger.info("Architecture for sequential sentence classification")
    logger.info("=" * 55)
    logger.info(f"Backbone model: {model_config['model_name']}")
    logger.info(f"Classes: {model_config['num_classes']} (multi-classe)")
    logger.info(f"LSTM hidden size: {model_config['lstm_hidden_size']}")
    logger.info(f"Max sentence length: {model_config['max_seq_length']} tokens")
    logger.info(f"Max sentences per project: {model_config['max_sentences']}")
    logger.info("\nComponent:")
    logger.info("1. Encoder of sentences (BERT)")
    logger.info("2. Bidirectionnel LSTM ")
    logger.info("3. Attention Mechanism")
    logger.info("4. Classifier and dropout")
    logger.info("5. CrossEntropyLoss for classification")
    logger.info("6. Metrics: Accuracy, F1-score, Precision, Recall")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    results = {}
    seeds = [46, 47, 48, 49, 50]
    for seed in seeds:
        args.seed = seed
        logger.info(f"--- Run for seed = {seed} ---")

        wandb.init(
            project="seq_sent_cls",
            name=f"{args.model}_seed_{seed}",
            config=model_config,
            dir=out_dir,
            reinit=True
        )

        test_metrics = run(dataset, model_config, epochs=args.epochs, batch_size=args.batch_size, seed=seed, device=device)

        logger.info(f"Results for seed {seed} : {test_metrics}")
        results[seed] = test_metrics

        # Sauvegarde matrice de confusion
        conf_matrix = test_metrics.get("confusion_matrix", None)
        if conf_matrix is not None:
            conf_matrix_path = os.path.join(out_dir, f"cm_seed_{seed}.png")
            save_confusion_matrix(conf_matrix, conf_matrix_path)

        wandb.finish()

    logger.info("=== End of runs ===")
    save_mean_std_results(results, results_file)

    return results

def save_mean_std_results(results, result_file):
    test_f1, test_prec, test_rec = [], [], []
    for seed, metrics in results.items():
        test_f1.append(metrics.get("f1_macro", None))
        test_prec.append(metrics.get("precision_macro", None))
        test_rec.append(metrics.get("recall_macro", None))

    test_f1 = [v for v in test_f1 if v is not None]
    test_prec = [v for v in test_prec if v is not None]
    test_rec = [v for v in test_rec if v is not None]

    with open(result_file, 'a', encoding="utf-8") as f:
        f.write('test f1:' + str(test_f1) + '\n')
        f.write('test prec:' + str(test_prec) + '\n')
        f.write('test rec:' + str(test_rec) + '\n')

        if test_prec:
            f.write('mean test precision:' + str(np.mean(test_prec)) + '\n')
            f.write('std test precision:' + str(np.std(test_prec, ddof=1)) + '\n')
        if test_rec:
            f.write('mean test recall:' + str(np.mean(test_rec)) + '\n')
            f.write('std test recall:' + str(np.std(test_rec, ddof=1)) + '\n')
        if test_f1:
            f.write('mean test f1:' + str(np.mean(test_f1)) + '\n')
            f.write('std test f1:' + str(np.std(test_f1, ddof=1)) + '\n')
        f.write('\n\n')

def save_confusion_matrix(conf_matrix, out_path, class_names=None):
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def save_confusion_matrix(cm, out_path, title=" Test Confusion Matrix"):
    num_classes = cm.shape[0]
    fig_size = max(6, int(num_classes * 0.6))
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap='Blues', ax=ax, colorbar=False)
    ax.set_title(title)
    plt.tight_layout()

    fig.savefig(out_path, dpi=150)
    plt.close(fig)

if __name__ == "__main__":
    args = parse_args()
    dataset = get_dataset(text_col=args.text_col, label_col=args.label_col) 
    main(args, dataset)
