import argparse
import os
import torch
import pandas as pd
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
from sklearn.model_selection import train_test_split
from src.hubert_classifier_new import (
    FTConfig,
    AudioCollator,
    train_finetune,
    load_for_inference,
)
from src.dataset import KoreanStopDataset

def parse_args():
    parser = argparse.ArgumentParser()

    # modes
    parser.add_argument("--mode", choices=["train", "test", "train_and_test"], default="train_and_test", required=True)
    parser.add_argument("--skip-training", action="store_true", help="Skip training")

    # data settings
    parser.add_argument("--csv", type=str, required=True)
    parser.add_argument("--audio-dir", type=str, required=True)
    parser.add_argument("--split", type=float, default=0.8)

    # training settings
    parser.add_argument("--tune-mode", choices=["full", "head", "two_phase"], default="two_phase")
    parser.add_argument("--freeze-epochs", type=int, default=3)
    parser.add_argument("--ft-epochs", type=int, default=5)
    parser.add_argument("--lr-head", type=float, default=1e-5)
    parser.add_argument("--lr-all", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)

    # audio windowing
    parser.add_argument("--target-sr", type=int, default=16000)
    parser.add_argument("--max-seconds", type=float, default=3.0)

    # model & output
    parser.add_argument("--model-id", type=str, default="team-lucid/hubert-large-korean")
    parser.add_argument("--output-dir", type=str, default="model")
    parser.add_argument("--ckpt-path", type=str, default="model/best.pt") # for inference

    parser.add_argument("--save-results", action="store_true", help="Save inference results to CSV")
    parser.add_argument("--results-file", type=str, default="results.csv", help="Output file for results")

    # regularization
    parser.add_argument("--spec-augment", action="store_true")
    parser.add_argument("--time-mask-param", type=int, default=40)

    return parser.parse_args()

def prepare_splits(args):
    df = pd.read_csv(args.csv, low_memory=False)
    # 'cons_type' is the target label
    if 'cons_type' in df.columns:
        df['cons_type'] = df['cons_type'].replace({
            'Underlyingly Aspirated': 'aspirated ',
            'Underlyingly Tense': 'tense',
            'Underlyingly Plain': 'plain'
        })
        df = df[df['cons_type'] != 'Underlyingly Nasal']
        label_col = 'cons_type'
    else:
        label_col = 'label'

    train_df, val_df = train_test_split(
        df, test_size= 1 - args.split, stratify=df[label_col], random_state=42
        )
    val_df, test_df = train_test_split(
        val_df, test_size=0.5, stratify=val_df[label_col], random_state=42
    )

    # label mappings
    classes = sorted(train_df[label_col].unique().tolist())
    label2id = {c: i for i, c in enumerate(classes)}
    id2label = {i: c for i, c in enumerate(classes)} 

    # datasets expect the label as integer id
    train_ds = KoreanStopDataset(train_df, args.audio_dir, label_col=label_col, label2id=label2id)
    val_ds = KoreanStopDataset(val_df, args.audio_dir, label_col=label_col, label2id=label2id)
    test_ds = KoreanStopDataset(test_df, args.audio_dir, label_col=label_col, label2id=label2id)

    print(f"Splits → Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")
   
    return train_ds, val_ds, test_ds, label2id, id2label


def build_cfg(args, num_labels, id2label, label2id):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return FTConfig(
        model_id=args.model_id,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
        target_sr=args.target_sr,
        max_seconds=args.max_seconds,

        # training
        freeze_epochs=args.freeze_epochs,
        ft_epochs=args.ft_epochs,
        lr_head=args.lr_head,
        lr_all=args.lr_all,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        fp16=args.fp16,
        device=device,

        # regularization
        spec_augment=args.spec_augment,
        time_mask_param=args.time_mask_param,

        # io
        output_dir=args.output_dir,
    )


@torch.no_grad()
def evaluate_model(model, data_loader, criterion, device, 
                   save_results=False, output_path=None, id2label=None, dataset=None):
    model.eval()
    total_loss, total_n, correct = 0.0, 0, 0

    all_predictions = []
    all_labels = []
    all_probabilities = []
    all_filenames = []

    for batch_idx, batch in enumerate(data_loader):
        batch = {k: v.to(device) for k, v in batch.items()}
        out = model(
            input_values=batch["input_values"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
        )
        loss, logits = out["loss"], out["logits"]
        total_loss += float(loss) * batch["labels"].size(0)
        total_n += batch["labels"].size(0)
        preds = logits.argmax(-1)
        correct += (preds == batch["labels"]).sum().item()
    
        # Collect data for saving
        if save_results:
            batch_size = batch["labels"].size(0)
            probs = torch.softmax(logits, dim=-1)
            
            # Store predictions, labels, and probs
            all_predictions.extend(preds.cpu().numpy())
            all_labels.extend(batch["labels"].cpu().numpy())
            all_probabilities.extend(probs.cpu().numpy())
        
            # Get file names from dataset if available
            if dataset is not None:
                start_idx = batch_idx * data_loader.batch_size
                end_idx = start_idx + batch_size
                batch_filenames = []

                for i in range(batch_size):
                    global_idx = start_idx + i
                    if global_idx < len(dataset):
                        if hasattr(dataset, 'df'):
                            if 'filename' in dataset.df.columns:
                                batch_filenames.append(dataset.df.iloc[global_idx]['filename'])
                            elif 'audio_path' in dataset.df.columns:
                                batch_filenames.append(os.path.basename(dataset.df.iloc[global_idx]['audio_path']))
                            else:
                                batch_filenames.append(f"sample_{global_idx}")
                        else:
                            batch_filenames.append(f"sample_{global_idx}")
                    else:
                        batch_filenames.append(f"sample_{global_idx}")
                all_filenames.extend(batch_filenames)
    
    # Save results to csv
    if save_results and output_path and all_predictions:
        results_df = pd.DataFrame()
        if all_filenames:
            results_df['filename'] = all_filenames
        else:
            results_df['sample_id'] = range(len(all_predictions))
        
        if id2label:
            results_df['true_label'] = [id2label[l] for l in all_labels ]
            results_df['predicted_label'] = [id2label[p] for p in all_predictions]
        else:
            results_df['true_label'] = all_labels
            results_df['predicted_label'] = all_predictions
        
        results_df['confidence'] = [max(probs) for probs in all_probabilities]

    if id2label:
        for class_id, class_name in id2label.items():
            results_df[f'prob_{class_name}'] = [probs[class_id] for probs in all_probabilities]
    else:
        for class_id in range(len(all_probabilities[0])):
            results_df[f'prob_class_{class_id}'] = [probs[class_id] for probs in all_probabilities]
    
    results_df['correct'] = [pred == true for pred, true in zip(all_predictions, all_labels)]

    results_df.to_csv(output_path, index=False)
    print(f"Saved results to {output_path}")
    print(f"Accuracy: {(results_df['correct'].sum() / len(results_df)) * 100:.2f}%")

    return (total_loss / max(1, total_n), (100.0 * correct / max(1, total_n)))

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Data & label maps
    train_ds, val_ds, test_ds, label2id, id2label = prepare_splits(args)

    # 2. Config
    cfg = build_cfg(args, num_labels=len(label2id), id2label=id2label, label2id=label2id)

    # 3. Collator
    collate = AudioCollator(target_sr=cfg.target_sr, max_seconds=cfg.max_seconds)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              collate_fn=collate, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            collate_fn=collate, num_workers=max(1, args.num_workers // 2), pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             collate_fn=collate, num_workers=max(1, args.num_workers // 2), pin_memory=True)
    
    device = cfg.device

    # Initialize checkpoint path
    ckpt_path = args.ckpt_path

    if args.mode == "train" or (args.mode == "train_and_test" and not args.skip_training):
        # 4. Train
        trained_ckpt_path, _ = train_finetune(train_loader, val_loader, cfg)
        print(f"Saved best model to {ckpt_path}")
    
    # Test/evaluation phase
    if args.mode == "test" or args.mode == "train_and_test":
        if args.mode == "train_and_test" and not args.skip_training:
            ckpt_path = os.path.join(args.output_dir, "best.pt")
        else:
            ckpt_path = args.ckpt_path # use specified checkpoint
            
        # Check if checkpoint exists
        if not os.path.isfile(ckpt_path):
            raise FileNotFoundError(f"Checkpoint file {ckpt_path} not found. Cannot perform testing.")

        print(f"Loading model from {ckpt_path}")
        model = load_for_inference(cfg, ckpt_path)

        # 5. Evaluate best model on test set
        output_path = args.results_file if args.save_results else None
        test_loss, test_acc = evaluate_model(
            model, test_loader, CrossEntropyLoss(), device,
            save_results=args.save_results, output_path=output_path,
            id2label=id2label, dataset=test_ds
        )
        print(f"[Test] loss {test_loss:.4f} acc {test_acc:.3f}")
    

if __name__ == "__main__":
    main()