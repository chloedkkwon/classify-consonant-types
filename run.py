import argparse
import torch
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
import pandas as pd
from sklearn.model_selection import train_test_split

from src.hubert_classifier import KoreanStopClassifier
from src.dataset import KoreanStopDataset
from src.utils import collate_fn, evaluate_model, save_model, load_model

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "test"], required=True)
    parser.add_argument("--csv", type=str, required=True)
    parser.add_argument("--audio_dir", type=str, required=True)
    parser.add_argument("--save_path", type=str, default="model/saved_model.pt")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--pooling", type=str, default="mean", choices=["mean", "attention"])
    parser.add_argument("--freeze_encoder", action="store_true")
    parser.add_argument("--split", type=float, default=0.8)
    return parser.parse_args()

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load and clean data
    df = pd.read_csv(args.csv, low_memory=False)
    df['cons_type'] = df['cons_type'].replace({
        'Underlyingly Aspirated': 'aspirated',
        'Underlyingly Tense': 'tense',
        'Underlyingly Plain': 'plain'
    })
    df = df[df['cons_type'] != 'Underlyingly Nasal']

    # Stratified train/val/test split
    train_df, temp_df = train_test_split(df, test_size=1 - args.split, stratify=df['cons_type'], random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df['cons_type'], random_state=42)

    print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    # Datasets and loaders
    train_dataset = KoreanStopDataset(train_df, args.audio_dir)
    val_dataset = KoreanStopDataset(val_df, args.audio_dir)
    test_dataset = KoreanStopDataset(test_df, args.audio_dir)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=2)

    # Initialize model
    model = KoreanStopClassifier(pooling=args.pooling, freeze_encoder=args.freeze_encoder).to(device)
    criterion = CrossEntropyLoss()

    if args.mode == "train":
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

        best_val_loss = float('inf')
        for epoch in range(args.epochs):
            model.train()
            total_loss = 0.0

            for input_values, labels in train_loader:
                input_values, labels = input_values.to(device), labels.to(device)
                optimizer.zero_grad()
                logits = model(input_values)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * input_values.size(0)

            avg_train_loss = total_loss / len(train_loader.dataset)
            val_loss, val_acc = evaluate_model(model, val_loader, criterion, device)

            print(f"[Epoch {epoch + 1}] Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_model(model, args.save_path)
                print(f"✅ Saved best model to {args.save_path}")

    elif args.mode == "test":
        model = load_model(model, args.save_path, device)
        test_loss, test_acc = evaluate_model(model, test_loader, criterion, device)
        print(f"🧪 Test Loss: {test_loss:.4f} | Test Accuracy: {test_acc:.2f}%")

    else:
        raise ValueError("Mode must be either 'train' or 'test'.")

if __name__ == "__main__":
    main()
