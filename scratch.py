
# ----------------------------
# Utilities
# ----------------------------
def freeze_encoder(model: HuBERTClassifier):
    for p in model.encoder.parameters():
        p.requires_grad = False

def unfreeze_encoder(model: HuBERTClassifier):
    for p in model.encoder.parameters():
        p.requires_grad = True

def spec_augment_waveform(batch: Dict[str, torch.Tensor], cfg: FTConfig):
    """
    Very lightweight SpecAugment-like masking on log-mel features would be better,
    but if you operate on the raw encoder (which handles features internally),
    you can optionally add simple time masking by zeroing random spans.
    """
    if not cfg.spec_augment:
        return batch
    x = batch["input_values"]  # (B, T)
    B, T = x.shape
    # time masking
    max_len = min(cfg.time_mask_param, T // 10 + 1)
    for b in range(B):
        span = torch.randint(low=1, high=max_len, size=(1,)).item()
        start = torch.randint(low=0, high=max(1, T - span), size=(1,)).item()
        x[b, start:start+span] = 0.0
    return batch

# ----------------------------
# Collator
# ----------------------------
class AudioCollator:
    """
    Pads variable-length waveforms to the longest in batch and builds attention_mask.
    Optionally truncates very long clips for memory.
    """
    def __init__(self, processor: AutoProcessor, target_sr: int, max_seconds: float = None):
        self.processor = processor
        self.target_sr = target_sr
        self.max_len = int(target_sr * max_seconds) if max_seconds else None

    def __call__(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
        # features: each has {"input_values": 1D numpy/torch, "labels": int}
        waves = []
        labels = []
        for f in features:
            w = torch.as_tensor(f["input_values"], dtype=torch.float32)
            if self.max_len is not None and w.shape[0] > self.max_len:
                # random crop for training robustness; for eval you might center-crop
                start = torch.randint(0, w.shape[0] - self.max_len + 1, (1,)).item()
                w = w[start:start + self.max_len]
            waves.append(w)
            labels.append(f["labels"])

        lengths = [len(w) for w in waves]
        maxlen = max(lengths)
        padded = torch.zeros(len(waves), maxlen, dtype=torch.float32)
        attn = torch.zeros(len(waves), maxlen, dtype=torch.long)
        for i, w in enumerate(waves):
            L = w.shape[0]
            padded[i, :L] = w
            attn[i, :L] = 1

        batch = {"input_values": padded, "attention_mask": attn, "labels": torch.tensor(labels, dtype=torch.long)}
        return batch


# ----------------------------
# Train / Eval loops
# ----------------------------
def run_epoch(model, loader, optimizer, scheduler, scaler, cfg: FTConfig, train: bool = True):
    model.train(train)
    total_loss, total_n = 0.0, 0
    correct = 0

    for step, batch in enumerate(loader):
        batch = {k: v.to(cfg.device) for k, v in batch.items()}
        batch = spec_augment_waveform(batch, cfg) if train else batch

        def fwd():
            out = model(input_values=batch["input_values"],
                        attention_mask=batch["attention_mask"],
                        labels=batch["labels"])
            return out["loss"], out["logits"]

        if train:
            optimizer.zero_grad(set_to_none=True)
            if cfg.fp16 and scaler is not None:
                with torch.cuda.amp.autocast():
                    loss, logits = fwd()
                scaler.scale(loss).backward()
                clip_grad_norm_(model.parameters(), cfg.grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss, logits = fwd()
                loss.backward()
                clip_grad_norm_(model.parameters(), cfg.grad_clip)
                optimizer.step()

            if scheduler is not None:
                scheduler.step()
        else:
            with torch.no_grad():
                loss, logits = fwd()

        total_loss += float(loss.detach()) * batch["labels"].size(0)
        total_n += batch["labels"].size(0)
        preds = logits.argmax(-1)
        correct += (preds == batch["labels"]).sum().item()

    avg_loss = total_loss / max(1, total_n)
    acc = correct / max(1, total_n)
    return avg_loss, acc


def fit_two_phase(model: HuBERTClassifier,
                  train_loader: DataLoader,
                  val_loader: DataLoader,
                  cfg: FTConfig):
    os.makedirs(cfg.output_dir, exist_ok=True)
    model.to(cfg.device)

    # ----- Phase A: freeze encoder, train head -----
    if cfg.freeze_epochs > 0:
        freeze_encoder(model)
        head_params = [p for n, p in model.named_parameters() if p.requires_grad]
        opt = AdamW(head_params, lr=cfg.lr_head, weight_decay=cfg.weight_decay)

        num_steps = math.ceil(len(train_loader) * cfg.freeze_epochs)
        warmup = int(num_steps * cfg.warmup_ratio)
        sch = get_cosine_schedule_with_warmup(opt, warmup, num_steps)

        scaler = torch.cuda.amp.GradScaler() if (cfg.fp16 and cfg.device.startswith("cuda")) else None

        best_val = float("inf")
        best_path = os.path.join(cfg.output_dir, "best_head.pt")

        for ep in range(1, cfg.freeze_epochs + 1):
            tr_loss, tr_acc = run_epoch(model, train_loader, opt, sch, scaler, cfg, train=True)
            va_loss, va_acc = run_epoch(model, val_loader, None, None, None, cfg, train=False)
            print(f"[Freeze] epoch {ep}/{cfg.freeze_epochs} | train loss {tr_loss:.4f} acc {tr_acc:.3f} | "
                  f"val loss {va_loss:.4f} acc {va_acc:.3f}")
            if va_loss < best_val:
                best_val = va_loss
                torch.save(model.state_dict(), best_path)

        # load best head
        if os.path.exists(best_path):
            model.load_state_dict(torch.load(best_path, map_location=cfg.device))

    # ----- Phase B: unfreeze encoder, fine-tune all -----
    unfreeze_encoder(model)
    opt = AdamW(model.parameters(), lr=cfg.lr_all, weight_decay=cfg.weight_decay)

    num_steps = math.ceil(len(train_loader) * cfg.ft_epochs)
    warmup = int(num_steps * cfg.warmup_ratio)
    sch = get_cosine_schedule_with_warmup(opt, warmup, num_steps)

    scaler = torch.cuda.amp.GradScaler() if (cfg.fp16 and cfg.device.startswith("cuda")) else None

    best_val = float("inf")
    best_path = os.path.join(cfg.output_dir, "best.pt")

    for ep in range(1, cfg.ft_epochs + 1):
        tr_loss, tr_acc = run_epoch(model, train_loader, opt, sch, scaler, cfg, train=True)
        va_loss, va_acc = run_epoch(model, val_loader, None, None, None, cfg, train=False)
        print(f"[FT] epoch {ep}/{cfg.ft_epochs} | train loss {tr_loss:.4f} acc {tr_acc:.3f} | "
              f"val loss {va_loss:.4f} acc {va_acc:.3f}")
        if va_loss < best_val:
            best_val = va_loss
            torch.save(model.state_dict(), best_path)

    # return path to best checkpoint
    return os.path.join(cfg.output_dir, "best.pt")


# ----------------------------
# Public API (called by run.py)
# ----------------------------
def build_model(cfg: FTConfig) -> HuBERTClassifier:
    return HuBERTClassifier(cfg)

def train_finetune(train_loader: DataLoader, val_loader: DataLoader, cfg: FTConfig):
    model = build_model(cfg)
    ckpt = fit_two_phase(model, train_loader, val_loader, cfg)
    print(f"Saved best model to: {ckpt}")
    return ckpt, model.processor  # return processor so run.py can save it too

def load_for_inference(cfg: FTConfig, ckpt_path: str) -> HuBERTClassifier:
    model = build_model(cfg)
    sd = torch.load(ckpt_path, map_location=cfg.device)
    model.load_state_dict(sd)
    model.to(cfg.device).eval()
    return model
