from torch.utils.data import DataLoader
import torch.nn.functional as F

class Trainer:
    def __init__(self, model, optimizer, train_dataset, val_dataset, tokenizer, id_to_label, batch_size=16, device="cpu"):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.tokenizer = tokenizer
        self.id_to_label = id_to_label
        self.batch_size = batch_size
        self.device = device

    def train_epoch(self):
        self.model.train()
        loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
        total_loss = 0

        for batch in loader:
            self.optimizer.zero_grad()
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["labels"].to(self.device)

            outputs = self.model(input_ids, attention_mask, labels)
            loss = outputs["loss"]
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()

        return total_loss / len(loader)

    def evaluate(self):
        self.model.eval()
        loader = DataLoader(self.val_dataset, batch_size=self.batch_size)
        preds, labels = [], []

        with torch.no_grad():
            for batch in loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                label_ids = batch["labels"].to(self.device)

                outputs = self.model(input_ids, attention_mask)
                logits = outputs["logits"]
                pred_ids = torch.argmax(logits, dim=-1)

                for p, l in zip(pred_ids.cpu().numpy(), label_ids.cpu().numpy()):
                    preds.append(p)
                    labels.append(l)

        return preds, labels