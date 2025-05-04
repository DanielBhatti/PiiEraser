import torch
import torch.nn as nn
from transformers import BertModel

class PiiBertTokenClassifier(nn.Module):
    def __init__(self, pretrained_model_name: str, num_labels: int):
        super().__init__()
        self.bert = BertModel.from_pretrained(pretrained_model_name)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask=None, labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = self.dropout(outputs.last_hidden_state)
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            # reshape to (batch * seq_len, num_labels)
            loss = loss_fct(logits.view(-1, logits.shape[-1]), labels.view(-1))

        return {"loss": loss, "logits": logits}
