import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer, AutoModel

class BaseModel(nn.Module):
    def __init__(self):
        super(BaseModel, self).__init__()
        self.cfg = 1
        self.backbone = AutoModel.from_pretrained("roberta-large")
        self.projector = nn.Linear(self.backbone.config.hidden_size, 6)
        
        
    def forward(self, input_ids, attention_mask):
        x = self.backbone(input_ids=input_ids, attention_mask=attention_mask)["last_hidden_state"].mean(1)
        outputs = self.projector(x)
        return outputs