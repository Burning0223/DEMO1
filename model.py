from torch import nn
from transformers import BertModel
class BertClassifier(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config=config

        self.bert=BertModel.from_pretrained(self.config.model_path)
        self.dropout=nn.Dropout(self.config.dropout)
        self.fc=nn.Linear(self.bert.config.hidden_size,self.config.num_classes)

    def forward(self,**batch):
        labels=batch.pop('labels')
        outputs=self.bert(**batch)
        pooled_output=outputs.pooler_output
        x=self.dropout(pooled_output)
        logits=self.fc(x)
        return logits,labels
