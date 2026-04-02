from torch import nn
from transformers import BertModel
import config
class BertClassifier(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.bert=BertModel.from_pretrained(config.model_name)
        self.dropout=nn.Dropout(0.1)
        self.fc=nn.Linear(self.bert.config.hidden_size,config.num_classes)

    def forward(self,input_ids,attention_mask,token_type_ids):
        outputs=self.bert(input_ids=input_ids,attention_mask=attention_mask,token_type_ids=token_type_ids)
        pooled_output=outputs.pooler_output
        x=self.dropout(pooled_output)
        logits=self.fc(x)
        return logits
