import torch
from torch.utils.data import Dataset,DataLoader
from transformers import BertTokenizer
import config
from sklearn.model_selection import train_test_split
def load_data(data_path):
    
    texts=[]
    labels=[]
    keywords=[]

    labels2id={
        100:0,101:1,102:2,103:3,104:4,106:5,
        107:6,108:7,109:8,110:9,112:10,
        113:11,114:12,115:13,116:14
    }

    with open(data_path,'r',encoding='utf-8') as f:
        for line in f:
            parts=line.strip().split("_!_")

            label_code=int(parts[1])
            text=parts[3]
            keyword=parts[4]

            keyword=keyword.replace(","," ")

            label=labels2id[label_code]

            texts.append(text)
            keywords.append(keyword)
            labels.append(label)

    return texts,keywords,labels

def data_split(texts,keywords,labels,test_size=0.15,dev_size=0.15,random=42):
    x_train,x_temp,y_train,y_temp=train_test_split(
        texts,labels,test_size=test_size+dev_size,random_state=random,stratify=labels
    )
    x_dev,x_test,y_dev,y_test=train_test_split(
        x_temp,y_temp,test_size=test_size,random_state=random,stratify=y_temp
    )
    keywords_train=keywords[:len(x_train)]
    keywords_dev=keywords[len(x_train):len(x_train)+len(x_dev)]
    keywords_test=keywords[len(x_train)+len(x_dev):]
    return x_train,keywords_train,y_train,x_dev,keywords_dev,y_dev,x_test,keywords_test,y_test

tokenizer=BertTokenizer.from_pretrained(config.model_name)
class TextClassificationDataset(Dataset):
    def __init__(self,texts,keywords,labels):
        super().__init__()
        self.texts=texts
        self.keywords=keywords
        self.labels=labels

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self,idx):
        text=self.texts[idx]
        keyword=self.keywords[idx]
        label=self.labels[idx]
        return text,keyword,label


def collate_fn(batch):
    texts,keywords,labels=zip(*batch)
    encoding=tokenizer(list(texts),list(keywords),return_tensors='pt',max_length=config.max_length,
                           padding=True,truncation=True)
    return {
            'input_ids':encoding['input_ids'],
            'attention_mask':encoding['attention_mask'],
            'token_type_ids':encoding['token_type_ids'],
            'labels':torch.tensor(labels)
        }

def data_loader(texts,keywords,labels,shuffle):
    dataset=TextClassificationDataset(texts,keywords,labels)
    dataloader=DataLoader(dataset,batch_size=config.batch_size,shuffle=shuffle,
                          collate_fn=collate_fn)
    return dataloader

    