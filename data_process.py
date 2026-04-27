import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer
import json
import pandas as pd

class TextClassificationDataset(Dataset):
    def __init__(self,config,dataset_type):
        super().__init__()
        self.config=config
        if dataset_type=="train":
            csv_file=self.config.train_data_path
        elif dataset_type=="dev":
            csv_file=self.config.dev_data_path
        elif dataset_type=="test":
            csv_file=self.config.test_data_path
        else:
            raise ValueError(f"Unknown dataset type: {dataset_type}")
        self.label2id,self.id2label=self.get_id_label()
        self.texts,self.keywords,self.labels=self.load_data(csv_file)
        self.tokenizer=BertTokenizer.from_pretrained(config.model_path)
        
    def get_id_label(self):
        label_set=set()
        for dataset in ['train','dev','test']:
            csv_file=self.config.config_dict.get(f"{dataset}_data_path")
            data=pd.read_csv(csv_file)
            label_set.update(data['label'].tolist())
        label2id={label:id for id,label in enumerate(sorted(label_set))}
        id2label={id:label for label,id in label2id.items()}
        if len(label2id)!=self.config.num_classes:
            print(f"标签类别数不匹配！实际类别数：{len(label2id)}，配置文件中的类别数：{self.config.num_classes}")
        mappings={
                    "label2id":label2id,
                    "id2label":id2label
                }
        with open("label_mapping.json",'w',encoding="utf-8") as f:
                json.dump(mappings,f,ensure_ascii=False,indent=4)

        return label2id,id2label
    def load_data(self,csv_file):
        data=pd.read_csv(csv_file)
        texts=data['text'].tolist()
        keywords=data['keyword'].tolist()
        labels=data['label'].tolist()
        labels=[self.label2id.get(label,-1) for label in labels]
        return texts,keywords,labels

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self,idx):
        text=self.texts[idx]
        keyword=self.keywords[idx]
        label=self.labels[idx]
        return text,keyword,label

    def collate_fn(self,batch):
        texts,keywords,labels=zip(*batch)
        encoding=self.tokenizer([str(text) for text in texts],[str(keyword) for keyword in keywords],
                                return_tensors='pt',max_length=self.config.max_length,
                                padding=True,truncation=True)
        return {
                'input_ids':encoding['input_ids'],
                'attention_mask':encoding['attention_mask'],
                'token_type_ids':encoding['token_type_ids'],
                'labels':torch.tensor(labels)
                }
    


    