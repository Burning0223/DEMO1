import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer
from sklearn.model_selection import train_test_split
import json


class TextClassificationDataset(Dataset):
    def __init__(self,dataset_type,config):
        super().__init__()
        self.config=config
        self.label2id,self.id2label=self.get_id_label()
        self.texts,self.keywords,self.labels=self.load_data()
        self.num_classes=len(self.label2id)
        self.train_texts,self.train_keywords,self.train_labels,\
        self.dev_texts,self.dev_keywords,self.dev_labels,\
        self.test_texts,self.test_keywords,self.test_labels=self.data_split(
            self.texts,self.keywords,self.labels
            )
        
        if dataset_type=="train":
            self.texts,self.keywords,self.labels=self.train_texts,self.train_keywords,self.train_labels
        elif dataset_type=="dev":
            self.texts,self.keywords,self.labels=self.dev_texts,self.dev_keywords,self.dev_labels
        elif dataset_type=="test":
            self.texts,self.keywords,self.labels=self.test_texts,self.test_keywords,self.test_labels
        else:
            print("dataset_type无效")
        
        self.tokenizer=BertTokenizer.from_pretrained(config.model_path)
        
    def get_id_label(self):
        label_set=set()
        with open(self.config.data_path,'r',encoding='utf-8') as f:
            for line in f:
                parts=line.strip().split("_!_")

                label_code=int(parts[1])
                label_set.add(label_code)

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
    def load_data(self):
        texts=[]
        labels=[]
        keywords=[]
        with open(self.config.data_path,'r',encoding='utf-8') as f:
            for line in f:
                parts=line.strip().split("_!_")

                label_code=int(parts[1])
                text=parts[3]
                keyword=parts[4]

                keyword=keyword.replace(","," ")
                texts.append(text)
                keywords.append(keyword)

                if label_code in self.label2id:
                    label=self.label2id[label_code]
                else:
                    print(f"警告：标签{label_code}未在label2id中找到,默认为-1")
                    label=-1
                labels.append(label)

        return texts,keywords,labels
    
    def data_split(self,texts,keywords,labels):
        x_train,x_temp,y_train,y_temp=train_test_split(
            texts,labels,test_size=self.config.test_size+self.config.dev_size,random_state=self.config.random,stratify=labels
        )
        x_dev,x_test,y_dev,y_test=train_test_split(
            x_temp,y_temp,test_size=0.5,random_state=self.config.random,stratify=y_temp
        )
        keywords_train=keywords[:len(x_train)]
        keywords_dev=keywords[len(x_train):len(x_train)+len(x_dev)]
        keywords_test=keywords[len(x_train)+len(x_dev):]
        return x_train,keywords_train,y_train,x_dev,keywords_dev,y_dev,x_test,keywords_test,y_test

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self,idx):
        text=self.texts[idx]
        keyword=self.keywords[idx]
        label=self.labels[idx]
        return text,keyword,label

    def collate_fn(self,batch):
        texts,keywords,labels=zip(*batch)
        encoding=self.tokenizer(list(texts),list(keywords),return_tensors='pt',max_length=self.config.max_length,
                            padding=True,truncation=True)
        return {
                'input_ids':encoding['input_ids'],
                'attention_mask':encoding['attention_mask'],
                'token_type_ids':encoding['token_type_ids'],
                'labels':torch.tensor(labels)
                }
    


    