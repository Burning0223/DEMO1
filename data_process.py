import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer


class TextClassificationDataset(Dataset):
    def __init__(self,config,dataset_type,label2id):
        super().__init__()
        self.config=config
        self.label2id=label2id
        self.texts,self.keywords,self.labels=self.load_data(dataset_type)
        self.tokenizer=BertTokenizer.from_pretrained(config.model_path)

    def load_data(self,dataset_type):
        if dataset_type == "train":
            file_path = f"{self.config.data_path}/train_3k.txt"
        elif dataset_type == "dev":
            file_path = f"{self.config.data_path}/dev_1k.txt"
        elif dataset_type == "test":
            file_path = f"{self.config.data_path}/test_1k.txt"
        else:
            raise ValueError(f"Unknown dataset type: {dataset_type}")
        texts=[]
        labels=[]
        keywords=[]
        with open(file_path,'r',encoding='utf-8') as f:
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
    


    