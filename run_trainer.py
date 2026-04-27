import swanlab
from torch import nn
import torch
from untils import Metrics,EarlyStopping,Cls_Config
import random
import numpy as np
from transformers import get_linear_schedule_with_warmup
from data_process import TextClassificationDataset
from model import BertClassifier
from torch.utils.data import DataLoader

def random_seed(seed):
    random .seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class Trainer():
    def __init__(self,model,config,id2label,optimizer,scheduler):
        self.model=model
        self.config=config
        self.optimizer=optimizer
        self.scheduler=scheduler
        self.early_stopping=EarlyStopping(self.config,verbose=True)
        self.id2label=id2label
        self.loss_fn=nn.CrossEntropyLoss()

        swanlab.init(project="Bert_text_classification",
             experiment_name=self.early_stopping.experiment_name,
             config={
                "max_length":self.config.max_length,
                "num_epochs":self.config.num_epochs,
                "batch_size":self.config.batch_size,
                "learning_rate":self.config.learning_rate
             },
             mode="offline")

    def train(self,dataloader):
        self.model.train()
        total_loss=0.0
        all_labels=[]
        all_preds=[]
        for batch in dataloader:
            self.optimizer.zero_grad()
            output,labels=self.model(**batch)
            preds=torch.argmax(output,dim=1)
            all_labels.extend(labels.numpy())
            all_preds.extend(preds.numpy())
            loss=self.loss_fn(output, labels)
            total_loss+=loss.item()
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()
        ave_loss=total_loss/len(dataloader)
        train_metrics=Metrics(all_labels,all_preds,self.id2label,self.config)
        train_acc=train_metrics.acc
        return ave_loss,train_acc
    
    def dev(self,dataloader):
        self.model.eval()
        total_loss=0.0
        all_labels=[]
        all_preds=[]
        with torch.no_grad():
            for batch in dataloader:
                output,labels=self.model(**batch)
                loss=self.loss_fn(output, labels)
                preds=torch.argmax(output,dim=1)
                all_labels.extend(labels.numpy())
                all_preds.extend(preds.numpy())
                total_loss+=loss.item()
        ave_loss=total_loss/len(dataloader)
        dev_metrics=Metrics(all_labels,all_preds,self.id2label,self.config)
        dev_acc=dev_metrics.acc
        return ave_loss,dev_acc

    def train_with_early_stopping(self,train_dataloader,dev_dataloader):
        dev_best_acc=0
        for epoch in range(self.config.num_epochs):
            print(f"Epoch {epoch+1}")
            train_loss,train_acc=self.train(train_dataloader)
            dev_loss,dev_acc=self.dev(dev_dataloader)
            print(f"训练损失:{train_loss:.4f}")
            print(f"训练准确率:{train_acc:.4f}")
            print(f"验证损失:{dev_loss:.4f}")
            print(f"验证准确率:{dev_acc:.4f}")
            
            swanlab.log({
                "epoch":epoch,
                "训练损失":train_loss,
                "训练准确率":train_acc,
                "验证损失":dev_loss,
                "验证准确率":dev_acc
            })

            if dev_acc>dev_best_acc:
                dev_best_acc=dev_acc
                print(f"当前模型最优准确率为：{dev_best_acc}")
            if self.early_stopping(dev_loss,dev_acc,self.model,self.optimizer,self.scheduler,epoch):
                print(f"在epoch{epoch+1}发生训练早停")
                break
            
            
        print(f"模型最优准确率为：{dev_best_acc}")
    
def main(config_path="Bert_Config.json"):
    config=Cls_Config(config_path)
    random_seed(config.random)

    train_dataset=TextClassificationDataset(config=config,dataset_type="train")
    dev_dataset=TextClassificationDataset(config=config,dataset_type="dev")

    id2label=train_dataset.id2label

    train_dataloader=DataLoader(train_dataset,config.batch_size,shuffle=True,collate_fn=train_dataset.collate_fn)
    dev_dataloader=DataLoader(dev_dataset,config.batch_size,shuffle=False,collate_fn=dev_dataset.collate_fn)
   
    model=BertClassifier(config)

    optimizer=torch.optim.AdamW(
        model.parameters(),lr=config.learning_rate
    )
    total_steps = len(train_dataloader) * config.num_epochs
    num_warmup_steps = int(total_steps * 0.1)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=total_steps)
    
    trainer=Trainer(model,config,id2label,optimizer,scheduler)
    trainer.train_with_early_stopping(train_dataloader,dev_dataloader)
    
if __name__=="__main__":
    config_path="Bert_Config/Bert_Config_exp1.json"
    main(config_path)