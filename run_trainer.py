import swanlab
from torch import nn
import os
import torch
from untils import Metrics,Cls_Config,create_labels_mapping,load_label_mapping
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
    def __init__(self,model,config,optimizer,scheduler,id2label):
        self.model=model
        self.config=config
        self.optimizer=optimizer
        self.scheduler=scheduler
        self.id2label=id2label
        self.loss_fn=nn.CrossEntropyLoss()

        self.experiment_name=f"max_len_{self.config.max_length}_num_epochs_{self.config.num_epochs}_bs_{self.config.batch_size}_lr_{self.config.learning_rate}_use_key_{self.use_keyword}"
        self.experiment_dir=os.path.join("experiment",self.experiment_name)
        os.makedirs(self.experiment_dir,exist_ok=True)
        swanlab.init(project="Bert_text_classification",
             experiment_name=self.experiment_name,
             config=self.config.config_dict,
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
        train_metrics=Metrics(all_labels,all_preds,self.config,self.id2label)
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
        dev_metrics=Metrics(all_labels,all_preds,self.config,self.id2label)
        dev_acc=dev_metrics.acc
        return ave_loss,dev_acc

    def save_checkpoint(self,train_dataloader,dev_dataloader,test_dataloader):
        dev_best_acc=0.0
        for epoch in range(self.config.num_epochs):
            print(f"Epoch {epoch+1}")
            train_loss,train_acc=self.train(train_dataloader)
            dev_loss,dev_acc=self.dev(dev_dataloader)
            print(f"训练损失:{train_loss:.4f}")
            print(f"训练准确率:{train_acc:.4f}")
            print(f"验证损失:{dev_loss:.4f}")
            print(f"验证准确率:{dev_acc:.4f}")
            
            checkpoint_name=f"checkpoint_epoch_{epoch+1}.pt"
            checkpoint_path=os.path.join(self.experiment_dir,checkpoint_name)
            checkpoint={
                'epoch':epoch,
                'model':self.model.state_dict(),
                'loss':dev_loss,
                'acc':dev_acc
            }
            if self.optimizer is not None:
                checkpoint["optimizer"] = self.optimizer.state_dict()

            if self.scheduler is not None:
                checkpoint["scheduler"] = self.scheduler.state_dict()
            torch.save(checkpoint,checkpoint_path)
            print(f"保存epoch{epoch+1}的checkpoint:{checkpoint_path}")
            if dev_acc>=dev_best_acc:
                dev_best_acc=dev_acc
                best_model_path=checkpoint_path
                print(f"当前模型最优准确率为：{dev_best_acc}")
                print(f"保存最佳模型：{best_model_path}")
            
            swanlab.log({
                "epoch":epoch,
                "训练损失":train_loss,
                "训练准确率":train_acc,
                "验证损失":dev_loss,
                "验证准确率":dev_acc
            })
          
        print(f"模型最优准确率为：{dev_best_acc}")
        test_loss,test_acc=self.test(test_dataloader,best_model_path) 
        print(f"训练损失:{test_loss:.4f}")
        print(f"训练准确率:{test_acc:.4f}")
        swanlab.log({
                "测试损失":test_loss,
                "测试准确率":test_acc
            })

    def test(self,dataloader,best_model_path):
        if not os.path.exists(best_model_path):
            print("未找到最优模型，无法进行测试！")
            return
        else:
            checkpoint=torch.load(best_model_path)
            self.model.load_state_dict(checkpoint['model'])
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
        test_metrics=Metrics(all_labels,all_preds,self.config,self.id2label)
        test_acc=test_metrics.acc
        return ave_loss,test_acc
    
def main(config_path="Bert_Config.json"):
    config=Cls_Config(config_path)
    random_seed(config.random)

    if not os.path.exists(os.path.join(config.data_path,"labels_mapping.json")):
        train_data_path=os.path.join(config.data_path,"train_3k.txt")
        create_labels_mapping(train_data_path)
    label2id, id2label = load_label_mapping(config.data_path)
    if len(label2id)!=config.num_classes:
            print(f"标签类别数不匹配！实际类别数：{len(label2id)}，配置文件中的类别数：{config.num_classes}")
    
    train_dataset=TextClassificationDataset(config=config,dataset_type="train",label2id=label2id)
    dev_dataset=TextClassificationDataset(config=config,dataset_type="dev",label2id=label2id)
    test_dataset=TextClassificationDataset(config=config,dataset_type="test",label2id=label2id)

    train_dataloader=DataLoader(train_dataset,config.batch_size,shuffle=True,collate_fn=train_dataset.collate_fn)
    dev_dataloader=DataLoader(dev_dataset,config.batch_size,shuffle=False,collate_fn=dev_dataset.collate_fn)
    test_dataloader=DataLoader(test_dataset,config.batchsize,shuffle=False,collate_fn=test_dataset.collate_fn)
    model=BertClassifier(config)

    optimizer=torch.optim.AdamW(
        model.parameters(),lr=config.learning_rate
    )
    total_steps = len(train_dataloader) * config.num_epochs
    num_warmup_steps = int(total_steps * 0.1)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=total_steps)
    
    trainer=Trainer(model,config,optimizer,scheduler,id2label)
    trainer.save_checkpoint(train_dataloader,dev_dataloader,test_dataloader)
    
if __name__=="__main__":
    config_path="config_exp/config_exp1.json"
    main(config_path)