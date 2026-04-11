import swanlab
from torch import nn
import torch
from untils import Metrics,EarlyStopping



class Trainer():
    def __init__(self,model,optimizer,scheduler,patience,id2label,config):
        self.model=model
        self.optimizer=optimizer
        self.scheduler=scheduler
        self.early_stopping=EarlyStopping(patience=patience,verbose=True)
        self.id2label=id2label
        self.max_length=config.get("max_length",128)
        self.num_epochs=config.get("num_epochs",10)
        self.batch_size=config.get("batch_size",16)
        self.learning_rate=config.get("learning_rate",2e-5)


        experiment_name=f"max_length:{self.max_length},num_epochs:{self.nfig.num_epochs},batch_size:{self.batch_size},learning_rate:{self.learning_rate}"
        swanlab.init(project="Bert_text_classification",
             experiment_name=experiment_name,
             config={
                "max_length":self.max_length,
                "num_epochs":self.num_epochs,
                "batch_size":self.batch_size,
                "learning_rate":self.learning_rate
             },
             mode="offline")

    def train(self,dataloader):
        self.model.train()
        total_loss=0.0

        for batch in dataloader:
            self.optimizer.zero_grad()
            input_ids=batch['input_ids']
            attention_mask=batch['attention_mask']
            token_type_ids=batch['token_type_ids']
            labels=batch['labels']
            output=self.model(input_ids,attention_mask,token_type_ids)
            preds=torch.argmax(output,dim=1)
            loss=nn.CrossEntropyLoss()(output, labels)
            total_loss+=loss.item()
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()
           
        ave_loss=total_loss/len(dataloader)
        train_metrics=Metrics(labels,preds)
        train_acc=train_metrics.acc
        train_support=train_metrics.support
        return ave_loss,train_acc,train_support
    
    def dev(self,dataloader):
        self.model.eval()
        total_loss=0.0
        
        with torch.no_grad():
            for batch in dataloader:
                input_ids=batch['input_ids']
                attention_mask=batch['attention_mask']
                token_type_ids=batch['token_type_ids']
                labels=batch['labels']
                output=self.model(input_ids,attention_mask,token_type_ids)
                loss=nn.CrossEntropyLoss()(output, labels)
                preds=torch.argmax(output,dim=1)
                total_loss+=loss.item()
        ave_loss=total_loss/len(dataloader)
        dev_metrics=Metrics(labels,preds)
        dev_acc=dev_metrics.acc
        dev_report=dev_metrics.report
        return ave_loss,dev_acc,dev_report

    def train_with_early_stopping(self,train_dataloader,dev_dataloader):
        dev_best_acc=0
        for epoch in range(self.num_epochs):
            print(f"Epoch {epoch+1}")
            train_loss,train_acc,train_support=self.train(train_dataloader)
            dev_loss,dev_acc,dev_report=self.dev(dev_dataloader)
        
            print(f"训练损失:{train_loss:.4f}")
            print(f"训练准确率:{train_acc:.4f}")
            print(f"训练集各分类样本数量：{train_support}")
            print(f"验证损失:{dev_loss:.4f}")
            print(f"验证准确率:{dev_acc:.4f}")
            print(f"验证报告:\n{dev_report}")
            """
            :.4f 这种格式化写法只能用在 f-string 里
            不能直接写在字典里
            """
            
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
    
