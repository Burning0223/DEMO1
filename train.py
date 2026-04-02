import swanlab
from torch import nn
import torch
from sklearn.metrics import accuracy_score,classification_report
from early_stopping import EarlyStopping
import config

experiment_name=f"max_length:{config.max_length},num_epochs:{config.num_epochs},batch_size:{config.batch_size},learning_rate:{config.learning_rate}"
swanlab.init(project="Bert_text_classification",
             experiment_name=experiment_name,
             config={
                "max_length":config.max_length,
                "num_epochs":config.num_epochs,
                "batch_size":config.batch_size,
                "learning_rate":config.learning_rate
             },
             mode="offline")
id2label = {
    0: 100,1: 101,2: 102,3: 103,4: 104,
    5: 106,6: 107,7: 108,8: 109,9: 110,
    10: 112,11: 113,12: 114,13: 115,
    14: 116
}

class Trainer():
    def __init__(self,model,optimizer,scheduler,patience):
        self.model=model
        self.optimizer=optimizer
        self.scheduler=scheduler
        self.early_stopping=EarlyStopping(patience=patience,verbose=True)

    def train(self,dataloader):
        self.model.train()
        total_loss=0.0
        all_labels=[]
        all_preds=[]
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
            try:
                all_labels.extend(id2label[label.item()] for label in labels.numpy())
                all_preds.extend(id2label[pred.item()] for pred in preds.numpy())
            except KeyError as e:
                print(f"标签匹配错误：{e}")
                continue
        ave_loss=total_loss/len(dataloader)
        acc=accuracy_score(all_labels,all_preds)
        return ave_loss,acc
    
    def dev(self,dataloader):
        self.model.eval()
        total_loss=0.0
        all_labels=[]
        all_preds=[]
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
                
                all_labels.extend( id2label[label.item()] for label in labels.numpy())
                all_preds.extend(id2label[pred.item()] for pred in preds.numpy())
        ave_loss=total_loss/len(dataloader)
        acc=accuracy_score(all_labels,all_preds)
        report=classification_report(all_labels,all_preds)
        return ave_loss,acc,report

    def test(self,dataloader):
        self.model=self.early_stopping.load_best_model(self.model)
        return self.dev(dataloader)


    def train_with_early_stopping(self,train_dataloader,dev_dataloader):
        dev_best_acc=0
        for epoch in range(config.num_epochs):
            print(f"Epoch {epoch+1}")
            train_loss,train_acc=self.train(train_dataloader)
            dev_loss,dev_acc,dev_report=self.dev(dev_dataloader)
        
            print(f"训练损失:{train_loss:.4f}")
            print(f"训练准确率:{train_acc:.4f}")
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

            if self.early_stopping(dev_acc,self.model,self.optimizer,self.scheduler,epoch,dev_loss):
                print(f"在epoch{epoch}发生训练早停")
                dev_best_acc=self.early_stopping.get_dev_best_acc()
                print(f"最优验证准确率为：{dev_best_acc:.4f}")
                break

        
    
