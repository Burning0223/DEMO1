import os
import json
import torch
from collections import Counter

class Cls_Config:
    def __init__(self,config_file):
        self.config_file=config_file
        self.config=self.load_config()
        self.model_path=self.config.get("model_path","../bert-base-uncased")
        self.max_length=self.config.get("max_length",128)
        self.num_epochs=self.config.get("num_epochs",10)
        self.batch_size=self.config.get("batch_size",16)
        self.learning_rate=self.config.get("learning_rate",2e-5)
        self.patience=self.config.get("patience",2)
        self.num_classes=self.config.get("num_classes",15)
        self.data_path=self.config.get("data_path","../0.demo1文本分类/toutiao_cat_data.txt")
        self.dropout=self.config.get("dropout",0.1)
        self.test_size=self.config.get("test_size",0.15)
        self.dev_size=self.config.get("dev_size",0.15)
        self.random=self.config.get("random",42)
        self.delta=self.config.get("delta",0)
    def load_config(self):
        with open(self.config_file,'r', encoding='utf-8') as f:
            return json.load(f)

class EarlyStopping:
    def __init__(self,config,verbose=False):
        self.config=config
        self.patience=self.config.patience
        self.verbose=verbose
        self.delta=self.config.delta
        self.counter=0
        self.best_score=None
        self.early_stop=False
        self.best_model_path=None
        self.dev_best_acc=0

        self.experiment_name=f"max_length_{self.config.max_length}_num_epochs_{self.config.num_epochs}_batch_size_{self.config.batch_size}_lr_{self.config.learning_rate}"
        self.experiment_dir=os.path.join("experiment",self.experiment_name)
        os.makedirs(self.experiment_dir,exist_ok=True)
    def __call__(self,dev_loss,dev_acc,model,optimizer,scheduler,epoch):
        score=-dev_loss

        if self.best_score is None:
            self.best_score=score
            self.dev_best_acc=dev_acc
            self.save_checkpoint(model,optimizer,scheduler,epoch,dev_loss,dev_acc)
        elif score<self.best_score+self.delta:
            self.counter+=1
            if dev_acc>self.dev_best_acc:
                self.dev_best_acc=dev_acc
                self.save_checkpoint(model,optimizer,scheduler,epoch,dev_loss,dev_acc)
            if self.verbose:
                print(f'早停计数器:{self.counter}/{self.patience}')
            if self.counter>=self.patience:
                self.early_stop=True
        else:
            self.best_score=score
            if dev_acc>self.dev_best_acc:
                self.dev_best_acc=dev_acc
                self.save_checkpoint(model,optimizer,scheduler,epoch,dev_loss,dev_acc)
            self.counter=0
        return self.early_stop
    
    def save_checkpoint(self,model,optimizer,scheduler,epoch,loss,acc):
        checkpoint_name=f"checkpoint_epoch_{epoch+1}.pt"
        checkpoint_path=os.path.join(self.experiment_dir,checkpoint_name)
        checkpoint={
            'epoch':epoch,
            'model':model.state_dict(),
            'optimizer':optimizer.state_dict(),
            'scheduler':scheduler.state_dict(),
            'loss':loss,
            'acc':acc
        }
        torch.save(checkpoint,checkpoint_path)
        print(f"保存epoch{epoch+1}的checkpoint:{checkpoint_path}")
        if acc==self.dev_best_acc:
            self.best_model_path=checkpoint_path
            print(f"保存最佳模型：{checkpoint_path}")

class Metrics:
    def __init__(self,true_labels,pred_labels,id2label,config):
        self.num_classes=config.num_classes
        self.true_labels=true_labels
        self.pred_labels=pred_labels
        self.id2label=id2label
        self.tp=[0]*self.num_classes
        self.fp=[0]*self.num_classes
        self.fn=[0]*self.num_classes
        self.tn=[0]*self.num_classes
        self.calculate_tp_fp_fn_tn()
        self.acc=self.calculate_acc()
        self.precision=self.calculate_precision()
        self.recall=self.calculate_recall()
        self.f1=self.calculate_f1()
        self.support=self.calculate_support()
        self.report=self.print_result()
    
    def calculate_tp_fp_fn_tn(self):
        for t,p in zip(self.true_labels,self.pred_labels):
            for i in range(self.num_classes):
                if t==i and p==i:
                    self.tp[i]+=1
                elif t!=i and p==i:
                    self.fp[i]+=1
                elif t==i and p!=i:
                    self.fn[i]+=1
                elif t!=i and p!=i:
                    self.tn[i]+=1

    def calculate_acc(self):
        correct_predictions=sum([1 for t,p in zip(self.true_labels,self.pred_labels) if t==p])
        total_samples=len(self.true_labels)
        return correct_predictions/total_samples if total_samples>0 else 0

    def calculate_precision(self):
        precision=[]
        for i in range(self.num_classes):
            if self.tp[i]+self.fp[i]>0:
                precision.append(self.tp[i]/(self.tp[i]+self.fp[i]))
            else:
                precision.append(0)
        return precision
    def calculate_recall(self):
        recall=[] 
        for i in range(self.num_classes):
            if self.tp[i]+self.fn[i]>0:
                recall.append(self.tp[i]/(self.tp[i]+self.fn[i]))
            else:
                recall.append(0)
        return recall
    def calculate_f1(self):
        f1=[]
        for i in range(self.num_classes):
            if self.precision[i]+self.recall[i]>0:
                f1.append(2*self.precision[i]*self.recall[i]/(self.precision[i]+self.recall[i]))
            else:
                f1.append(0)
        return f1  
    def calculate_support(self):
        support=[0]*self.num_classes
        for label in self.true_labels:
            label=int(label)
            support[label]+=1
        return support
    def print_result(self):
        print(f"{'Class':<10}{'Precision':<15}{'Recall':<15}{'F1-Score':<15}{'Support':<10}")
        for i in range(self.num_classes):
            real_label=self.id2label[i]
            print(f"{real_label:<10}{self.precision[i]:<15.2f}{self.recall[i]:<15.2f}{self.f1[i]:<15.2f}{self.support[i]:<10}")
        macro_precision=sum(self.precision)/self.num_classes
        macro_recall=sum(self.recall)/self.num_classes
        macro_f1=sum(self.f1)/self.num_classes
        sum_support=sum(self.support)
        print(f"\n{'macro avg':<10}{macro_precision:<15.2f}{macro_recall:<15.2f}{macro_f1:<15.2f}{sum_support:<10}")
        weighted_precision=sum([self.precision[i]*self.support[i] for i in range(self.num_classes)])/sum_support
        weighted_recall=sum([self.recall[i]*self.support[i] for i in range(self.num_classes)])/sum_support
        weighted_f1=sum([self.f1[i]*self.support[i] for i in range(self.num_classes)])/sum_support
        print(f"\n{'weighted avg':<10}{weighted_precision:<15.2f}{weighted_recall:<15.2f}{weighted_f1:<15.2f}{sum_support:<10}")
      


        
