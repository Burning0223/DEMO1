import random
import numpy as np
from transformers import get_linear_schedule_with_warmup
import config
from data_process import data_loader,load_data
from model import BertClassifier
from train import Trainer
import torch 
import config

def random_seed(seed):
    random .seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True  # 强制使用确定性的算法
    torch.backends.cudnn.benchmark = False  # 禁用非确定性算法
random_seed(42)



def main():
    train_texts,train_keywords,train_labels=load_data('0.demo1文本分类/train_3k.txt')
    dev_texts,dev_keywords,dev_labels=load_data('0.demo1文本分类/dev_1k.txt')
    ##test_texts,test_keywords,test_labels=load_data('0.demo1文本分类/test_1k.txt')

    train_dataloader=data_loader(train_texts,train_keywords,train_labels,shuffle=True)
    dev_dataloader=data_loader(dev_texts,dev_keywords,dev_labels,shuffle=False)

    model=BertClassifier()

    optimizer=torch.optim.AdamW(
        model.parameters(),lr=config.learning_rate,eps=1e-8
    )
    total_steps = len(train_dataloader) * config.num_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
    
    trainer=Trainer(model,optimizer,scheduler,patience=config.patience)
    trainer.train_with_early_stopping(train_dataloader,dev_dataloader)
    ##test_loss,test_acc,test_report=trainer.test(model,test_dataloader)
    #print(f"测试损失:{test_loss:.4f}")
    #print(f"测试准确率:{test_acc:.4f}")
    #print(f"测试报告:/n{test_report}")
if __name__=="__main__":
    main()
    