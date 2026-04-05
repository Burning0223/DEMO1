import torch

class EarlyStopping:
    def __init__(self,patience,verbose=False,delta=0,path='checkpoint.pt'):
        self.patience=patience
        self.verbose=verbose
        self.delta=delta
        self.path=path
        self.counter=0
        self.best_score=None
        self.early_stop=False
        self.best_model=None
        self.dev_best_acc=0
        self.best_model=None
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
        checkpoint={
            'epoch':epoch,
            'model':model.state_dict(),
            'optimizer':optimizer.state_dict(),
            'scheduler':scheduler.state_dict(),
            'loss':loss,
            'acc':acc
        }
        torch.save(checkpoint,self.path)
        print(f"保存epoch{epoch+1}的checkpoint")

    def load_best_model(self,model):
        checkpoint=torch.load(self.path)
        model.load_state_dict(checkpoint['model'])
        return model
    
    