import json

class Cls_Config:
    def __init__(self,config_file):
        self.config_file=config_file
        self.config=self.load_config()

    def load_config(self):
        with open(self.config_file,'r', encoding='utf-8') as f:
            return json.load(f)
        
    def get(self,key,default=None):
        try:
            return self.config[key]
        except KeyError:
            print(f"警告：未找到配置项{key}，返回默认值{default}")
            return default
                
