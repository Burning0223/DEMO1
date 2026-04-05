# 新闻文本分类任务
基于 bert-base-chinese模型，使用 PyTorch 和 Hugging Face ，完成文本分类任务的训练与评估。
## 数据集
来源：https://github.com/aceimnorstuvwxz/toutiao-text-classfication-dataset来自今日头条客户端，共15个类别：
'''
100 民生 故事 news_story
101 文化 文化 news_culture
102 娱乐 娱乐 news_entertainment
103 体育 体育 news_sports
104 财经 财经 news_finance
106 房产 房产 news_house
107 汽车 汽车 news_car
108 教育 教育 news_edu 
109 科技 科技 news_tech
110 军事 军事 news_military
112 旅游 旅游 news_travel
113 国际 国际 news_world
114 证券 股票 stock
115 农业 三农 news_agriculture
116 电竞 游戏 news_game
'''
## 项目结构
···
├── config.py          # 配置类
├── model.py           # BERT 模型定义
├── data_process.py    #数据预处理
├── train.py           # 训练器
├── early_stopping.py  # 早停机制        
├── main.py            #主程序
├── requirements.txt  # 环境依赖
└── README.md         # 项目说明
···
## 环境依赖
'''
pip install -r requirements.txt
'''
##模型参数设置
'''
model_name="bert-base-uncased"
max_length=128
num_epochs=10
batch_size=16
num_classes=15
learning_rate=2e-5
patience=2
'''
## 实验结果
'''
训练损失:0.6435
训练准确率:0.7987
验证损失:1.1352
验证准确率:0.6954
验证报告:
              precision    recall  f1-score   support

         100       0.68      0.66      0.67      1600
         101       0.72      0.74      0.73      7148
         102       0.67      0.75      0.71     10046
         103       0.78      0.73      0.76      9579
         104       0.54      0.62      0.58      6906
         106       0.66      0.66      0.66      4507
         107       0.73      0.67      0.69      9126
         108       0.74      0.74      0.74      6899
         109       0.68      0.71      0.69     10593
         110       0.70      0.64      0.67      6371
         112       0.62      0.63      0.63      5463
         113       0.72      0.69      0.70      6862
         114       0.25      0.01      0.02        87
         115       0.64      0.59      0.61      4927
         116       0.82      0.79      0.80      7471

    accuracy                           0.70     97585
   macro avg       0.66      0.64      0.64     97585
weighted avg       0.70      0.70      0.70     97585

'''

