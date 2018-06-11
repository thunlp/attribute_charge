DOC_LEN=500
REG_LEN=1000
neg_table_size=1000000
NEG_SAMPLE_POWER=0.75
batch_size=32
num_epochs=5
embed_size=100
lr=1e-3
top_num = 20
num_layers = 2
neg_times = 3
weight_to_see = 20

value2num = {'Y':0,'N':1,'NA':2,'Y*':3,'N*':4}
num2value = {value2num[key]:key for key in value2num}
attr2num = {"profit":0,"buy_and_sell":1,"death":2,"violence":3,"official":4,
"public":5,"occupy":6,"severe_injury":7,"purposed":8,"production":9}
num_of_attr = len(attr2num)
num2attr = {attr2num[key]:key for key in attr2num}
