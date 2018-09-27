import jieba
import sys
import jieba.posseg as pseg
import io
# import datetime
import time
import json

voc_text=io.open('keyword','r',encoding='utf-8').read()
lines=voc_text.split('\n')

voc=[]
for line in lines:
	v=line.split(' ')
	voc.append(v[0])

print(voc)
fp=open('keyword.json','w')
json.dump(voc,fp)

# now=datetime.datetime.now()
# print(datetime.time(now.year))
# print(datetime.time(now.month))
# print(datetime.time(now.day))
# print(time.strftime('%Y'))

# words=pseg.cut("洋蔥")
# for w in words:
# 	print('%s %s' % (w.word, w.flag))

# category_lines = {}
# all_categories = []
# voc_all=[]

# file_name='voc_sample'
# poem_txt=io.open(file_name,'r',encoding='utf-8').read()
# poem=poem_txt.split('\n')
# print('#poem= ',len(poem))

# voc_all=[]
# for p in poem:
#     theme=int(p[0])
#     p=p[2:]
#     print(p)
#     sentences=p.split(' ')
#     pstr=[]
#     for s in sentences:
#         cut=list(jieba.cut_for_search(s))
#         voc_all.extend(cut)
#         pstr.extend(cut)
#         pstr.append('EOL')
    
#     if not theme in category_lines:
#         category_lines[theme]=[]
#     category_lines[theme].append(pstr)
         
# print(voc_all)
# print(category_lines)