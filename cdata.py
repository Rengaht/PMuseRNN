# Practical PyTorch: Generating Names with a Conditional Character-Level RNN
# https://github.com/spro/practical-pytorch

import glob
import unicodedata
import string
import random
import time
import math
import io
import os
import torch
from torch.autograd import Variable

import jieba
import json


dir_name=os.path.dirname(__file__)
raw_path=os.path.join(dir_name,'raw')
data_path=os.path.join(dir_name,'data')

# def init():
#     global list_voc
#     global all_categories
#     global category_lines
#     global n_categories
#     global n_vocs

n_vocs=0
n_categories=0
list_voc=[]
category_lines = {}
all_categories = []
EOS=0

mhuman_poem=302
human_poem={}

category_title={}

# load keyword
keyword_file=os.path.join(raw_path,'keyword')
jieba.load_userdict(keyword_file)

custom_file=os.path.join(raw_path,'custom_dict')
jieba.load_userdict(custom_file)

def parseData():
    print('parsing data....')
    file_name=os.path.join(raw_path,'poem_all_3')
    theme_txt=dict([(1,'money'),(2,'work'),(3,'ex'),(4,'sleep'),(5,'cat')])
    
    poem_txt=io.open(file_name,'r',encoding='utf-8').read()
    poem=poem_txt.split('\n')
    print('#poem= ',len(poem))

    voc_all=[]
    for idx,p in enumerate(poem):        
        category=int(p[0])
        theme=theme_txt[category]
        print('poem of category=',theme)
        p=p[2:]
        if not theme in all_categories:
            all_categories.append(theme)
        
        if not theme in human_poem:
            human_poem[theme]=[]

        if not theme in category_title:
            category_title[theme]=[]

        sentences=p.split('|')

        if idx<mhuman_poem:
            human_poem[theme].append(sentences)
        
        vlist_=jieba.lcut(sentences[0],cut_all=False,HMM=True)
        # print(vlist_)
        # os.system("pause")

        category_title[theme].append(vlist_)

        pstr=[]
        for s in sentences:
            cut=jieba.lcut(s,cut_all=False,HMM=True)
            print(cut)
            time.sleep(0.1)
            # os.system("pause")
            voc_all.extend(cut)
            pstr.extend(cut)
            pstr.append('EOL')
        
        if not theme in category_lines:
            category_lines[theme]=[]
        category_lines[theme].append(pstr)

    
    voc=list(set(voc_all))
    voc.append('EOL')
    voc.append('EOS')
    # print(voc)

    mvoc=len(voc)+1
    EOS=mvoc-1
    mcategory = len(all_categories)    
    # print(category_title['cat'])
    print('load data: #voc=',mvoc,'  #categories=',mcategory)
    return mcategory,mvoc,voc



def unicode_to_ascii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in all_letters
    )

def read_lines(filename):
    lines = open(filename,encoding='utf-8').read().strip().split('\n')
    return [unicode_to_ascii(line) for line in lines]



# Preparing for Training

def random_training_pair():
    category = random.choice(all_categories)
    line = random.choice(category_lines[category])
    return category, line

def make_category_input(category):
    li = all_categories.index(category)
    tensor = torch.zeros(1, n_categories)
    tensor[0][li] = 1
    return Variable(tensor)

def make_chars_input(chars):
    if chars in list_voc:
        tensor = torch.zeros(1, n_vocs)
        try:
            tensor[0][list_voc.index(chars)] = 1
        except Exception as e:
            print(str(e))
        tensor = tensor.view(-1, 1, n_vocs)
        return Variable(tensor)
    else:    
        tensor = torch.zeros(len(chars), n_vocs)
        for ci in range(len(chars)):
            try:
                char = chars[ci]
                tensor[ci][list_voc.index(char)] = 1
            except Exception as e:
                print(str(e))
        tensor = tensor.view(-1, 1, n_vocs)
        return Variable(tensor)


def make_target(line):
    letter_indexes = [list_voc.index(line[li]) for li in range(1, len(line))]
    letter_indexes.append(n_vocs - 1) # EOS
    tensor = torch.LongTensor(letter_indexes)
    return Variable(tensor)

def random_training_set():
    category, line = random_training_pair()    
    category_input = make_category_input(category)
    line_input = make_chars_input(line)    
    line_target = make_target(line)
    return category_input, line_input, line_target

def saveData():
    print('save file...')
    fp=open(os.path.join(data_path,'voc.json'),'w')
    json.dump(list_voc,fp)

    fp2=open(os.path.join(data_path,'category.json'),'w')
    json.dump(all_categories,fp2)
    
    fp3=open(os.path.join(data_path,'category_line.json'),'w')
    json.dump(category_lines,fp3)

    fp4=open(os.path.join(data_path,'category_title.json'),'w')
    json.dump(category_title,fp4)

    fp5=open(os.path.join(data_path,'human_poem.json'),'w')
    json.dump(human_poem,fp5)


def loadData():
    fp=open(os.path.join(data_path,'voc.json'),'r')
    voc=json.loads(fp.read())

    fp2=open(os.path.join(data_path,'category.json'),'r')
    cat=json.load(fp2)

    fp3=open(os.path.join(data_path,'category_line.json'),'r')
    catlines=json.load(fp3)

    fp4=open(os.path.join(data_path,'category_title.json'),'r')
    cattitles=json.load(fp4)

    fp5=open(os.path.join(data_path,'human_poem.json'),'r')
    human=json.load(fp5)

    return voc,cat,catlines,cattitles,human


# init()

try:
    list_voc,all_categories,category_lines,category_title,human_poem=loadData()
    n_vocs=len(list_voc)
    n_categories=len(all_categories)    
except Exception as e:
    print(str(e))
    print('load data failed!!!')
    n_categories,n_vocs,list_voc=parseData()
    saveData()    


