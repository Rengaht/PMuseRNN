import sys
import json
import time
from datetime import datetime
import torch.nn.functional as F
import numpy as np
import os
import pytz

# if len(sys.argv) < 2:
#     print("Usage: generate.py [keyword] [mood]")
#     sys.exit()

# else:
#     language = sys.argv[1]

import torch
import torch.nn as nn
from torch.autograd import Variable

from cdata import *
from model import *

import jieba.posseg as pseg

dir_name=os.path.dirname(__file__)
# Generating from the Network

max_length = 20
def clamp(n, minn,maxn):
    return max(min(maxn,n),minn)

def to_prob(vec):
    s = sum(vec)
    return [v / s for v in vec]

def generate_time_char(pstr):
    zone_=pytz.timezone('Asia/Taipei')
    loctime=datetime.now(zone_)
    pstr=pstr.replace('__yy__',loctime.strftime('%Y'))
    pstr=pstr.replace('__mmm__',loctime.strftime('%m'))
    pstr=pstr.replace('__dd__',loctime.strftime('%d'))
    pstr=pstr.replace('__hh__',loctime.strftime('%H'))
    pstr=pstr.replace('__mm__',loctime.strftime('%M'))
    pstr=pstr.replace('__ss__',loctime.strftime('%S'))
    pstr=pstr.replace('__ww__',loctime.strftime('%w'))
    return pstr

def generate_one(rnn,title,category, temperature=0.8):

    category_input=make_category_input(category)
    # if start_char:
    #     seg_list=list(jieba.cut_for_search(start_char))
    # start_char=random.choice(list_voc)
    # print('generate from ',title)
    chars_input=make_chars_input([x for x in title if x!='|'])
    hidden=rnn.init_hidden()

    output_str=title
    output_str+='|'
    
    sentence_length=0
    i=0
    while i<6:       
        output, hidden = rnn(category_input, chars_input[0], hidden)
        # Sample as a multinomial distribution
        # print(output)
        # output_dist = output.data.view(-1).div(temperature).exp()
        # top_i = torch.multinomial(output_dist, 1)[0]
        # top_i=clamp(top_i,0,len(list_voc)-1) 
        pred=to_prob(F.softmax(output,dim=1).data[0].numpy())
        char=np.random.choice(list_voc, p=pred)
        # char=generate_time_char(char)        
        if char == 'EOS':
            break
        if sentence_length+len(char)>max_length:
            char='EOL'
        try:
            # char = list_voc[top_i]
            # print(char)
            if char=='EOL':
                if sentence_length>2:
                    # print(sentence_length,'add EOL')
                    output_str+='|'
                    i+=1
                    sentence_length=0                
            else:
                sentence_length+=len(char)
                output_str += char

            chars_input = make_chars_input(char)
        except Exception as e:
            print(str(e))
    return output_str,i

def switchKeyword(keyword,pstr):
    # switch keyword
    words=pseg.cut(pstr)    
    keys=pseg.cut(keyword)
    plist=pstr.split('|')

    for k in keys:
        if k.flag=='o':
            ### find right length sentence ###
            proper_length=max_length-len(k.word)
            sentence_to_switch=[x for x in plist if len(x)<=proper_length]
            try:
                i_=np.random.randint(1,len(plist)-1)   
                if np.random.random_sample()<0.5:
                    plist[i_]+=k.word
                else:
                    plist[i_]=k.word+plist[i_]                     
            except:
                print("no sentence with proper length")                    
        else:
            pos=[]
            for w in words:
                if w.flag==k.flag:
                    pos.append(w.word)
            try:
                re=np.random.choice(pos)                            
                for idx,s in enumerate(plist):
                    plist[idx]=s.replace(re,k.word,1)
                    # print('replace ',re,' to ',k.word)
            except:
                print('switch word exception: ',pos)               
                # re=np.random.choice([x.word for x in words])                            
                # for idx,s in enumerate(plist):
                #     plist[idx]=s.replace(re,k.word,1)
                # print('replace ',re,' to ',k.word)
    return plist

def generate(keyword,mood):    
    # print('generate with keyword= ',keyword)
    category=np.random.choice(all_categories)    
    # print('theme: ',category,'-',title)
    data={}
    data['_theme']=category
    data['_mood']=mood
    m=0
    try:
        m=float(mood)
    except:
        m=0.9    
    if m==0:
        print('human poem!!!')
        p=np.random.choice(human_poem[category])
        p='|'.join(p)
    else:
        plen=0
        print('model loading...')
        rnn=torch.load(os.path.join(dir_name,'conditional-char-rnn.pt'))
        while plen<4:
            ptitle=np.random.choice(category_title[category])
            # print('title=',ptitle)
            p,plen=generate_one(rnn,ptitle,category,m)   
    pstr=''.join(p)
    pstr=generate_time_char(pstr)    
    plist=switchKeyword(keyword,pstr)    
    plist=[x for x in plist if len(x)>0]
    print('/'.join(plist))
    data['_poem']=plist
    jdata=json.dumps(data,ensure_ascii=False).encode('utf-8')
    return jdata

def generateByKeyword(keyword):
    mood=0
    if np.random.randint(20)>1:
        mood=np.random.random()
    print('mood=',mood)
    return generate(keyword,mood)

def tmpGenerate(rnn):
    category=np.random.choice(all_categories)
    ptitle=np.random.choice(category_title[category])
    plen=0
    t_=0
    while plen<4:
        p,plen=generate_one(rnn,ptitle,category,1)
        
    pstr=''.join(p)
    return pstr

# generateByKeyword(sys.argv[1])
