import sys
import random
import time

from pythonosc import osc_message_builder
from pythonosc import udp_client

from pythonosc import dispatcher
from pythonosc import osc_server

from generate import *

client_ip=['192.168.2.171','192.168.2.172','192.168.2.211']
keyjson=io.open(os.path.join(data_path,'keyword.json')).read();
keyword=json.loads(keyjson)

pdelay=0
pfadein=800
pshow=10000
pfadeout=3000

def sendMessage(plist,ip,tdelay,tin,tshow,tout):
  poem='|'.join(plist)
  poem+='|'
  msg=osc_message_builder.OscMessageBuilder(address='/poem')
  msg.add_arg(poem.encode('utf-8'),arg_type='b')
  msg.add_arg(tdelay,arg_type='i') #delay
  msg.add_arg(tin,arg_type='i') #in
  msg.add_arg(tshow,arg_type='i') #show
  msg.add_arg(tout,arg_type='i') #out

  msg=msg.build()
  # print(msg)
  client = udp_client.SimpleUDPClient(ip,12345)
  client.send(msg)
  

def face_handler(unused_addr,args):
  key_=np.random.choice(keyword)
  print(key_)
  data=generate(key_,0.9)
  jdata=json.loads(data)

  plist=jdata['_poem']  
  sendMessage(plist[0:4],client_ip[0],pdelay,pfadein*4,pshow+pfadein*3,pfadeout)
  sendMessage(plist[4:6],client_ip[1],pdelay+pfadein*4,pfadein*2,pshow+pfadein*1,pfadeout)
  sendMessage(plist[6:],client_ip[2],pdelay+pfadein*6,pfadein,pshow,pfadeout)


dispatcher=dispatcher.Dispatcher()
dispatcher.map("/face", face_handler)

server=osc_server.ThreadingOSCUDPServer(('127.0.0.1',12345),dispatcher)
print("Serving on {}".format(server.server_address))
server.serve_forever()