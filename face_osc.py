import sys
import random
import time

from pythonosc import osc_message_builder
from pythonosc import udp_client

from pythonosc import dispatcher
from pythonosc import osc_server


msg=osc_message_builder.OscMessageBuilder(address='/face')
msg.add_arg(1,arg_type='f')
msg=msg.build()

client = udp_client.SimpleUDPClient('127.0.0.1',12345)
client.send(msg)


