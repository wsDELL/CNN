#!/bin/bash
screen -dmS "server" python server.py
screen -dmS "worker1" python worker1.py
screen -dmS "worker6" python worker6.py