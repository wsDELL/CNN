import os
import time

os.system("start cmd.exe cmd /k python server.py")
time.sleep(5)
os.system("start cmd.exe cmd /k python worker1.py")
os.system("start cmd.exe cmd /k python worker6.py")
# os.system("start cmd.exe cmd /k python worker7.py")
# os.system("start cmd.exe cmd /k python worker8.py")
# os.system("start cmd.exe cmd /k python worker9.py")
