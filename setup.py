import os
import time

os.system("start cmd.exe cmd /k python server1.py")
time.sleep(5)
os.system("start cmd.exe cmd /k python worker11.py")
os.system("start cmd.exe cmd /k python worker12.py")
# os.system("start cmd.exe cmd /k python worker13.py")
# os.system("start cmd.exe cmd /k python worker14.py")
# os.system("start cmd.exe cmd /k python worker7.py")
# os.system("start cmd.exe cmd /k python worker8.py")
# os.system("start cmd.exe cmd /k python worker9.py")
