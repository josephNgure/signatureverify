import os
home =os.stat('file.json').st_size==0
print(home)