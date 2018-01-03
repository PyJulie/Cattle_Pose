from pylab import *
from PIL import Image
import json
import os
catagory = 1 # 设置成视频文件的名字
path = 'data/train/'
if not os.path.exists('{}.lable'.format(catagory)):
    fp = open("{}.lable".format(catagory), "w")
    fp.write(json.dumps([]))
    fp.close()

fp = open("{}.lable".format(catagory), "r")
start = len(json.loads(fp.read())) # 接着start的位置继续标记
fp.close()
def save(catagory, item):
    handle = open('{}.lable'.format(catagory), 'r+')
    body = handle.read()
    handle.close()
    handle = open('{}.lable'.format(catagory), 'r+')
    in_json = json.loads(body)
    in_json.append(item)
    handle.write(json.dumps(in_json, sort_keys=True,indent =4,separators=(',', ': '), ensure_ascii=True))
    handle.close()
for file in os.listdir(path):
    if os.path.isfile(os.path.join(path, file)) == True:
        im = array(Image.open(os.path.join(path, file)))
        imshow(im)
        x = ginput(16,timeout=100000000)
        save(catagory,x)
        show()
