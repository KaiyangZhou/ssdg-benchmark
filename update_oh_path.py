"""
Correct the image paths of OfficeHome
e.g. change 'clipart/train/000_Alarm_Clock/000013.jpg'
to 'clipart/train/Alarm_Clock/000013.jpg'

Please create a backup (copy) for the old files in case
something goes wrong.
"""
import os
import json
from dassl.utils import read_json, write_json


root = "path/to/splits_ssdg"
names = os.listdir(root)

for name in names:
    filepath = os.path.join(root, name)
    split = read_json(filepath)
    train_x = split["train_x"]
    train_u = split["train_u"]
    train_x_new, train_u_new = [], []
    
    for path, label, domain in train_x:
        elements = path.split("/")
        elements[2] = elements[2][4:]
        path = "/".join(elements)
        train_x_new.append((path, label, domain))
    
    for path, label, domain in train_u:
        elements = path.split("/")
        elements[2] = elements[2][4:]
        path = "/".join(elements)
        train_u_new.append((path, label, domain))
    
    output = {
        "train_x": train_x_new,
        "train_u": train_u_new
    }

    # save to the same path
    write_json(output, filepath)
