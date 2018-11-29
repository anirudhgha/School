

# Delete every Kth file in dir_to_clean--------------------------------------------------------------------------
# import os
#
# dir_to_clean = '/home/anirudh/Downloads/skin-cancer-mnist-ham10000/Data'
# l = os.listdir(dir_to_clean)
# k = 2
# for n in l[::k]:
#     target = dir_to_clean + '/' + n
#     if os.path.isfile(target):
#         os.unlink(target)
#
#----------------------------------------------------------------------------------------------------------------
#Move every file of class k from folder1 to folderk-------------------------------------------------------------
import pandas as pd
import shutil
import os

df = pd.read_csv("HAM10000_metadata.csv")
train_label = df['dx']
train_id = df['image_id']

dir_to_move_from = 'D:/Temp/HAM10000_images_part_2'
dir_to_move_to = 'D:/Temp/Data/test/'

filelist = os.listdir(dir_to_move_from)
i = 0
for n in filelist:
    if i % 100 == 0:
        print(i)
    x = os.path.splitext(n)[0]
    for i in range(0, 10015):
        if x == train_id[i]:
            if train_label[i] == 'nv':
                move_from = dir_to_move_from + '/' + n
                move_to = dir_to_move_to + 'nv/' + n
                shutil.move(move_from, move_to)
                continue
            else:
                move_from = dir_to_move_from + '/' + n
                move_to = dir_to_move_to + 'not_nv/' + n
                shutil.move(move_from, move_to)
                continue
    i += 1
