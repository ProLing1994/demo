import os

def get_filelist(path, typefile=['.png']):
    Filelist = []
    for home, dirs, files in os.walk(path):
        for filename in files:
            Filelist.append(os.path.join(home, filename))
    Filelist = [i.replace('\\', '/') for i in Filelist]
    Filelist = [i for i in Filelist if os.path.splitext(i)[-1] in typefile]
    return Filelist

li = get_filelist('/lirui/DATA/SchoolBusSeg/SchoolBusSeg/base', ['.jpg'])

print(len(li))
