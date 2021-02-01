import json

def loadData(path):
    data = {}
    image_list = []
    with open(path,"r") as read_file:
        data = json.load(read_file)
        for scene, frame in enumerate(data):
            image_list.append(frame)
    return data, image_list

def saveData(path,data):
    with open(path, 'w', encoding='utf8') as outfile:
        str_ = json.dumps(data,
                          indent=4, sort_keys=True,
                          separators=(',', ': '), ensure_ascii=False)
        outfile.write(str_)