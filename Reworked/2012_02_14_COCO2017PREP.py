import json
cocodir = r"C:\Users\Daniel\PycharmProjects\LabellingFilters\Data\COCO raw\annotations_trainval2017\annotations\instances_train2017.json"

with open(cocodir, 'r') as COCO:
    js = json.loads(COCO.read())

image_labels={}

for el in js["annotations"]:
    try:
        image_labels[el["image_id"]].append(el["category_id"])
    except (AttributeError, KeyError):
        image_labels[el["image_id"]] = [el["category_id"]]

num_to_class={}
for el in js["categories"]:
    num_to_class[el["id"]] = [el["name"]]

for el in image_labels:#
    if el.type=='int':
        el=

print("finito")