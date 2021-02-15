import json

cocodir = r"C:\Users\Daniel\PycharmProjects\LabellingFilters\Data\COCO raw\annotations_trainval2017\annotations\person_keypoints_train2017.json"

with open(cocodir, 'r') as COCO:
    js = json.loads(COCO.read())

print("Test")