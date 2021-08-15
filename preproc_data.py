# CREATE A SMALL VALIDATION SET
import os
import shutil
import json


def main():
    data_path = './data/coco_crops_few_shot/test'
    folders = os.listdir(data_path)

    val_path = './data/coco_crops_few_shot/val'
    os.mkdir(val_path)

    for folder in folders:
        if not os.path.exists(os.path.join(val_path, folder)):
            os.mkdir(os.path.join(val_path, folder))

        imgs = os.listdir(os.path.join(data_path, folder))

        for img in imgs[:10]:
            shutil.move(os.path.join(data_path, folder, img), os.path.join(val_path, folder, img))

    print("Val set with 10 images per class created from Test set!")

    splits = ["train", "test", "val"]

    for split in splits:
        class_names = os.listdir("./data/coco_crops_few_shot/" + split)

        ann_dict = {}
        ann_dict["class_names"] = class_names
        ann_dict["class_roots"] = ["./data/coco_crops_few_shot/" + split + "/" + str(i) for i in class_names]

        with open("./data/" + split + ".json", "w") as fp:
            json.dump(ann_dict, fp)

    print("Json files generation complete!")


if __name__ == "__main__":
    main()
