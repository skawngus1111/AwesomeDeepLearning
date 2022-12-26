import os

from torch.utils.data import Dataset, DataLoader

import cv2
import numpy as np
import matplotlib.pyplot as plt
from pycocotools.coco import COCO
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection

class COCODetection(Dataset) :
    def __init__(self, root_dir, image_set, transform=None, viz=False):
        super(COCODetection, self).__init__()

        self.root_dir = root_dir
        self.image_set = image_set
        self.transform = transform
        self.viz = viz

        self.coco = COCO(os.path.join(self.root_dir, 'annotations', 'instances_' + self.image_set + '.json'))
        whole_image_ids = self.coco.getImgIds()
        self.image_ids = []
        # to remove not annotated image idx
        self.no_anno_list = []
        for idx in whole_image_ids:
            flag = 1
            annotations_ids = self.coco.getAnnIds(imgIds=idx, iscrowd=False)
            coco_annotations = self.coco.loadAnns(annotations_ids)
            if len(annotations_ids) == 0 : self.no_anno_list.append(idx)
            else:
                for anns in coco_annotations :
                    if anns['category_id'] in [12, 26, 29, 30, 45, 66, 68, 69, 71, 83, 91] :
                        self.no_anno_list.append(idx)
                        flag = 0
                        break
                if flag :
                    self.image_ids.append(idx)

        self.load_classes()

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image = self.load_image(idx)
        annot = self.load_annotations(idx)

        if self.viz :
            self.showAnns(image, annot)
            plt.show()

        return image, annot

    def load_image(self, image_index):
        image_info = self.coco.loadImgs(self.image_ids[image_index])[0]
        image_path = os.path.join(self.root_dir, self.image_set, image_info['file_name'])
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        return img.astype(np.float32) / 255.

    def load_annotations(self, image_index):
        # get ground truth annotations
        annotations_ids = self.coco.getAnnIds(imgIds=self.image_ids[image_index], iscrowd=False)
        annotations = np.zeros((0, 6))

        # some images appear to miss annotations
        if len(annotations_ids) == 0:
            return annotations

        # parse annotations
        coco_annotations = self.coco.loadAnns(annotations_ids)
        for idx, a in enumerate(coco_annotations):

            # some annotations have basically no width / height, skip them
            if a['bbox'][2] < 1 or a['bbox'][3] < 1:
                continue

            annotation = np.zeros((1, 6))
            annotation[0, :4] = a['bbox']
            annotation[0, 4] =  self.classes[self.labels[a['category_id']]]
            annotation[0, 5] = a['category_id']

            annotations = np.append(annotations, annotation, axis=0)

        # transform from [x, y, w, h] to [x1, y1, x2, y2]
        annotations[:, 2] = annotations[:, 0] + annotations[:, 2]
        annotations[:, 3] = annotations[:, 1] + annotations[:, 3]

        return annotations

    def load_classes(self):
        # load class names (name -> label)
        categories = self.coco.loadCats(self.coco.getCatIds())
        categories.sort(key=lambda x: x['id'])

        self.viz_classes, self.classes = {}, {}
        for c in categories:
            self.viz_classes[c['name']] = c['id']
            self.classes[c['name']] = len(self.classes)

        # also load the reverse (label -> name)
        self.labels = {}
        for key, value in self.viz_classes.items():
            self.labels[value] = key

    def showAnns(self, image, anns):
        ax = plt.gca()
        ax.imshow(image)
        ax.set_autoscale_on(False)
        polygons, colors = [], []

        for ann in anns:
            bbox_x1, bbox_y1, bbox_x2, bbox_y2, _, category_id = ann
            c = (np.random.random((1, 3)) * 0.6 + 0.4).tolist()[0]
            poly = [[bbox_x1, bbox_y1], [bbox_x1, bbox_y2], [bbox_x2, bbox_y2], [bbox_x2, bbox_y1]]
            np_poly = np.array(poly).reshape((4, 2))
            polygons.append(Polygon(np_poly))
            colors.append(c)
            ax.text(bbox_x1, bbox_y1, self.labels[category_id], color=c)

        p = PatchCollection(polygons, facecolor='none', edgecolors=colors, linewidths=2)
        ax.add_collection(p)

if __name__ == '__main__' :
    # root_dir = DATASET_DIR
    coco = COCODetection(root_dir, image_set='train2017', viz=True)
    loader = DataLoader(coco, shuffle=False)
    for idx, (image, target) in enumerate(loader) :
        print("image")