from detectron2.data.datasets import register_coco_instances
from detectron2.data.datasets.coco import MetadataCatalog, DatasetCatalog
register_coco_instances("grape_train", {}, image_root = "/workspace/grape_label/coco/images", json_file="/workspace/grape_label/coco/train_annotations.json")
register_coco_instances("grape_val", {}, image_root="/workspace/grape_label/coco/images", json_file="/workspace/grape_label/coco/val_annotations.json")
register_coco_instances("plant_train", {}, json_file="/workspace/PlantDoc-Object-Detection-Dataset/plant_annotations_train.json", image_root="/workspace/PlantDoc-Object-Detection-Dataset/TRAIN")
register_coco_instances("plant_val", {}, json_file="/workspace/PlantDoc-Object-Detection-Dataset/plant_annotations_test.json", image_root="/workspace/PlantDoc-Object-Detection-Dataset/TEST")

metas = MetadataCatalog.get("grape_val")
print(metas)
print(DatasetCatalog.get("grape_train"))
# Metadata(evaluator_type='coco', image_root='/workspace/grape_label/coco/images', json_file='/workspace/grape_label/coco/train_annotations.json', name=grape_train,
# thing_classes=['grape'], thing_dataset_id_to_contiguous_id={1: 0})