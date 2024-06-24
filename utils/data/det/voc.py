import glob
import os
import xml.etree.ElementTree as ET
from typing import List, Union

import numpy as np
from prettytable import PrettyTable
from tqdm import tqdm
class VOCParaser:
    def __init__(self, root_dir, image_set='train', data_split: Union[str, List] = None):
        if data_split is None:
            data_split = ['train.txt', 'val.txt']
        if isinstance(data_split, str):
            data_split = [data_split]
        self.root_dir = root_dir
        self.image_set = image_set
        self.image_dir = os.path.join(root_dir, 'JPEGImages')
        self.annotation_dir = os.path.join(root_dir, 'Annotations')
        self.image_set_file = [os.path.join(root_dir, 'ImageSets', 'Main', f) for f in data_split]
        self.image_list = self._read_image_set()

    def _read_image_set(self):
        image_list = []
        for image_set_file in self.image_set_file:
            with open(image_set_file) as f:
                image_list.extend(f.read().strip().split())
        return image_list

    def _load_annotation(self, index):
        annotation_file = os.path.join(self.annotation_dir, f'{index}.xml')
        tree = ET.parse(annotation_file)
        root = tree.getroot()

        bboxes = []
        labels = []

        for obj in root.findall('object'):
            name = obj.find('name').text
            bndbox = obj.find('bndbox')
            bbox = [
                int(bndbox.find('xmin').text),
                int(bndbox.find('ymin').text),
                int(bndbox.find('xmax').text),
                int(bndbox.find('ymax').text)
            ]
            bboxes.append(bbox)
            labels.append(name)

        return np.array(bboxes), labels

    def get_annotations(self):
        annotations = []
        for index in self.image_list:
            bboxes, labels = self._load_annotation(index)
            annotations.append({
                'index': index,
                'bboxes': bboxes,
                'labels': labels
            })
        return annotations

    def get_stats(self):
        num_images = len(self.image_list)
        num_objects = 0
        class_counts = {}
        process_bar = tqdm(self.image_list, desc='Analyzing dataset')
        for index in process_bar:
            _, labels = self._load_annotation(index)
            num_objects += len(labels)
            for label in labels:
                if label not in class_counts:
                    class_counts[label] = 0
                class_counts[label] += 1

        return {
            'num_images': num_images,
            'num_objects': num_objects,
            'class_counts': class_counts
        }

    def show_data_info(self):
        stats = self.get_stats()
        table_view = PrettyTable()
        table_view.title = 'Dataset Info'
        table_view.field_names = ['Image Number', 'Object Number', 'Class Number']
        table_view.add_row([stats['num_images'], stats['num_objects'], len(stats['class_counts'])])

        table_cls = PrettyTable()
        table_cls.title = 'Class Info'
        table_cls.field_names = ['Class', 'Count']
        for cls, count in stats['class_counts'].items():
            table_cls.add_row([cls, count])
        # 按照类别数量排序
        table_cls.sortby = 'Count'
        table_cls.reversesort = True


        print(table_view)
        print(table_cls)


    def get_image_path(self, index):
        return os.path.join(self.image_dir, f'{index}.jpg')

    def get_image(self, index):
        from PIL import Image
        image_path = self.get_image_path(index)
        return Image.open(image_path)

    def add_difficult_to_annotations(self, annotation_file=None) -> None:
        """
        为没有 difficult 元素的项添加 difficult 元素。

        Args:
            annotation_file (str): XML 注释文件的路径。
        """
        if annotation_file is None:
            process_bar = tqdm(self.image_list, desc='Adding difficult flag')
        else:
            assert os.path.exists(annotation_file), f'Annotation file {annotation_file} does not exist.'
            process_bar = [os.path.splitext(os.path.basename(annotation_file))[0]]
        for index in process_bar:
            annotation_file = os.path.join(self.annotation_dir, f'{index}.xml')
            tree = ET.parse(annotation_file)
            root = tree.getroot()

            for obj in root.findall('object'):
                difficult = obj.find('difficult')
                if difficult is None:
                    difficult = ET.SubElement(obj, 'difficult')
                    difficult.text = '0'  # 默认值设为0（未标记为困难）
            tree.write(annotation_file)
    def ignore_class(self, ignore_classes: list) -> None:
        """
        忽略指定类别的对象。

        Args:
            class_name (str): 要忽略的类别名称。
        """
        filtered_image_list = []
        process_bar = tqdm(self.image_list, desc='Filtering classes')
        for index in process_bar:
            bboxes, labels = self._load_annotation(index)
            filtered_bboxes = []
            filtered_labels = []
            for bbox, label in zip(bboxes, labels):
                if label not in ignore_classes:
                    filtered_bboxes.append(bbox)
                    filtered_labels.append(label)
            if filtered_bboxes:  # 只保留包含其他类别的图像
                filtered_image_list.append(index)
                # 更新注释文件，去除指定类别
                self._update_annotation(index, filtered_bboxes, filtered_labels)

    def rename_class(self, old_class_name: str, new_class_name: str) -> None:
        """
        重命名指定类别的对象。

        Args:
            old_class_name (str): 要重命名的旧类别名称。
            new_class_name (str): 新的类别名称。
        """
        process_bar = tqdm(self.image_list, desc='Renaming classes')
        for index in process_bar:
            bboxes, labels = self._load_annotation(index)
            updated_labels = [new_class_name if label == old_class_name else label for label in labels]
            self._update_annotation(index, bboxes, updated_labels)

    def remove_backup_xml(self) -> None:
        """
        删除备份的注释文件。
        """
        backup_files = glob.glob(os.path.join(self.annotation_dir, '*.bak*'))
        if backup_files:
            process_bar = tqdm(backup_files, desc='Removing backup files')
            for file in process_bar:
                os.remove(file)

    def _update_annotation(self, index, bboxes, labels) -> None:
        """
        更新注释文件，去除指定类别。

        Args:
            index (str): 图像索引。
            bboxes : 过滤后的边界框。
            labels : 过滤后的标签。
        """
        annotation_file = os.path.join(self.annotation_dir, f'{index}.xml')
        tree = ET.parse(annotation_file)
        root = tree.getroot()

        # 删除原有的object元素
        for obj in root.findall('object'):
            root.remove(obj)

        # 添加新的object元素
        for bbox, label in zip(bboxes, labels):
            obj = ET.SubElement(root, 'object')
            name = ET.SubElement(obj, 'name')
            name.text = label
            pose = ET.SubElement(obj, 'pose')
            pose.text = 'Unspecified'
            truncated = ET.SubElement(obj, 'truncated')
            truncated.text = '0'
            difficult = ET.SubElement(obj, 'difficult')
            difficult.text = '0'
            bndbox = ET.SubElement(obj, 'bndbox')
            xmin = ET.SubElement(bndbox, 'xmin')
            xmin.text = str(bbox[0])
            ymin = ET.SubElement(bndbox, 'ymin')
            ymin.text = str(bbox[1])
            xmax = ET.SubElement(bndbox, 'xmax')
            xmax.text = str(bbox[2])
            ymax = ET.SubElement(bndbox, 'ymax')
            ymax.text = str(bbox[3])
        # 备份原始注释文件
        bak_count = len(glob.glob(f'{annotation_file}.bak*'))
        os.rename(annotation_file, f'{annotation_file}.bak{bak_count}')
        tree.write(annotation_file)


