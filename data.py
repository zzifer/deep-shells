import os
import numpy as np
import torch
from shape_utils import *
from typing import List


def input_to_batch(mat_dict):
    dict_out = dict()

    for attr in ["vert", "triv", "evecs", "evals", "SHOT"]:
        # 如果mat_dict[attr][0]的数据类型是整数类型，将其转换为np.int32类型
        # 否则，将mat_dict[attr][0]转换为np.float32类型
        if mat_dict[attr][0].dtype.kind in np.typecodes["AllInteger"]:
            dict_out[attr] = np.asarray(mat_dict[attr][0], dtype=np.int32)
        else:
            dict_out[attr] = np.asarray(mat_dict[attr][0], dtype=np.float32)

    for attr in ["A"]:
        # 将mat_dict[attr][0]的对角线元素提取出来，并转换为np.float32类型
        dict_out[attr] = np.asarray(mat_dict[attr][0].diagonal(), dtype=np.float32)

    return dict_out


def batch_to_shape(batch):
    shape = Shape(batch["vert"].squeeze().to(device), batch["triv"].squeeze().to(device, torch.long) - 1)

    for attr in ["evecs", "evals", "SHOT", "A"]:
        setattr(shape, attr, batch[attr].squeeze().to(device))

    shape.compute_xi_()

    return shape


class ShapeDatasetOnePair(torch.utils.data.Dataset):
    def __init__(self, file_name_1, file_name_2=None):

        load_data = scipy.io.loadmat(file_name_1)

        self.data_x = input_to_batch(load_data["X"][0])

        if file_name_2 is None:
            self.data_y = input_to_batch(load_data["Y"][0])
            print("Loaded file ", file_name_1, "")
        else:
            load_data = scipy.io.loadmat(file_name_2)
            self.data_y = input_to_batch(load_data["X"][0])
            print("Loaded files ", file_name_1, " and ", file_name_2)

    def _get_index(self, i):
        return i

    def __getitem__(self, index):
        data_curr = dict()
        if index == 0:
            data_curr["X"] = self.data_x
            data_curr["Y"] = self.data_y
        else:
            data_curr["X"] = self.data_y
            data_curr["Y"] = self.data_x
        return data_curr

    def __len__(self):
        return 2


class ShapeDatasetCombine(torch.utils.data.Dataset):
    def __init__(self, file_fct, num_shapes):
        self.file_fct = file_fct
        self.num_shapes = num_shapes
        self.num_pairs = num_shapes ** 2

        self.data = []

        self._init_data()

    def _init_data(self):
        for i in range(self.num_shapes):
            file_name = self.file_fct(self._get_index(i))
            load_data = scipy.io.loadmat(file_name)

            data_curr = input_to_batch(load_data["X"][0])

            self.data.append(data_curr)

            print("Loaded file ", file_name, "")

    def _get_index(self, i):
        return i

    def __getitem__(self, index):
        i1 = int(index / self.num_shapes)
        i2 = int(index % self.num_shapes)
        data_curr = dict()
        data_curr["X"] = self.data[i1]
        data_curr["Y"] = self.data[i2]
        return data_curr

    def __len__(self):
        return self.num_pairs


class ShapeDatasetCombineMulti(ShapeDatasetCombine):
    def __init__(self, datasets: List[ShapeDatasetCombine]):
        self.datasets = datasets
        num_shapes = sum([d.num_shapes for d in datasets])
        super().__init__(None, num_shapes)

    def _init_data(self):
        for d in self.datasets:
            self.data += d.data


def get_faustremeshed_file(i):
    # faustremeshed文件所在的文件夹路径
    folder_path = ""
    assert folder_path != "", "Specify the location of FAUST remeshed"
    # 获取 folder_path 文件夹中所有的文件名，并将它们存储在列表 faust_files
    faust_files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
    # 对获取的文件名进行排序
    faust_files.sort()
    # 使用 os.path.join 函数，根据指定的索引 i 获取第 i 个文件名，并返回完整的文件路径
    return os.path.join(folder_path, faust_files[i])


def get_scaperemeshed_file(i):
    folder_path = ""
    assert folder_path != "", "Specify the location of SCAPE remeshed"
    scape_files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
    scape_files.sort()
    return os.path.join(folder_path, scape_files[i])

class Faustremeshed_train(ShapeDatasetCombine):
    def __init__(self):
        super().__init__(get_faustremeshed_file, 80)
        print("loaded FAUST_remeshed with " + str(self.num_pairs) + " pairs")


class Scaperemeshed_train(ShapeDatasetCombine):
    def __init__(self):
        super().__init__(get_scaperemeshed_file, 51)
        print("loaded SCAPE_remeshed with " + str(self.num_pairs) + " pairs")


# 这个类通过继承ShapeDatasetCombineMulti类来加载组合了FAUST数据集和Scape数据集中remeshed后的训练数据
class FaustScapeRemeshedTrain(ShapeDatasetCombineMulti):
    def __init__(self):
        super().__init__([Faustremeshed_train(), Scaperemeshed_train()])
        print("loaded FaustScapeRemeshedTrain with " + str(self.num_pairs) + " pairs")

