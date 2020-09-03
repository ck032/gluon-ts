# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# A copy of the License is located at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# or in the "license" file accompanying this file. This file is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
# express or implied. See the License for the specific language governing
# permissions and limitations under the License.

# Standard library imports
import abc
from typing import Callable, Iterable, Iterator, List

# First-party imports
from gluonts.core.component import validated
from gluonts.dataset.common import DataEntry
from gluonts.runtime_params import GLUONTS_MAX_IDLE_TRANSFORMS


class Transformation(metaclass=abc.ABCMeta):
    """
    Base class for all Transformations.

    A Transformation processes works on a stream (iterator) of dictionaries.
    """

    @abc.abstractmethod
    def __call__(
        self, data_it: Iterable[DataEntry], is_train: bool
    ) -> Iterator[DataEntry]:
        # 必须要定义__call__，也就是如何对数据进行操作
        pass

    def chain(self, other: "Transformation") -> "Chain":
        return Chain(self, other)

    def __add__(self, other: "Transformation") -> "Chain":
        # 可以拓展transformation
        return self.chain(other)


class Chain(Transformation):
    """
    Chain multiple transformations together.
    """

    @validated()
    def __init__(self, trans: List[Transformation]) -> None:
        # 形成一系列的transformations,是一个list
        self.transformations = []
        for transformation in trans:
            # flatten chains
            if isinstance(transformation, Chain):
                self.transformations.extend(transformation.transformations)
            else:
                self.transformations.append(transformation)

    def __call__(
        self, data_it: Iterable[DataEntry], is_train: bool
    ) -> Iterator[DataEntry]:
        # 对DataEntry做transformation,有序的？
        # 有序的，因为self.tranformations是一个list，list中append是有序的
        # 直接把 data_it、is_train作为输入，注意是data_it，而非单个的DataEntry
        tmp = data_it
        for t in self.transformations:
            tmp = t(tmp, is_train)
        return tmp


class Identity(Transformation):
    # 这个类，在项目中没有发现被用到
    def __call__(
        self, data_it: Iterable[DataEntry], is_train: bool
    ) -> Iterator[DataEntry]:
        return data_it


class MapTransformation(Transformation):
    """
    Base class for Transformations that returns exactly one result per input in the stream.

    为什么叫Map呢？因为是针对data_it中的每个data_entry做操作，yield map_transform后的结果
    举个例子吧？比如特征处理部分，AddConstFeature(MapTransformation)，就是针对每个data_entry做操作
    所以，继承了这个类以后，只需要定义map_transform部分，怎么处理data_entry就好了

    """

    def __call__(
        self, data_it: Iterable[DataEntry], is_train: bool
    ) -> Iterator:
        for data_entry in data_it:
            try:
                yield self.map_transform(data_entry.copy(), is_train)
            except Exception as e:
                raise e

    @abc.abstractmethod
    def map_transform(self, data: DataEntry, is_train: bool) -> DataEntry:
        pass


class SimpleTransformation(MapTransformation):
    """
    Element wise transformations that are the same in train and test mode

    位操作。在train/tet的模式下是相同的。同时有transform、map_tranform方法。两种方法是一样的

    比如删除某类特征 - RemoveFields

    """

    def map_transform(self, data: DataEntry, is_train: bool) -> DataEntry:
        return self.transform(data)

    @abc.abstractmethod
    def transform(self, data: DataEntry) -> DataEntry:
        pass


class AdhocTransform(SimpleTransformation):
    """
    Applies a function as a transformation
    This is called ad-hoc, because it is not serializable.
    It is OK to use this for experiments and outside of a model pipeline that
    needs to be serialized.

    直接把一个函数转化为transformation
    这被称作ad-hoc，因为是不可以序列化的
    可以在model pipeline之外使用

    这个函数只在make_evaluation_predictions中有用到
    """

    def __init__(self, func: Callable[[DataEntry], DataEntry]) -> None:
        self.func = func

    def transform(self, data: DataEntry) -> DataEntry:
        return self.func(data.copy())


class FlatMapTransformation(Transformation):
    """
    Transformations that yield zero or more results per input, but do not combine
    elements from the input stream.

    什么是FlatMap呢？yield - 每个input， yield 0 个或者更多的result，但是并不合并元素。
    举个例子吧？ 比如 covert中的 SampleTargetDim(FlatMapTransformation)，可以把target转化为以下4个部分

    f"past_{self.target_field}",
    f"future_{self.target_field}",
    f"past_{self.observed_values_field}",
    f"future_{self.observed_values_field}",

    """

    def __call__(
        self, data_it: Iterable[DataEntry], is_train: bool
    ) -> Iterator:
        num_idle_transforms = 0
        for data_entry in data_it:
            num_idle_transforms += 1
            try:
                for result in self.flatmap_transform(
                    data_entry.copy(), is_train
                ):
                    num_idle_transforms = 0
                    yield result
            except Exception as e:
                raise e

            # 如果一个transformation并没有真的输出什么结果，那么会报错（默认100）
            # 这样可以避免infinite loops or inefficiencies
            # 详见 GLUONTS_MAX_IDLE_TRANSFORMS 说明
            # 因为是用os.environ.get("GLUONTS_MAX_IDLE_TRANSFORMS", "100") 方法，所以如果没有设定这个参数，会取到100
            if num_idle_transforms > GLUONTS_MAX_IDLE_TRANSFORMS:
                raise Exception(
                    f"Reached maximum number of idle transformation calls.\n"
                    f"This means the transformation looped over "
                    f"GLUONTS_MAX_IDLE_TRANSFORMS={GLUONTS_MAX_IDLE_TRANSFORMS} "
                    f"inputs without returning any output.\n"
                    f"This occurred in the following transformation:\n{self}"
                )

    @abc.abstractmethod
    def flatmap_transform(
        self, data: DataEntry, is_train: bool
    ) -> Iterator[DataEntry]:
        pass


class FilterTransformation(FlatMapTransformation):
    def __init__(self, condition: Callable[[DataEntry], bool]) -> None:
        # condition：一般情况下，传入的lambda函数
        self.condition = condition

    def flatmap_transform(
        self, data: DataEntry, is_train: bool
    ) -> Iterator[DataEntry]:
        if self.condition(data):
            yield data
