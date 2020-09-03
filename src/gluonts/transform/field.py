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

from collections import Counter
from typing import Any, Dict, List

from gluonts.core.component import validated
from gluonts.dataset.common import DataEntry

from ._base import MapTransformation, SimpleTransformation


class RenameFields(SimpleTransformation):
    """
    Rename fields using a mapping, if source field present.

    Parameters
    ----------
    mapping
        Name mapping `input_name -> output_name`
    """

    @validated()
    def __init__(self, mapping: Dict[str, str]) -> None:
        self.mapping = mapping
        # assert 新的字段名称是否重复出现
        # 此处有两点需要学习：１．collections中的Counter方法　２．new_key先定义，然后在字符串中直接引用
        values_count = Counter(mapping.values())
        for new_key, count in values_count.items():
            assert count == 1, f"Mapped key {new_key} occurs multiple time"

    def transform(self, data: DataEntry):
        for key, new_key in self.mapping.items():
            if key in data:
                # 为啥这样写呢？这么写的好处是啥？
                # no implicit overriding
                assert new_key not in data
                data[new_key] = data[key]
                del data[key]
        return data


class RemoveFields(SimpleTransformation):
    """"
    Remove field names if present.
    删除不需要的字段，是针对DataEntry的操作

    Parameters
    ----------
    field_names
        List of names of the fields that will be removed
    """

    @validated()
    def __init__(self, field_names: List[str]) -> None:
        self.field_names = field_names

    def transform(self, data: DataEntry) -> DataEntry:
        for k in self.field_names:
            # 注意：１．pop的用法　２．因为DataEntry是一个dict，所以直接用pop来删除字段
            data.pop(k, None)
        return data


class SetField(SimpleTransformation):
    """
    Sets a field in the dictionary with the given value.
    设定指定的值

    Parameters
    ----------
    output_field
        Name of the field that will be set
    value
        Value to be set
    """

    @validated()
    def __init__(self, output_field: str, value: Any) -> None:
        self.output_field = output_field
        self.value = value

    def transform(self, data: DataEntry) -> DataEntry:
        data[self.output_field] = self.value
        return data


class SetFieldIfNotPresent(SimpleTransformation):
    """Sets a field in the dictionary with the given value, in case it does not
    exist already.

    Parameters
    ----------
    output_field
        Name of the field that will be set
    value
        Value to be set
    """

    @validated()
    def __init__(self, field: str, value: Any) -> None:
        self.output_field = field
        self.value = value

    def transform(self, data: DataEntry) -> DataEntry:
        if self.output_field not in data.keys():
            data[self.output_field] = self.value
        return data


class SelectFields(MapTransformation):
    """
    Only keep the listed fields

    Parameters
    ----------
    input_fields
        List of fields to keep.
    """

    @validated()
    def __init__(self, input_fields: List[str]) -> None:
        self.input_fields = input_fields

    def map_transform(self, data: DataEntry, is_train: bool) -> DataEntry:
        return {f: data[f] for f in self.input_fields}
