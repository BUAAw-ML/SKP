"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

from lavis.datasets.builders.base_dataset_builder import BaseDatasetBuilder

from lavis.common.registry import registry
from lavis.datasets.datasets.aok_vqa_datasets import AOKVQADataset, AOKVQAEvalDataset
from lavis.datasets.datasets.coco_vqa_datasets import COCOVQADataset, COCOVQAEvalDataset, OKVQADataset, OKVQAEvalDataset
from lavis.datasets.datasets.vg_vqa_datasets import VGVQADataset
from lavis.datasets.datasets.gqa_datasets import GQADataset, GQAEvalDataset
# from lavis.datasets.datasets.instruct_datasets import InstructDataset


@registry.register_builder("coco_vqa")
class COCOVQABuilder(BaseDatasetBuilder):
    train_dataset_cls = COCOVQADataset
    eval_dataset_cls = COCOVQAEvalDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/coco/defaults_vqa.yaml",
        "eval": "configs/datasets/coco/eval_vqa.yaml",
    }


@registry.register_builder("vg_vqa")
class VGVQABuilder(BaseDatasetBuilder):
    train_dataset_cls = VGVQADataset
    DATASET_CONFIG_DICT = {"default": "configs/datasets/vg/defaults_vqa.yaml"}


@registry.register_builder("ok_vqa")
class OKVQABuilder(BaseDatasetBuilder):
    train_dataset_cls = OKVQADataset
    eval_dataset_cls = OKVQAEvalDataset
    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/okvqa/defaults.yaml",
    }

# @registry.register_builder("instruct")
# class INSTRUCTBuilder(BaseDatasetBuilder):
#     train_dataset_cls = InstructDataset
#     eval_dataset_cls = OKVQAEvalDataset
#     DATASET_CONFIG_DICT = {
#         "default": "configs/datasets/kbvqa_instruct/defaults.yaml",
#     }


@registry.register_builder("aok_vqa")
class AOKVQABuilder(BaseDatasetBuilder):
    train_dataset_cls = AOKVQADataset
    eval_dataset_cls = AOKVQAEvalDataset

    DATASET_CONFIG_DICT = {"default": "configs/datasets/aokvqa/defaults.yaml"}


@registry.register_builder("gqa")
class GQABuilder(BaseDatasetBuilder):
    train_dataset_cls = GQADataset
    eval_dataset_cls = GQAEvalDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/gqa/defaults.yaml",
        "balanced_val": "configs/datasets/gqa/balanced_val.yaml",
        "balanced_testdev": "configs/datasets/gqa/balanced_testdev.yaml",
    }