"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import os
import json
import torch

from PIL import Image

from lavis.datasets.datasets.vqa_datasets import VQADataset, VQAEvalDataset

from collections import OrderedDict

from easydict import EasyDict

class __DisplMixin:
    def displ_item(self, index):
        sample, ann = self.__getitem__(index), self.annotation[index]

        return OrderedDict(
            {
                "file": ann["image"],
                "question": ann["question"],
                "question_id": ann["question_id"],
                "answers": "; ".join(ann["answer"]),
                "image": sample["image"],
            }
        )


class COCOVQADataset(VQADataset, __DisplMixin):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)

    def __getitem__(self, index):
        ann = self.annotation[index]

        image_path = os.path.join(self.vis_root, ann["image"])
        image = Image.open(image_path).convert("RGB")

        image = self.vis_processor(image)
        question = self.text_processor(ann["question"])

        answer_weight = {}
        for answer in ann["answer"]:
            if answer in answer_weight.keys():
                answer_weight[answer] += 1 / len(ann["answer"])
            else:
                answer_weight[answer] = 1 / len(ann["answer"])

        answers = list(answer_weight.keys())
        weights = list(answer_weight.values())

        return {
            "image": image,
            "text_input": question,
            "answers": answers,
            "weights": weights,
        }
class OKVQADataset(VQADataset, __DisplMixin):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)

    def collater(self, samples):
        image_list, question_list, answer_list, weight_list = [], [], [], []
        gold_answer = []
        passages_list = []
        experience_list = []

        num_answers = []

        for sample in samples:
            image_list.append(sample["image"])
            question_list.append(sample["text_input"])
            passages_list.append(sample["passages"])
            # experience_list.append(sample["experiences"])

            weight_list.extend(sample["weights"])

            answers = sample["answers"]
            gold_answer.append(sample["answer"])

            # answer_list.extend(answers)
            answer_list.append(sample["answers"])
            num_answers.append(len(answers))

        return {
            "image": torch.stack(image_list, dim=0),
            "text_input": question_list,
            "passages": passages_list,
            # "experiences": experience_list,
            "answer": answer_list,
            "gold_answer": gold_answer,
            "weight": torch.Tensor(weight_list),
            "n_answers": torch.LongTensor(num_answers),
        }

    def __getitem__(self, index):
        ann = self.annotation[index]

        image_path = os.path.join(self.vis_root, ann["image"])
        image = Image.open(image_path).convert("RGB")

        image = self.vis_processor(image)
        question = self.text_processor(ann["question"])

        passages = ann["passages"]
        # experiences = ann["experiences"]

        answer_weight = {}
        for answer in ann["answer"]:
            if answer in answer_weight.keys():
                answer_weight[answer] += 1 / len(ann["answer"])
            else:
                answer_weight[answer] = 1 / len(ann["answer"])

        answers = list(answer_weight.keys())
        weights = list(answer_weight.values())

        answer = answers[weights.index(max(weights))]

        return {
            "image": image,
            "text_input": question,
            "passages": passages,
            # "experiences":experiences,
            "answers": answers,
            "weights": weights,
            "answer": answer,
        }


class COCOVQAEvalDataset(VQAEvalDataset, __DisplMixin):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        """

        self.vis_root = vis_root

        self.annotation = json.load(open(ann_paths[0]))

        answer_list_path = ann_paths[1]
        if os.path.exists(answer_list_path):
            self.answer_list = json.load(open(answer_list_path))
        else:
            self.answer_list = None

        try:
            self.coco_fmt_qust_file = ann_paths[2]
            self.coco_fmt_anno_file = ann_paths[3]
        except IndexError:
            self.coco_fmt_qust_file = None
            self.coco_fmt_anno_file = None

        self.vis_processor = vis_processor
        self.text_processor = text_processor

        self._add_instance_ids()

    def __getitem__(self, index):
        ann = self.annotation[index]

        image_path = os.path.join(self.vis_root, ann["image"])
        image = Image.open(image_path).convert("RGB")

        image = self.vis_processor(image)
        question = self.text_processor(ann["question"])

        return {
            "image": image,
            "text_input": question,
            "question_id": ann["question_id"],
            "instance_id": ann["instance_id"],
        }

class OKVQAEvalDataset(VQAEvalDataset, __DisplMixin):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        """

        self.vis_root = vis_root

        self.annotation = json.load(open(ann_paths[0]))
        
        ###################

        self.caption_features = EasyDict()
    #   for caption_file_path in module_config.config.values():
        # experience_data = json.load(open("eval_experience_vqav2.json", "r"))

        knowledge_file = "evalRetriever10.json"  #EvalRetrieveKnowledge #evalRetriever10_que+imgdes.json

        if os.path.exists(knowledge_file):
            retrieve_knowledge = json.load(open(knowledge_file, "r", encoding='GBK'))
        else:
            for file_type in ['train','val','test']:
                with open('data/caption/'+file_type+'_predictions.json', "r") as f:
                    caption_data = json.load(f)
                    self.caption_features.update(caption_data)
# eval_experience_vqav2
        for idx, item in enumerate(self.annotation):
            
            image_id = int(item['image'].split('_')[-1].split('.')[0])
            
            if os.path.exists(knowledge_file):
                # print(retrieve_knowledge[str(item['question_id'])])
                self.annotation[idx]['passages'] = retrieve_knowledge[str(item['question_id'])]#.split("#")[0]
                # if str(int(image_id)) in experience_data.keys():
                #     self.annotation[idx]['experiences'] = experience_data[str(int(image_id))]
                # else:
                #     self.annotation[idx]['experiences'] = self.caption_features[str(image_id)][0]['caption']

                # print(self.annotation[idx]['passages'])
                # exit()
            else:
                self.annotation[idx]['img_description'] =  self.caption_features[str(image_id)][0]['caption']
            # print(self.annotation[idx]['img_description'])

        if not os.path.exists(knowledge_file):
            print('retrieve external knowledge!')
            self.retriever_data(self.annotation)
        ####################
            
        # self.obtain_experience_data(self.annotation, json.load(open(ann_paths[-2])), json.load(open(ann_paths[-1])))


        answer_list_path = ann_paths[1]
 
        if os.path.exists(answer_list_path):
            self.answer_list = json.load(open(answer_list_path))
        else:
            self.answer_list = None



        try:
            self.coco_fmt_qust_file = ann_paths[2]
            self.coco_fmt_anno_file = ann_paths[3]
        except IndexError:
            self.coco_fmt_qust_file = None
            self.coco_fmt_anno_file = None

        self.vis_processor = vis_processor
        self.text_processor = text_processor

        self._add_instance_ids()

    def collater(self, samples):
        image_list, question_list, question_id_list, instance_id_list = [], [], [], []
        passages_list = []
        answers_list = []
        num_answers = []
        experience_list = []

        for sample in samples:
            image_list.append(sample["image"])
            question_list.append(sample["text_input"])
            passages_list.append(sample["passages"])
            question_id_list.append(sample["question_id"])
            instance_id_list.append(sample["instance_id"])
            answers_list.append(sample["answers"])
            # experience_list.append(sample["experiences"])


        return {
            "image": torch.stack(image_list, dim=0),
            "text_input": question_list,
            "passages": passages_list,
            "question_id": torch.tensor(question_id_list),
            "instance_id": instance_id_list,
            "answers_list": answers_list,
            # "experiences":experience_list,
        }

    def __getitem__(self, index):
        ann = self.annotation[index]

        image_path = os.path.join(self.vis_root, ann["image"])
        image = Image.open(image_path).convert("RGB")

        image = self.vis_processor(image)
        question = self.text_processor(ann["question"])

        # passages = ""#ann["passages"]
        passages = ann["passages"]
        answers = ann["answer"]
        # experiences = ann["experiences"]

        return {
            "image": image,
            "text_input": question,
            "passages": passages,
            "question_id": ann["question_id"],
            "instance_id": ann["instance_id"],
            "answers": answers,
            # "experiences":experiences,
        }
