"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""
import os
import argparse
from llama_index.core.retrievers import VectorIndexRetriever

from llama_index.core import get_response_synthesizer
from llama_index.core.indices.postprocessor import SimilarityPostprocessor
from tqdm import tqdm
from llama_index.core.query_engine import RetrieverQueryEngine

# from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from llama_index.core import SimpleDirectoryReader, Document

from transformers import AutoTokenizer, AutoModel

import json

# from llama_index.llms.llama_cpp import LlamaCPP
# from llama_index.llms.llama_cpp.llama_utils import messages_to_prompt, completion_to_prompt

from llama_index.core import Settings
from llama_index.core import VectorStoreIndex
# sentence-transformers/gtr-t5-large'

from transformers import pipeline

import json
from typing import Iterable

from torch.utils.data import Dataset, ConcatDataset
from torch.utils.data.dataloader import default_collate
from easydict import EasyDict
import csv
from tqdm import tqdm
# from lavis.datasets.datasets.data_loader_wrapper import *

class BaseDataset(Dataset):
    def __init__(
        self, vis_processor=None, text_processor=None, vis_root=None, ann_paths=[]
    ):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file

        'img_caption': item.img_caption,
        'img_ocr': text_annotations,
        'question_id':  item.question_id,
        'question': item.question,
        'img_key_full': item.img_key_full,
        'img': item.img,
        'gold_answer': item.gold_answer,
        'answers': item.answers,
        'objects': objects,
        """
        self.vis_root = vis_root
        # print(ann_paths)
        self.annotation = []

#############################################################
        self.annotation = json.load(open(ann_paths[0]))

        self.vis_processor = vis_processor
        self.text_processor = text_processor

        self._add_instance_ids()

        if 'aokvqa' in ann_paths[0]:
            return

        # for ann_path in ann_paths:
        #     self.annotation.extend(json.load(open(ann_path, "r")))
 
        self.caption_features = EasyDict()
    #   for caption_file_path in module_config.config.values():
        for file_type in ['train','val','test']:

            with open('data/caption/'+file_type+'_predictions.json', "r") as f:
                caption_data = json.load(f)
                self.caption_features.update(caption_data)

        print('[Data Statistics] Caption features {}'.format(len(self.caption_features)))
        
        retrieve_knowledge_num = 10
        knowledge_file = "trainRetriever" + str(retrieve_knowledge_num) + ".json"

        if os.path.exists(knowledge_file):
            retrieve_knowledge = json.load(open(knowledge_file, "r")) 
        # experience_data = json.load(open("experience_vqav2.json", "r"))

        for idx, item in enumerate(self.annotation):
            
            image_id = int(item['image'].split('_')[-1].split('.')[0])

            description = self.caption_features[str(image_id)][0]['caption']

            if os.path.exists(knowledge_file):
                self.annotation[idx]['passages'] = retrieve_knowledge[str(item['question_id'])]

            self.annotation[idx]['img_description'] = description
        
        if not os.path.exists(knowledge_file):
            print('retrieve external knowledge!')
            self.retriever_data(self.annotation, output_filename = 'trainRetriever', top_k=retrieve_knowledge_num)
        # self.obtain_experience_data(self.annotation, json.load(open(ann_paths[-2])), json.load(open(ann_paths[-1])))
        

    def __len__(self):
        return len(self.annotation)

    def collater(self, samples):
        return default_collate(samples)

    def set_processors(self, vis_processor, text_processor):
        self.vis_processor = vis_processor
        self.text_processor = text_processor

    def _add_instance_ids(self, key="instance_id"):
        for idx, ann in enumerate(self.annotation):
            ann[key] = str(idx)

    def retriever_data(self, annotation, model_name='/public/qbwang/public/gtr-t5-large', output_filename='/public/qbwang/SKP/data', top_k=10,
                 max_input_size=4096, num_output=2048, max_chunk_overlap=0.5, chunk_size_limit=1024):

        tokenizer = AutoTokenizer.from_pretrained('/public/qbwang/public/gtr-t5-large',  local_files_only=True, cache_dir="../public")
        model = AutoModel.from_pretrained('/public/qbwang/public/gtr-t5-large',  local_files_only=True, cache_dir="../public")

        pipe = pipeline(
            "text2text-generation", model=model, tokenizer=tokenizer,
            max_length=1024, temperature=0, top_p=1, no_repeat_ngram_size=4, early_stopping=True
        )

        llm = HuggingFacePipeline(pipeline=pipe)
        embed_model = HuggingFaceEmbeddings(model_name=model_name) #, cache_folder=None

        # llm = LlamaCPP(
        #     model_url=None, # We'll load locally.
        #     model_path='./mnt/data/qbwang/public/gtr-t5-large', # 8-bit model
        #     temperature=0,
        #     # max_new_tokens=1024, # Increasing to support longer responses
        #     # context_window=8192, # 8K for a small model!
        #     generate_kwargs={},
        #     # set to at least 1 to use GPU
        #     model_kwargs={"n_gpu_layers": 40}, # 40 was a good amount of layers for the RTX 3090, you may need to decrease yours if you have less VRAM than 24GB
        #     messages_to_prompt=messages_to_prompt,
        #     completion_to_prompt=completion_to_prompt,
        #     verbose=True
        # )

        # llm = AutoModel.from_pretrained('/public/qbwang/public/gtr-t5-large',  local_files_only=True, cache_dir="../public")

        Settings.llm = llm
        Settings.embed_model = embed_model
        Settings.chunk_size=chunk_size_limit # Number of tokens in each chunk
        
        # documents = SimpleDirectoryReader("/public/qbwang/SKP/data/rag_data").load_data()

        csv_file = open('/public/qbwang/SKP/data/rag_data/okvqa_full_corpus.csv', 'r') 
        count_total = sum(1 for row in csv_file)
        csv_file.seek(0)
        read_tsv = csv.reader(csv_file, delimiter="\t")                           

        results = []

        documents = []
        for row in tqdm(read_tsv, total=count_total):
            try:
                documents.append(Document(text = row[0]))
            except:
                print(row) 

        index = VectorStoreIndex.from_documents(documents)

        retriever = VectorIndexRetriever(
            index=index,
            similarity_top_k=int(top_k),
        )
        response_synthesizer = get_response_synthesizer(
           response_mode="no_text",
        )

        query_engine = RetrieverQueryEngine(
            retriever=retriever,
            response_synthesizer=response_synthesizer,
            node_postprocessors=[
                SimilarityPostprocessor(similarity_cutoff=0.1)
            ]
        )

        results = {}
        for idx, item in tqdm(enumerate(annotation)):

            response = query_engine.query(item['img_description'])  #item['question']

            text = []
            score = []
            for i in range(len(response.source_nodes)):
                text.append(response.source_nodes[i].node.text)
                score.append(response.source_nodes[i].score)

            passenges_buf=''
            for passenge in text:
                passenges_buf += (passenge + "#")
  
            results[item['question_id']] = passenges_buf[:-1]

        output_filename = output_filename + str(top_k) + ".json"

        with open(output_filename, 'w', encoding='utf-8') as output_file:
            json.dump(results, output_file)#, indent=4, ensure_ascii=False)
        exit()



    def obtain_experience_data(self, annotation, relate_qustion_annotation, relate_answer_annotation):
        print("obtain_experience_data")
        results = {}

        image_list = [str(int(item['image'].split('_')[-1].split('.')[0])) for item in annotation]
        print(len(image_list))

        truth_answers = {}#EasyDict()
        for item in relate_answer_annotation['annotations']:
            truth_answers[item['question_id']] = [answer['answer'] for answer in item['answers']]


        for question in relate_qustion_annotation['questions']:
            if str(question['image_id']) in image_list:
                buf = question['question'] + ' '
                for answer in list(set(truth_answers[question['question_id']])):
                    buf += (answer + '#')
                buf = buf[:-1]
                if question['image_id'] not in results.keys():
                    results[question['image_id']] = [buf]
                else:
                    results[question['image_id']].append(buf)
                
        print(len(results))
        # print(count)

        with open('eval_experience_vqav2.json', 'w', encoding='utf-8') as output_file:
            json.dump(results, output_file)#, indent=4, ensure_ascii=False)

        # print(image_list)
        exit()



class ConcatDataset(ConcatDataset):
    def __init__(self, datasets: Iterable[Dataset]) -> None:
        super().__init__(datasets)

    def collater(self, samples):
        # TODO For now only supports datasets with same underlying collater implementations

        all_keys = set()
        for s in samples:
            all_keys.update(s)

        shared_keys = all_keys
        for s in samples:
            shared_keys = shared_keys & set(s.keys())

        samples_shared_keys = []
        for s in samples:
            samples_shared_keys.append({k: s[k] for k in s.keys() if k in shared_keys})

        return self.datasets[0].collater(samples_shared_keys)







