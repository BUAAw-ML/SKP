"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""
import os
import argparse
from llama_index.retrievers import BaseRetriever, VectorIndexRetriever, KeywordTableSimpleRetriever
from llama_index import ResponseSynthesizer
from llama_index.indices.postprocessor import SimilarityPostprocessor
from tqdm import tqdm
from llama_index.query_engine import RetrieverQueryEngine
from llama_index import LLMPredictor, download_loader, GPTListIndex, ServiceContext, PromptHelper, LangchainEmbedding, GPTVectorStoreIndex
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llm_predictor import HuggingFaceLLMPredictor
from langchain import OpenAI
import torch
from llama_index import SimpleDirectoryReader
from llama_index import Document
from langchain.llms import HuggingFacePipeline
from transformers import pipeline
from transformers import AutoTokenizer, AutoModel
from transformers import T5Tokenizer, T5ForConditionalGeneration
import json
from langchain.chat_models import ChatOpenAI
from llama_index import StorageContext, load_index_from_storage

# sentence-transformers/gtr-t5-large'


import json
from typing import Iterable

from torch.utils.data import Dataset, ConcatDataset
from torch.utils.data.dataloader import default_collate
from easydict import EasyDict
import csv
from tqdm import tqdm
# from lavis.datasets.datasets.data_loader_wrapper import *

def LoadOscarCaptionFeatures(description, module_config=None):
    '''
    Load oscar caption features
    {
        "type": "LoadOscarCaptionFeatures", "option": "default",
        "config": {
            "train": "..",
            "valid": "..",
            "test": "..",
        },
    },
    '''
    ######################
    #   Read caption data
    ######################

    caption_file_path = "data/train_predictions.json",
    caption_features = EasyDict()
 #   for caption_file_path in module_config.config.values():
    with open("data/train_predictions.json", "r") as f:
        caption_data = json.load(f)
        caption_features.update(caption_data)

    print('[Data Statistics] Caption features {}'.format(len(caption_features)))

    return caption_features

def LoadVinVLFeatures(module_config=None):
    '''
    Load vinvl features
    {
        "type": "LoadVinVLFeatures", "option": "default", 
        "config": {
            "train": "..",
            "test": "..",
        },
    },
    '''
    ######################
    #   Read VinVL data
    ######################
    csv.field_size_limit(100000000)
    # print(csv.field_size_limit())

    # vinvl_features = load_cached_data(self.config, 'vinvl_feature_preprocessed')
    # if not self.data.vinvl_features:
    vinvl_features = EasyDict()
    for data_split in ['train', 'test']:
        # Read pre-extracted features
        VinVL_feature_file = "data/scene_graph/vinvl/"+data_split+"/predictions.tsv"
        # logger.info(f'Reading: {VinVL_feature_file}')
        with open(VinVL_feature_file, 'r') as csv_file:
            count_total = sum(1 for row in csv_file)
            csv_file.seek(0)
            read_tsv = csv.reader(csv_file, delimiter="\t")

            for row in tqdm(read_tsv, total=count_total):
                image_key, prediction = row                                            
                prediction = json.loads(prediction)
                # print(prediction.keys())
                vinvl_features[image_key] = prediction
                # for obj in prediction['objects']:
                    # print(obj['rect'])
                #     print(obj['class'])
                #     print(obj['conf'])
                #     print(obj['attributes'])

                # input()
        # save_cached_data(self.config, self.data.vinvl_features, 'vinvl_feature_preprocessed')

    # logger.info('[Data Statistics] VinVL features {}'.format(len(self.data.vinvl_features)))
    return vinvl_features

class BaseDataset(Dataset):
    def __init__(
        self, vis_processor=None, text_processor=None, vis_root=None, ann_paths=[]
    ):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        """
        self.vis_root = vis_root

        self.annotation = []
        for ann_path in ann_paths:
            self.annotation.extend(json.load(open(ann_path, "r")))
        
        # self.annotation = self.annotation[:100]
 
# {
#             'img_caption': item.img_caption,
#             'img_ocr': text_annotations,
#             'question_id':  item.question_id,
#             'question': item.question,
#             'img_key_full': item.img_key_full,
#             'img': item.img,
#             'gold_answer': item.gold_answer,
#             'answers': item.answers,
#             'objects': objects,
#         }

        self.caption_features = EasyDict()
    #   for caption_file_path in module_config.config.values():
        for file_type in ['train','val','test']:

            with open('data/caption/'+file_type+'_predictions.json', "r") as f:
                caption_data = json.load(f)
                self.caption_features.update(caption_data)

        print('[Data Statistics] Caption features {}'.format(len(self.caption_features)))
        
        # vinvl_features = LoadVinVLFeatures()
        # print(len(vinvl_features.keys()))
        #output_filename = "retriever.json"

        retrieve_knowledge = json.load(open("retriever10.json", "r"))

        for idx, item in enumerate(self.annotation):
            
            image_id = int(item['image'].split('_')[-1].split('.')[0])

            description = self.caption_features[str(image_id)][0]['caption']

            
            # imgFilename = 'COCO_' + item['split'] + '_'+ str(item['image_id']).zfill(12) + '.jpg'
            
            # for item in vinvl_features[str(image_id).zfill(12)]['objects']:
            #     description += '#'
            #     description += item['class'] 
            
            # print(description)

            # retrieve_knowledge
            # print(retrieve_knowledge[str(item['question_id'])])
            self.annotation[idx]['passages'] = retrieve_knowledge[str(item['question_id'])]#.split("#")[0]
            # print(self.annotation[idx]['passages'])
            # exit()

            self.annotation[idx]['img_description'] = description

        # self.retriever_data()

        # self.annotation=self.annotation#[:8]
        self.vis_processor = vis_processor
        self.text_processor = text_processor

        self._add_instance_ids()


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

    def retriever_data(self, annotation, model_name='gtr-t5-large', output_filename='/mnt/data/qbwang/lavis/data', top_k=3,
                 max_input_size=4096, num_output=2048, max_chunk_overlap=0.5, chunk_size_limit=1024):
        print(output_filename)
        tokenizer = AutoTokenizer.from_pretrained('/mnt/data/qbwang/public/gtr-t5-large',  local_files_only=True, cache_dir="../public")
        model = AutoModel.from_pretrained('/mnt/data/qbwang/public/gtr-t5-large',  local_files_only=True, cache_dir="../public")

        pipe = pipeline(
            "text2text-generation", model=model, tokenizer=tokenizer,
            max_length=1024, temperature=0, top_p=1, no_repeat_ngram_size=4, early_stopping=True
        )

        llm = HuggingFacePipeline(pipeline=pipe)

        embed_model = LangchainEmbedding(HuggingFaceEmbeddings(model_name='/mnt/data/qbwang/public/gtr-t5-large',))
        llm_predictor = LLMPredictor(llm=llm)

        # llm_predictor = LLMPredictor(llm=ChatOpenAI(temperature=0, model_name="/mnt/data/qbwang/public/gtr-t5-large", max_tokens=1024))

        prompt_helper = PromptHelper(max_input_size, num_output, max_chunk_overlap, chunk_size_limit=chunk_size_limit)
        service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor, embed_model=embed_model, prompt_helper=prompt_helper)
        # service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor, prompt_helper=prompt_helper)
        # with open(file_path, 'r', encoding='utf-8') as file:
        #     json_data = json.load(file)
        
        csv_file = open('/mnt/data/qbwang/lavis/data/okvqa_full_corpus.csv', 'r') 
        count_total = sum(1 for row in csv_file)
        csv_file.seek(0)
        read_tsv = csv.reader(csv_file, delimiter="\t")                           

        results = []
        print("HuggingFaceLLMPredictor")
        documents = []
        for row in tqdm(read_tsv, total=count_total):
            try:
                documents.append(Document(row[0]))
            except:
                print(row)
        
        index = GPTVectorStoreIndex.from_documents(
            documents, service_context=service_context
        )
        
        # SimpleDirectoryReader(input_dir='./data').load_data()
        # if  os.path.exists("./storage2"):
        #     print("1111")
            # storage_context = StorageContext.from_defaults(persist_dir="./storage2")
            # index = load_index_from_storage(storage_context)
        # else:
        #     print("2222")
        # index.storage_context.persist(persist_dir="./storage2")

        retriever = VectorIndexRetriever(
            service_context=service_context,
            index=index,
            similarity_top_k=int(top_k),
        )
        response_synthesizer = ResponseSynthesizer.from_args(
            response_mode="no_text",
            service_context=service_context,
            node_postprocessors=[
                SimilarityPostprocessor(similarity_cutoff=0.1)
            ]
        )

        # retriever = index.as_retriever(retriever_mode='default')
        query_engine = RetrieverQueryEngine(
            retriever=retriever,
            response_synthesizer=response_synthesizer,
        )

        # query_engine = index.as_query_engine() #response_mode="tree_summarize"
        results = {}
        for idx, item in tqdm(enumerate(annotation)):
            response = query_engine.query(item['img_description'])  #item['question']
            text = []
            score = []

            for i in range(len(response.source_nodes)):
                text.append(response.source_nodes[i].node.text)
                score.append(response.source_nodes[i].score)
            # self.annotation[idx]['passages'] = text[0]
  
            results[item['question_id']] = text[0] + "#" + text[1] + "#" + text[2]

        # print(response.source_nodes[0].node.text)

        # result = {
        #     'question': item['question'],
        #     'answers': item['answers'],
        #     'ctxs': text,
        #     'scores': score
        # }
        # results.append(result)
        output_filename = "EvalRetrieveKnowledge.json"

        # a = json.load(open(output_filename, "r"))
        # print(a)
        # exit()

        with open(output_filename, 'w', encoding='utf-8') as output_file:
            json.dump(results, output_file)#, indent=4, ensure_ascii=False)
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







