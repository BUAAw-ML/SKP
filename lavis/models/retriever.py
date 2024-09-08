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

def retriever_data(file_path, model_name, output_filename, top_k=5,
                 max_input_size=4096, num_output=2048, max_chunk_overlap=0.5, chunk_size_limit=1024):
    print(output_filename)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    pipe = pipeline(
        "text2text-generation", model=model, tokenizer=tokenizer,
        max_length=1024, temperature=0, top_p=1, no_repeat_ngram_size=4, early_stopping=True
    )

    llm = HuggingFacePipeline(pipeline=pipe)

    embed_model = LangchainEmbedding(HuggingFaceEmbeddings(model_name=model_name))
    llm_predictor = LLMPredictor(llm=llm)

    prompt_helper = PromptHelper(max_input_size, num_output, max_chunk_overlap, chunk_size_limit=chunk_size_limit)
    service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor, embed_model=embed_model, prompt_helper=prompt_helper)

    with open(file_path, 'r', encoding='utf-8') as file:
        json_data = json.load(file)
    results = []
    print("HuggingFaceLLMPredictor")
    for item in tqdm(json_data):
        text_list = []
        for crtx in item['ctxs']:
            text_list.append(crtx['text'])
        documents = [Document(t) for t in crtx['text']]

        index = GPTVectorStoreIndex.from_documents(
            documents, service_context=service_context
        )

        #index.storage_context.persist(persist_dir="./storage2")

        retriever = VectorIndexRetriever(
            service_context=service_context,
            index=index,
            similarity_top_k=int(top_k),
        )
        response_synthesizer = ResponseSynthesizer.from_args(
            response_mode="no_text",
            service_context=service_context,
            node_postprocessors=[
                SimilarityPostprocessor(similarity_cutoff=0.7)
            ]
        )
        query_engine = RetrieverQueryEngine(
            retriever=retriever,
            response_synthesizer=response_synthesizer,
        )

        response = query_engine.query(item['question'])

        text = []
        score = []

        for i in range(len(response.source_nodes)):
            text.append(response.source_nodes[i].node.text)
            score.append(response.source_nodes[i].score)
        result = {
            'question': item['question'],
            'answers': item['answers'],
            'ctxs': text,
            'scores': score
        }
        results.append(result)

    with open(output_filename, 'w', encoding='utf-8') as output_file:
        json.dump(results, output_file, indent=4, ensure_ascii=False)

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--dataset", default=None, type=str, required=True,
#         help="dataset name: [nq, tqa, webq]",
#     )
#     parser.add_argument("--split", default=None, type=str, required=True,
#         help="dataset split: [train, dev, test]",
#     )
#     parser.add_argument('--model_name', required=True, help='Name of the Hugging Face model')
#     parser.add_argument('--top_k', required=False,default=5, help='Path to the output JSON file')
    
#     args = parser.parse_args()


#     retriever_data(f'indatasets/{args.dataset}/{args.split}.json', args.model_name, f'outdatasets/{args.dataset}/{args.split}.json', args.top_k)


def TextBasedVisionInput(self, sample: EasyDict, module: EasyDict) -> Optional[EasyDict]:
    """
    Default TextBasedVisionInput module parser
    object: text-based objects, with attributes and OCR'ed texts
    caption: iamge captions
    """
    return_dict = EasyDict(
        text_sequence="",
    )

    # Input from Vision
    vision_sentences = []
    if module.option == 'object':
        vision_sentences += [module.separation_tokens.start]
        for obj in sample.objects:
            attribute_max = module.get('attribute_max', 0)
            if attribute_max > 0:
                # find suitable attributes
                suitable_attributes = []
                for attribute, att_score in zip(obj['attributes'], obj['attribute_scores']):
                    if att_score > module.attribute_thres and len(suitable_attributes) < attribute_max:
                        suitable_attributes.append(attribute)
                # append to the sentence
                vision_sentences += suitable_attributes
            vision_sentences.append(obj['class'])
            vision_sentences.append(module.separation_tokens.sep)
        
        ocr = module.get('ocr', 0)
        if ocr > 0:
            text_annotations = sample.img_ocr
            filtered_descriptions = []
            for text_annoation in text_annotations:
                description = text_annoation['description'].strip()
                description = description.replace('\n', " ") # remove line switching
                # vision_sentences += [description]
                # print('OCR feature:', description)
                if description not in filtered_descriptions:
                    filtered_descriptions.append(description)
            # print('OCR feature:', filtered_descriptions)
            vision_sentences += filtered_descriptions

        vision_sentences += [module.separation_tokens.end]
        return_dict.text_sequence = ' '.join(vision_sentences)
    
    elif module.option == 'caption':
        return_dict.text_sequence = ' '.join([module.separation_tokens.start] + [sample.img_caption['caption']] + [module.separation_tokens.end])
        
    return return_dict
