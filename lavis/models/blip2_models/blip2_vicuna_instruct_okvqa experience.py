"""
Requires Transformer 4.28 and above, implementation may change according the Llama implementation
"""
import logging
import string
from packaging import version

import torch
from torch.cuda.amp import autocast as autocast
import torch.nn as nn

import transformers

from lavis.common.registry import registry
from lavis.models.blip2_models.blip2 import Blip2Base, disabled_train

from sentence_transformers import SentenceTransformer, util
from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F
import math
from easydict import EasyDict
import re

class DotProductSimilarity(nn.Module):
 
    def __init__(self, scale_output=False):
        super(DotProductSimilarity, self).__init__()
        self.scale_output = scale_output
 
    def forward(self, tensor_1, tensor_2):
        result = (tensor_1 * tensor_2).sum(dim=-1)
        if self.scale_output:
            result /= math.sqrt(tensor_1.size(-1))
        return result

class MultiHeadedSimilarity(nn.Module):
 
    def __init__(self,
                 num_heads,
                 tensor_1_dim,
                 tensor_1_projected_dim=None,
                 tensor_2_dim=None,
                 tensor_2_projected_dim=None,
                 internal_similarity=DotProductSimilarity()):
        super(MultiHeadedSimilarity, self).__init__()
        self.num_heads = num_heads
        self.internal_similarity = internal_similarity
        tensor_1_projected_dim = tensor_1_projected_dim or tensor_1_dim
        tensor_2_dim = tensor_2_dim or tensor_1_dim
        tensor_2_projected_dim = tensor_2_projected_dim or tensor_2_dim
        if tensor_1_projected_dim % num_heads != 0:
            raise ValueError("Projected dimension not divisible by number of heads: %d, %d"
                             % (tensor_1_projected_dim, num_heads))
        if tensor_2_projected_dim % num_heads != 0:
            raise ValueError("Projected dimension not divisible by number of heads: %d, %d"
                             % (tensor_2_projected_dim, num_heads))
        self.tensor_1_projection = nn.Parameter(torch.Tensor(tensor_1_dim, tensor_1_projected_dim))
        self.tensor_2_projection = nn.Parameter(torch.Tensor(tensor_2_dim, tensor_2_projected_dim))
        self.reset_parameters()
 
    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.tensor_1_projection)
        torch.nn.init.xavier_uniform_(self.tensor_2_projection)
 
    def forward(self, tensor_1, tensor_2):
        projected_tensor_1 = torch.matmul(tensor_1, self.tensor_1_projection.to(tensor_1.dtype))
        projected_tensor_2 = torch.matmul(tensor_2, self.tensor_2_projection.to(tensor_2.dtype))
 
        # Here we split the last dimension of the tensors from (..., projected_dim) to
        # (..., num_heads, projected_dim / num_heads), using tensor.view().
        last_dim_size = projected_tensor_1.size(-1) // self.num_heads
        new_shape = list(projected_tensor_1.size())[:-1] + [self.num_heads, last_dim_size]
        split_tensor_1 = projected_tensor_1.view(*new_shape)
        last_dim_size = projected_tensor_2.size(-1) // self.num_heads
        new_shape = list(projected_tensor_2.size())[:-1] + [self.num_heads, last_dim_size]
        split_tensor_2 = projected_tensor_2.view(*new_shape)
 
        # And then we pass this off to our internal similarity function. Because the similarity
        # functions don't care what dimension their input has, and only look at the last dimension,
        # we don't need to do anything special here. It will just compute similarity on the
        # projection dimension for each head, returning a tensor of shape (..., num_heads).
        return self.internal_similarity(split_tensor_1, split_tensor_2)#.mean(-1)


@registry.register_model("blip2_vicuna_instruct")
class Blip2VicunaInstruct(Blip2Base):
    """
    BLIP2 Vicuna model.
    Supported model types:
        - vicuna7b
        - vicuna13b
    Usage:
        >>> from lavis.models import load_model
        >>> model = load_model("blip2_vicuna_instruct", "vicuna7b")
    """

    PRETRAINED_MODEL_CONFIG_DICT = {
        "vicuna7b": "configs/models/blip2/blip2_instruct_vicuna7b.yaml",
        "vicuna13b": "configs/models/blip2/blip2_instruct_vicuna13b.yaml",
        "okvqa": "configs/models/blip2/blip2_instruct_vicuna7b_okvqa.yaml"
    }

    def __init__(
            self,
            vit_model="eva_clip_g",
            img_size=224,
            drop_path_rate=0,
            use_grad_checkpoint=False,
            vit_precision="fp16",
            freeze_vit=True,
            num_query_token=32,
            llm_model="",
            prompt="",
            max_txt_len=128,
            max_output_txt_len=256,
            apply_lemmatizer=False,
            qformer_text_input=True,
    ):
        super().__init__()
        transformers_version = version.parse(transformers.__version__)
        assert transformers_version >= version.parse("4.28"), "BLIP-2 Vicuna requires transformers>=4.28"
        from transformers import LlamaTokenizer
        from lavis.models.blip2_models.modeling_llama import LlamaForCausalLM

        self.tokenizer = self.init_tokenizer(truncation_side="left")

        self.visual_encoder, self.ln_vision = self.init_vision_encoder(
            vit_model, img_size, drop_path_rate, use_grad_checkpoint, vit_precision
        )
        if freeze_vit:
            for name, param in self.visual_encoder.named_parameters():
                param.requires_grad = False
            self.visual_encoder = self.visual_encoder.eval()
            self.visual_encoder.train = disabled_train
            logging.info("freeze vision encoder")
        
        self.Qformer, self.query_tokens = self.init_Qformer( #self.
            num_query_token, self.visual_encoder.num_features
        )

        # self.query_tokens = torch.cat([query_tokens, query_tokens2], dim=1)
        # print(self.query_tokens.shape)

        if not qformer_text_input:
            self.Qformer.bert.embeddings.word_embeddings = None
            self.Qformer.bert.embeddings.position_embeddings = None
            for layer in self.Qformer.bert.encoder.layer:
                layer.output = None
                layer.intermediate = None
        else:
            self.Qformer.resize_token_embeddings(len(self.tokenizer))
        self.Qformer.cls = None
 
        self.llm_tokenizer = LlamaTokenizer.from_pretrained(llm_model, use_fast=False, truncation_side="left")
        self.llm_model = LlamaForCausalLM.from_pretrained(
            llm_model, torch_dtype=torch.float16
        )
        self.llm_tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        self.llm_tokenizer.add_special_tokens({'bos_token': '</s>'})
        self.llm_tokenizer.add_special_tokens({'eos_token': '</s>'})
        self.llm_tokenizer.add_special_tokens({'unk_token': '</s>'})
        # self.llm_tokenizer.pad_token = self.llm_tokenizer.unk_token

        self.llm_model.resize_token_embeddings(len(self.llm_tokenizer))

        # self.eos_token_id = self.llm_tokenizer(
        #     self.llm_tokenizer.eos_token, add_special_tokens=False
        # ).input_ids[0]

        for name, param in self.llm_model.named_parameters():
            param.requires_grad = False

        self.llm_proj = nn.Linear(
            self.Qformer.config.hidden_size, self.llm_model.config.hidden_size
        )

        self.max_txt_len = max_txt_len
        self.max_output_txt_len = max_output_txt_len
        self.prompt = prompt
        # prompt_tokens = self.llm_tokenizer(self.prompt, return_tensors="pt")
        # self.prompt_length = prompt_tokens.attention_mask.sum(1)

        self._lemmatizer = None

        self.qformer_text_input = qformer_text_input

##########################
        self.Qformer2, self.query_tokens2 = self.init_Qformer2( #self.Qformer2
            num_query_token, self.visual_encoder.num_features
        )
        self.Qformer2.resize_token_embeddings(len(self.tokenizer))
        self.Qformer2.cls = None

        self.llm_proj2 = nn.Linear(
            self.Qformer.config.hidden_size, self.llm_model.config.hidden_size
            # self.Qformer.config.hidden_size, 768
        )

        self.Qformer3, self.query_tokens3 = self.init_Qformer3( #self.Qformer2
            num_query_token, self.visual_encoder.num_features
        )
        self.Qformer3.resize_token_embeddings(len(self.tokenizer))
        self.Qformer3.cls = None
        self.llm_proj3 = nn.Linear(
            self.Qformer.config.hidden_size, self.llm_model.config.hidden_size
            # self.Qformer.config.hidden_size, 768
        )


        self.Qformer4, self.query_tokens4 = self.init_Qformer4( #self.Qformer2
            num_query_token, self.visual_encoder.num_features
        )
        self.Qformer4.resize_token_embeddings(len(self.tokenizer))
        self.Qformer4.cls = None
        self.llm_proj4 = nn.Linear(
            self.Qformer.config.hidden_size, self.llm_model.config.hidden_size
            # self.Qformer.config.hidden_size, 768
        )

        self.context_len = 100
        self.min_passages_len = 20
        self.input_num = 4

        self.sim_func = MultiHeadedSimilarity(32, 4096) #torch.nn.CosineSimilarity() DotProductSimilarity() #
        # self.Qformer2.resize_token_embeddings(len(self.tokenizer))
        # self.Qformer2.cls = None

        # self.knowledge_tokenizer  = AutoTokenizer.from_pretrained('/mnt/data/qbwang/public/clip-ViT-L-14/0_CLIPModel')
        # self.knowledge_SentenceTransformer = SentenceTransformer('/mnt/data/qbwang/public/clip-ViT-L-14')
        # self.knowledge_encoder = transformers.CLIPModel.from_pretrained('/mnt/data/qbwang/public/clip-ViT-L-14/0_CLIPModel').text_model

        # for name, param in self.knowledge_encoder.named_parameters():
        #     param.requires_grad = False
        # self.knowledge_encoder = self.knowledge_encoder.eval()
        # self.knowledge_encoder.train = disabled_train
        # logging.info("freeze knowledge encoder")

        # self.weight = nn.Parameter(torch.Tensor(1, 2))
        # torch.nn.init.xavier_uniform_(self.weight)
        self.weight1_proj1 = nn.Linear(
            self.llm_model.config.hidden_size, 200
        )
        self.activate = torch.nn.Tanh()#.Tanh() #ReLU()#Tanh()#
        self.weight1_proj2 = nn.Linear(
            200, 1
        )

        self.weight2_proj1 = nn.Linear(
            self.llm_model.config.hidden_size, 200
        )

        self.weight2_proj2 = nn.Linear(
            200, 1
        )

        self.weight3_proj1 = nn.Linear(
            self.llm_model.config.hidden_size, 200
        )

        self.weight3_proj2 = nn.Linear(
            200, 1
        )



    def concat_text_input_output(self, input_ids, input_atts, output_ids, output_atts):
        input_part_targets_len = []
        llm_tokens = {"input_ids": [], "attention_mask": []}
        for i in range(input_ids.size(0)):
            this_input_ones = input_atts[i].sum()
            input_part_targets_len.append(this_input_ones)
            llm_tokens['input_ids'].append(
                torch.cat([
                    input_ids[i][:this_input_ones],
                    output_ids[i][1:],
                    input_ids[i][this_input_ones:]
                ])
            )
            llm_tokens['attention_mask'].append(
                torch.cat([
                    input_atts[i][:this_input_ones],
                    output_atts[i][1:],
                    input_atts[i][this_input_ones:]
                ])
            )
        llm_tokens['input_ids'] = torch.stack(llm_tokens['input_ids'])
        llm_tokens['attention_mask'] = torch.stack(llm_tokens['attention_mask'])
        return llm_tokens, input_part_targets_len


    def forward(self, samples):
        print('-----------------')
        print(samples["text_input"]) 
        # print(samples["gold_answer"])
        # print(samples["text_output"])
        # print('-----------------')
        
        ##############update input###############
        # print(samples["experiences"])


        samples["image"] = samples["image"].expand(self.input_num, -1, -1, -1)
        samples["text_input"] = samples["text_input"] * self.input_num

        # text_input_origin = samples["text_input"] 
        # samples["passages"] = re.split('#|\.|\"', samples["passages"][0]) #[:3]

        samples["passages"] = samples["passages"][0].split("#") #[:3] #|
        samples["passages"] = [item.split(",",maxsplit=1)[-1].strip("\"").strip(".") for item in samples["passages"] if len(item) > self.min_passages_len]
        # samples["passages"] = [sentence.strip() for item in samples["passages"] for sentence in item.split(".") if len(sentence) > self.min_passages_len]
        # print(samples["passages"])
        # print(len(samples["passages"]))

        samples["answer"] = samples["answer"] * self.input_num
        samples["gold_answer"] = samples["gold_answer"] * self.input_num
        # samples["weight"] = samples["weight"].expand(16)
        # samples["n_answers"] = samples["n_answers"].expand(16)

        ##########################################

        image = samples["image"]
        with self.maybe_autocast():
            image_embeds = self.ln_vision(self.visual_encoder(image))

        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device)
        bs = image.size(0)

        query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1).to(image.device)

#######################################
        query_tokens2 = self.query_tokens2.expand(image_embeds.shape[0], -1, -1).to(image.device)
        query_tokens3 = self.query_tokens3.expand(image_embeds.shape[0], -1, -1).to(image.device)
        query_tokens4 = self.query_tokens4.expand(image_embeds.shape[0], -1, -1).to(image.device)
        
        if self.qformer_text_input:
            text_Qformer = self.tokenizer(
                samples["text_input"],
                padding='longest',
                truncation=True,
                max_length=self.max_txt_len,
                return_tensors="pt",
            ).to(image.device)
            query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(image.device)
            Qformer_atts = torch.cat([query_atts, text_Qformer.attention_mask], dim=1)
            query_output = self.Qformer.bert(
                text_Qformer.input_ids,
                attention_mask=Qformer_atts,
                query_embeds=query_tokens,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )

        else:
            query_output = self.Qformer.bert(
                query_embeds=query_tokens,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )

        inputs_llm = self.llm_proj(query_output.last_hidden_state[:, :query_tokens.size(1), :])
        atts_llm = torch.ones(inputs_llm.size()[:-1], dtype=torch.long).to(image.device)

        #######################knowledge enhancement###################
        text_Qformer2 = self.tokenizer(
            samples["text_input"],
            padding='longest',
            truncation=True,
            max_length=self.max_txt_len,
            return_tensors="pt",
        ).to(image.device)
        query_atts2 = torch.ones(query_tokens2.size()[:-1], dtype=torch.long).to(image.device)
        Qformer_atts2 = torch.cat([query_atts2, text_Qformer2.attention_mask], dim=1)
        query_output2 = self.Qformer2.bert(
            text_Qformer2.input_ids,
            attention_mask=Qformer_atts2,
            query_embeds=query_tokens2,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            return_dict=True,
        )
        inputs_llm2 = self.llm_proj2(query_output2.last_hidden_state[:, :query_tokens2.size(1), :])
        atts_llm2= torch.ones(inputs_llm2.size()[:-1], dtype=torch.long).to(image.device)
        inputs_llm = torch.cat([inputs_llm, inputs_llm2], dim=1)
        atts_llm = torch.cat([atts_llm, atts_llm2], dim=1)

        ##########################
        # print(samples["text_input"])
        # print(samples["passages"])
        text_Qformer4 = self.tokenizer(
            samples["text_input"],
            padding='longest',
            truncation=True,
            max_length=self.max_txt_len,
            return_tensors="pt",
        ).to(image.device)
        query_atts4 = torch.ones(query_tokens4.size()[:-1], dtype=torch.long).to(image.device)
        Qformer_atts4 = torch.cat([query_atts4, text_Qformer4.attention_mask], dim=1)
        query_output4 = self.Qformer4.bert(
            text_Qformer4.input_ids,
            attention_mask=Qformer_atts4,
            query_embeds=query_tokens4,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            return_dict=True,
        )
        inputs_llm4 = self.llm_proj4(query_output4.last_hidden_state[:, :query_tokens4.size(1), :])#.expand(self.input_num, -1, -1)

        print([item for item in samples["experiences"][0]])
        knowledge_tokens2 = self.llm_tokenizer(
            [item for item in samples["experiences"][0]], 
            padding="longest",
            return_tensors="pt"
        ).to(image.device)
        knowledge_embeds2 = self.llm_model.get_input_embeddings()(knowledge_tokens2.input_ids)#.mean(-2)
        knowledge_embeds2 = (knowledge_embeds2 * knowledge_tokens2['attention_mask'].unsqueeze(-1))#.sum(-2) 
        question_qformer_embeds2 = inputs_llm4[0,:,:].mean(-2).unsqueeze(-2).expand(knowledge_embeds2.shape[0], -1, -1)#.detach()
        sim_score2 = self.sim_func(question_qformer_embeds2, knowledge_embeds2)
        mean_sim_score2 = (sim_score2 * knowledge_tokens2['attention_mask'].unsqueeze(-1)).mean(-1).sum(-1) / knowledge_tokens2['attention_mask'].sum(-1) #sim_score.max(-1)[0]
        knowledge_scores_sigmoid2 = F.sigmoid(mean_sim_score2)
        relevate_knowledge_ind2 = (knowledge_scores_sigmoid2).topk(3)[1]
        relevate_knowledge_scores2 = (knowledge_scores_sigmoid2).topk(3)[0]
        att_score2 = F.softmax(sim_score2.index_select(0,relevate_knowledge_ind2).masked_fill_((1 - knowledge_tokens2['attention_mask'].index_select(0,relevate_knowledge_ind2).unsqueeze(-1)).bool(), -9999.9)  , dim=-2)
        knowledge_embeds_final2 = (att_score2.unsqueeze(-1) * knowledge_embeds2.index_select(0,relevate_knowledge_ind2).unsqueeze(2)).sum(1)
        knowledge_embeds_final2 = (knowledge_embeds_final2[0].unsqueeze(0)).expand(3,-1,-1)

        ########
        text_Qformer3 = self.tokenizer(
            [samples["text_input"][0]+' '+ samples["experiences"][0][relevate_knowledge_ind2[0]]] * 4,
            padding='longest',
            truncation=True,
            max_length=self.max_txt_len,
            return_tensors="pt",
        ).to(image.device)
        query_atts3 = torch.ones(query_tokens3.size()[:-1], dtype=torch.long).to(image.device)
        Qformer_atts3 = torch.cat([query_atts3, text_Qformer3.attention_mask], dim=1)
        query_output3 = self.Qformer3.bert(
            text_Qformer3.input_ids,
            attention_mask=Qformer_atts3,
            query_embeds=query_tokens3,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            return_dict=True,
        )
        inputs_llm3 = self.llm_proj3(query_output3.last_hidden_state[:, :query_tokens3.size(1), :])
        # atts_llm3= torch.ones(inputs_llm3.size()[:-1], dtype=torch.long).to(image.device)
        # inputs_llm = torch.cat([inputs_llm, inputs_llm3], dim=1)
        # atts_llm = torch.cat([atts_llm, atts_llm3], dim=1)

        knowledge_tokens = self.llm_tokenizer(
            [item[:self.context_len] for item in samples["passages"]], 
            padding="longest",
            return_tensors="pt"
        ).to(image.device)

        knowledge_embeds = self.llm_model.get_input_embeddings()(knowledge_tokens.input_ids)#.mean(-2)
        #这句话其实可以删掉
        # knowledge_embeds = (knowledge_embeds * knowledge_tokens['attention_mask'].unsqueeze(-1))#.sum(-2) 
        # knowledge_embeds = knowledge_embeds /  knowledge_tokens['attention_mask'].sum(-1).unsqueeze(-1)
        # sim_score = (sim_score * )

        question_qformer_embeds = inputs_llm3[0,:,:].mean(-2).unsqueeze(-2).expand(knowledge_embeds.shape[0], -1, -1)#.detach()
        
        sim_score = self.sim_func(question_qformer_embeds, knowledge_embeds)
        # print(sim_score.shape)
        mean_sim_score = (sim_score * knowledge_tokens['attention_mask'].unsqueeze(-1)).mean(-1).sum(-1) / knowledge_tokens['attention_mask'].sum(-1) #sim_score.max(-1)[0]
        # print(mean_sim_score.shape)
        knowledge_scores_sigmoid = F.sigmoid(mean_sim_score)
        relevate_knowledge_ind = (knowledge_scores_sigmoid).topk(3)[1]
        relevate_knowledge_scores = (knowledge_scores_sigmoid).topk(3)[0]
        # print(relevate_knowledge_ind)
        # print(relevate_knowledge_scores)

        #att_score应该改成先求平均了再用 #确认下是否有32维sim_score
        att_score = F.softmax(sim_score.index_select(0,relevate_knowledge_ind).masked_fill_((1- knowledge_tokens['attention_mask'].index_select(0,relevate_knowledge_ind).unsqueeze(-1)).bool(), -9999.9), dim=-2)
        # print(att_score)
        # print(att_score.shape)
        knowledge_embeds_final = (att_score.unsqueeze(-1) * knowledge_embeds.index_select(0,relevate_knowledge_ind).unsqueeze(2)).sum(1) #3 L 32 1 * 3 L 1 4096
        
        


        ##########################################################################


        text_input_buf = []
        for i in range(3):
            # text_input_buf.append('Context: ' + samples["passages"][relevate_knowledge_ind[i]][:self.context_len] + '. Question: ' +samples["text_input"][i] + ' Short answer:') #  referring to Reference # (referring to Context if Context is useful)
            text_input_buf.append('Question: ' +samples["text_input"][i] + ' Short answer:')

        text_input_buf.append('Question: ' +samples["text_input"][3] + ' Short answer:')
        
        samples["text_input"] = text_input_buf
        # print(samples["text_input"])

        self.llm_tokenizer.padding_side = "right"
        self.llm_tokenizer.truncation_side = 'left'

        text_input_tokens = self.llm_tokenizer(
            samples['text_input'],
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=self.max_txt_len,
        ).to(image.device)

        self.llm_tokenizer.truncation_side = 'right'

        text_output_tokens = self.llm_tokenizer(
            [t + self.llm_tokenizer.eos_token for t in samples['answer']],
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=self.max_output_txt_len,
        ).to(image.device)

        # print(text_input_tokens.input_ids)

        llm_tokens, input_part_targets_len = self.concat_text_input_output(
            text_input_tokens.input_ids,
            text_input_tokens.attention_mask,
            text_output_tokens.input_ids,
            text_output_tokens.attention_mask,
        )

        # do not apply loss to the padding
        targets = llm_tokens['input_ids'].masked_fill(
            llm_tokens['input_ids'] == self.llm_tokenizer.pad_token_id, -100
        )

        

        # do not apply loss to the text input (i.e., instruction)
        for i, l in enumerate(input_part_targets_len):
            targets[i][:l] = -100

#######################################
        # do not apply loss to the query tokens
        empty_targets = (
            torch.ones(atts_llm[:,:32].size(), dtype=torch.long).to(image.device).fill_(-100) #atts_llm.size()  
        )

        targets = torch.cat([empty_targets, targets], dim=1)

        inputs_embeds = self.llm_model.get_input_embeddings()(llm_tokens['input_ids'])


       
######################################################
        # a = torch.cat([empty_targets[:3,:], targets[:3,:]], dim=1)
        # b = torch.cat([targets[3,:].unsqueeze(0),empty_targets[3,:].unsqueeze(0)],dim=1)
        # targets = torch.cat([a, b], dim=0)
        # pad_embeds = self.llm_model.get_input_embeddings()(torch.tensor([self.llm_tokenizer.pad_token_id] * 32).to(image.device))
        # pad_att = torch.zeros([1,32], dtype=torch.long).to(image.device)
        # inputs_embeds1 = torch.cat([torch.cat([inputs_llm[:3,:32,:], knowledge_embeds2], dim=1), inputs_embeds[:3,:,:]], dim=1)
        # attention_mask1 = torch.cat([torch.cat([atts_llm[:3,:32], atts_llm[:3,:32]], dim=1),  llm_tokens['attention_mask'][:3,:]], dim=1)
        # inputs_embeds2 = torch.cat([torch.cat([inputs_llm[3,32:64,:].unsqueeze(0), inputs_embeds[3,:,:].unsqueeze(0)], dim=1),pad_embeds.unsqueeze(0)], dim=1)
        # attention_mask2 = torch.cat([torch.cat([atts_llm[3,32:64].unsqueeze(0), llm_tokens['attention_mask'][3,:].unsqueeze(0)], dim=1),pad_att], dim=1)

        # a = F.softmax(self.weight)[0][0].unsqueeze(-1).unsqueeze(-1) 
        # b = F.softmax(self.weight)[0][1].unsqueeze(-1).unsqueeze(-1) 
        # c = 9 * inputs_llm[:3,:32,:] + 1 * knowledge_embeds2

        w = F.softmax(torch.cat([self.weight1_proj2(self.activate(self.weight1_proj1(inputs_llm[:3,:32,:]))),self.weight2_proj2(self.activate(self.weight2_proj1(knowledge_embeds_final))),self.weight3_proj2(self.activate(self.weight3_proj1(knowledge_embeds_final2)))], dim=-1), dim=-1)
        print(w)
        c = w[:,:,0].unsqueeze(-1) * inputs_llm[:3,:32,:] + w[:,:,1].unsqueeze(-1) * knowledge_embeds_final  + w[:,:,2].unsqueeze(-1) * knowledge_embeds_final2

        inputs_embeds1 = torch.cat([c, inputs_embeds[:3,:,:]], dim=1)
        attention_mask1 = torch.cat([atts_llm[:3,:32], llm_tokens['attention_mask'][:3,:]], dim=1)
        inputs_embeds2 = torch.cat([inputs_llm[3,32:64,:].unsqueeze(0), inputs_embeds[3,:,:].unsqueeze(0)], dim=1)
        attention_mask2 = torch.cat([atts_llm[3,32:64].unsqueeze(0), llm_tokens['attention_mask'][3,:].unsqueeze(0)], dim=1)
        inputs_embeds = torch.cat([inputs_embeds1, inputs_embeds2], dim=0)
        attention_mask = torch.cat([attention_mask1, attention_mask2], dim=0)

        with self.maybe_autocast():
            outputs = self.llm_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                return_dict=True,
                labels=targets,
                reduction="none",
            )


###################################knowledge sim cal###############################

        loss_labels2 = outputs.attentions
        # ignore_mask = torch.tensor([False, False, False]).to(loss_labels2.device)
        # if not loss_labels2[3]:
        ignore_mask = (loss_labels2[3].unsqueeze(-1) ==  loss_labels2[:3])
        print(loss_labels2)

        knowledge_loss = F.binary_cross_entropy(relevate_knowledge_scores, loss_labels2[:3].float(), reduction='none')

        knowledge_loss.masked_fill_(ignore_mask, 0.0)  
        print(knowledge_loss)
        outputs.loss = torch.cat([outputs.loss,knowledge_loss],dim=-1)
        # outputs.loss[4:] = 0
        # print(outputs.loss)

# ###################################################
        # print((outputs.loss > 0).sum())
        loss = outputs.loss.sum() / (outputs.loss > 0).sum()

        # if knowledge_labels[:-1].int().sum() == 0 or knowledge_labels[:-1].int().sum() == 3:
        #     loss *= 0

        # loss = outputs.loss[knowledge_labels].mean() #+ knowledge_loss  #[idx] [knowledge_labels]
 
        return {"loss": loss} #loss

    @torch.no_grad()
    def generate(
            self,
            samples,
            use_nucleus_sampling=False,
            num_beams=5,
            max_length=256,
            min_length=1,
            top_p=0.9,
            repetition_penalty=1.5,
            length_penalty=1,
            num_captions=1,
            temperature=1,
    ):

        print('-----------------')
        self.llm_tokenizer.padding_side = "left"

        if "prompt" in samples.keys():
            prompt = samples["prompt"]
        else:
            prompt = self.prompt

        image = samples["image"]

        bs = image.size(0)

        if isinstance(prompt, str):
            prompt = [prompt] * bs
        else:
            assert len(prompt) == bs, "The number of prompts must be equal to the batch size."

        # For TextCaps
        if "ocr_tokens" in samples.keys() and "{}" in prompt[0]:
            prompt = [p.format(', '.join(samples['ocr_tokens'][i][:30])) for i, p in enumerate(prompt)]


#####################################
        print(samples["passages"])
        samples["passages"] = samples["passages"][0].split("#") #[:3] #|
        samples["passages"] = [item.split(",",maxsplit=1)[-1].strip("\"").strip(".") for item in samples["passages"] if len(item) > self.min_passages_len]
        # samples["passages"] = [sentence.strip() for item in samples["passages"] for sentence in item.split(".") if len(sentence) > self.min_passages_len]
        print(samples["passages"])
        print(len(samples["passages"]))


        query_tokens = self.query_tokens.expand(bs, -1, -1).to(image.device)
#######################################
        query_tokens2 = self.query_tokens2.expand(bs, -1, -1).to(image.device)
        query_tokens3 = self.query_tokens3.expand(bs, -1, -1).to(image.device)
        query_tokens4 = self.query_tokens4.expand(bs, -1, -1).to(image.device)

        if self.qformer_text_input:
            # remove ocr tokens in q_former (for eval textvqa)
            # qformer_prompt = prompt
            # qformer_prompt = ['Question: ' + qp.split(' Question: ')[1] for qp in qformer_prompt]

            text_Qformer = self.tokenizer(
                samples["text_input"],
                padding='longest',
                truncation=True,
                max_length=self.max_txt_len,
                return_tensors="pt",
            ).to(image.device)
            query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(image.device)
            Qformer_atts = torch.cat([query_atts, text_Qformer.attention_mask], dim=1)

        # For video data
        if image.dim() == 5:
            inputs_llm, atts_llm = [], []
            for j in range(image.size(2)):
                this_frame = image[:, :, j, :, :]
                with self.maybe_autocast():
                    frame_embeds = self.ln_vision(self.visual_encoder(this_frame))
                frame_atts = torch.ones(frame_embeds.size()[:-1], dtype=torch.long).to(image.device)

                if self.qformer_text_input:
                    frame_query_output = self.Qformer.bert(
                        text_Qformer.input_ids,
                        attention_mask=Qformer_atts,
                        query_embeds=query_tokens,
                        encoder_hidden_states=frame_embeds,
                        encoder_attention_mask=frame_atts,
                        return_dict=True,
                    )
                else:
                    frame_query_output = self.Qformer.bert(
                        query_embeds=query_tokens,
                        encoder_hidden_states=frame_embeds,
                        encoder_attention_mask=frame_atts,
                        return_dict=True,
                    )
                frame_inputs_llm = self.llm_proj(frame_query_output.last_hidden_state[:, :query_tokens.size(1), :])
                frame_atts_llm = torch.ones(frame_inputs_llm.size()[:-1], dtype=torch.long).to(image.device)
                inputs_llm.append(frame_inputs_llm)
                atts_llm.append(frame_atts_llm)
            inputs_llm = torch.cat(inputs_llm, dim=1)
            atts_llm = torch.cat(atts_llm, dim=1)
        else:
            with self.maybe_autocast():
                image_embeds = self.ln_vision(self.visual_encoder(image))
            image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device)

            if self.qformer_text_input:
                query_output = self.Qformer.bert(
                    text_Qformer.input_ids,
                    attention_mask=Qformer_atts,
                    query_embeds=query_tokens,
                    encoder_hidden_states=image_embeds,
                    encoder_attention_mask=image_atts,
                    return_dict=True,
                )
            else:
                query_output = self.Qformer.bert(
                    query_embeds=query_tokens,
                    encoder_hidden_states=image_embeds,
                    encoder_attention_mask=image_atts,
                    return_dict=True,
                )

            inputs_llm = self.llm_proj(query_output.last_hidden_state[:, :query_tokens.size(1), :]).expand(self.input_num, -1, -1) #
            atts_llm = torch.ones(inputs_llm.size()[:-1], dtype=torch.long).to(image.device)

        ##################################knowledge enhancement############
        text_Qformer2 = self.tokenizer(
            samples["text_input"],
            padding='longest',
            truncation=True,
            max_length=self.max_txt_len,
            return_tensors="pt",
        ).to(image.device)
        query_atts2 = torch.ones(query_tokens2.size()[:-1], dtype=torch.long).to(image.device)
        Qformer_atts2 = torch.cat([query_atts2, text_Qformer2.attention_mask], dim=1)
        query_output2 = self.Qformer2.bert(
            text_Qformer2.input_ids,
            attention_mask=Qformer_atts2,
            query_embeds=query_tokens2,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            return_dict=True,
        )
        inputs_llm2 = self.llm_proj2(query_output2.last_hidden_state[:, :query_tokens2.size(1), :]).expand(self.input_num, -1, -1)
        atts_llm2= torch.ones(inputs_llm2.size()[:-1], dtype=torch.long).to(image.device)
        inputs_llm = torch.cat([inputs_llm, inputs_llm2], dim=1)
        atts_llm = torch.cat([atts_llm, atts_llm2], dim=1)

        ##########################
        text_Qformer4 = self.tokenizer(
            samples["text_input"], #samples["text_input"],
            padding='longest',
            truncation=True,
            max_length=self.max_txt_len,
            return_tensors="pt",
        ).to(image.device)
        query_atts4 = torch.ones(query_tokens4.size()[:-1], dtype=torch.long).to(image.device)
        Qformer_atts4 = torch.cat([query_atts4, text_Qformer4.attention_mask], dim=1)
        query_output4 = self.Qformer4.bert(
            text_Qformer4.input_ids,
            attention_mask=Qformer_atts4,
            query_embeds=query_tokens4,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            return_dict=True,
        )
        inputs_llm4 = self.llm_proj4(query_output4.last_hidden_state[:, :query_tokens4.size(1), :])#.expand(self.input_num, -1, -1)
        knowledge_tokens2 = self.llm_tokenizer(
            [item for item in samples["experiences"][0]], 
            padding="longest",
            return_tensors="pt"
        ).to(image.device)
        knowledge_embeds2 = self.llm_model.get_input_embeddings()(knowledge_tokens2.input_ids)#.mean(-2)
        knowledge_embeds2 = (knowledge_embeds2 * knowledge_tokens2['attention_mask'].unsqueeze(-1))#.sum(-2) 
        question_qformer_embeds2 = inputs_llm4[0,:,:].mean(-2).unsqueeze(-2).expand(knowledge_embeds2.shape[0], -1, -1)#.detach()
        sim_score2 = self.sim_func(question_qformer_embeds2, knowledge_embeds2)
        mean_sim_score2 = (sim_score2 * knowledge_tokens2['attention_mask'].unsqueeze(-1)).mean(-1).sum(-1) / knowledge_tokens2['attention_mask'].sum(-1) #sim_score.max(-1)[0]
        knowledge_scores_sigmoid2 = F.sigmoid(mean_sim_score2)
        relevate_knowledge_ind2 = (knowledge_scores_sigmoid2).topk(3)[1]
        relevate_knowledge_scores2 = (knowledge_scores_sigmoid2).topk(3)[0]
        att_score2 = F.softmax(sim_score2.index_select(0,relevate_knowledge_ind2).masked_fill_((1 - knowledge_tokens2['attention_mask'].index_select(0,relevate_knowledge_ind2).unsqueeze(-1)).bool(), -9999.9)  , dim=-2)
        knowledge_embeds_final2 = (att_score2.unsqueeze(-1) * knowledge_embeds2.index_select(0,relevate_knowledge_ind2).unsqueeze(2)).sum(1)
        knowledge_embeds_final2 = (knowledge_embeds_final2[0].unsqueeze(0)).expand(3,-1,-1)

        ########
        text_Qformer3 = self.tokenizer(
            [samples["text_input"][0]+' '+ samples["experiences"][0][relevate_knowledge_ind2[0]]],
            padding='longest',
            truncation=True,
            max_length=self.max_txt_len,
            return_tensors="pt",
        ).to(image.device)
        query_atts3 = torch.ones(query_tokens3.size()[:-1], dtype=torch.long).to(image.device)
        Qformer_atts3 = torch.cat([query_atts3, text_Qformer3.attention_mask], dim=1)
        query_output3 = self.Qformer3.bert(
            text_Qformer3.input_ids,
            attention_mask=Qformer_atts3,
            query_embeds=query_tokens3,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            return_dict=True,
        )
        inputs_llm3 = self.llm_proj3(query_output3.last_hidden_state[:, :query_tokens3.size(1), :])#.expand(self.input_num, -1, -1)

        knowledge_tokens = self.llm_tokenizer(
            [item[:self.context_len] for item in samples["passages"]], 
            padding="longest",
            return_tensors="pt"
        ).to(image.device)

        knowledge_embeds = self.llm_model.get_input_embeddings()(knowledge_tokens.input_ids)#.mean(-2)
        knowledge_embeds = (knowledge_embeds * knowledge_tokens['attention_mask'].unsqueeze(-1))#.sum(-2) 

        question_qformer_embeds = inputs_llm3[0,:,:].mean(-2).unsqueeze(-2).expand(knowledge_embeds.shape[0], -1, -1)#.detach()
        sim_score = self.sim_func(question_qformer_embeds, knowledge_embeds)
        mean_sim_score = (sim_score * knowledge_tokens['attention_mask'].unsqueeze(-1)).mean(-1).sum(-1) / knowledge_tokens['attention_mask'].sum(-1) #sim_score.max(-1)[0]
        knowledge_scores_sigmoid = F.sigmoid(mean_sim_score)
        relevate_knowledge_ind = (knowledge_scores_sigmoid).topk(3)[1]
        relevate_knowledge_scores = (knowledge_scores_sigmoid).topk(3)[0]
        att_score = F.softmax(sim_score.index_select(0,relevate_knowledge_ind).masked_fill_((1 - knowledge_tokens['attention_mask'].index_select(0,relevate_knowledge_ind).unsqueeze(-1)).bool(), -9999.9)  , dim=-2)
        knowledge_embeds_final = (att_score.unsqueeze(-1) * knowledge_embeds.index_select(0,relevate_knowledge_ind).unsqueeze(2)).sum(1)
        



    
        samples["text_input"] = samples["text_input"] * self.input_num

        text_input_buf = []
        for i in range(3):
            text_input_buf.append('Question: ' +samples["text_input"][i] + ' Short answer:')
        text_input_buf.append('Question: ' +samples["text_input"][3] + ' Short answer:')
               
        samples["text_input"] = text_input_buf

        ####################################################################        

        llm_tokens = self.llm_tokenizer(
            samples["text_input"], #prompt,
            padding="longest",
            return_tensors="pt"
        ).to(image.device)

        with self.maybe_autocast():

            inputs_embeds = self.llm_model.get_input_embeddings()(llm_tokens.input_ids)

###################################################################################
            # pad_embeds = self.llm_model.get_input_embeddings()(torch.tensor([self.llm_tokenizer.pad_token_id] * 32).to(image.device))
            # pad_att = torch.zeros([1,32], dtype=torch.long).to(image.device)
            # inputs_embeds1 = torch.cat([torch.cat([inputs_llm[:3,:32,:], knowledge_embeds2], dim=1), inputs_embeds[:3,:,:]], dim=1)
            # attention_mask1 = torch.cat([torch.cat([atts_llm[:3,:32], atts_llm[:3,:32]], dim=1),  llm_tokens['attention_mask'][:3,:]], dim=1)
            # inputs_embeds2 = torch.cat([torch.cat([inputs_llm[3,32:64,:].unsqueeze(0), inputs_embeds[3,:,:].unsqueeze(0)], dim=1),pad_embeds.unsqueeze(0)], dim=1)
            # attention_mask2 = torch.cat([torch.cat([atts_llm[3,32:64].unsqueeze(0), llm_tokens['attention_mask'][3,:].unsqueeze(0)], dim=1),pad_att], dim=1)
            # a = F.softmax(self.weight)[0][0].unsqueeze(-1).unsqueeze(-1) 
            # b = F.softmax(self.weight)[0][1].unsqueeze(-1).unsqueeze(-1) 
            # c = 9 * inputs_llm[:3,:32,:] + 1 * knowledge_embeds2
            # w = F.softmax(torch.cat([self.weight_proj(inputs_llm[:3,:32,:]),self.weight2_proj(knowledge_embeds2)], dim=-1), dim=-1)
            w = F.softmax(torch.cat([self.weight1_proj2(self.activate(self.weight1_proj1(inputs_llm[:3,:32,:]))),self.weight2_proj2(self.activate(self.weight2_proj1(knowledge_embeds_final))),self.weight3_proj2(self.activate(self.weight3_proj1(knowledge_embeds_final2)))], dim=-1), dim=-1)
            c = w[:,:,0].unsqueeze(-1) * inputs_llm[:3,:32,:] + w[:,:,1].unsqueeze(-1) * knowledge_embeds_final  + w[:,:,2].unsqueeze(-1) * knowledge_embeds_final2

            inputs_embeds1 = torch.cat([c, inputs_embeds[:3,:,:]], dim=1)
            attention_mask1 = torch.cat([atts_llm[:3,:32], llm_tokens['attention_mask'][:3,:]], dim=1)
            inputs_embeds2 = torch.cat([inputs_llm[3,32:64,:].unsqueeze(0), inputs_embeds[3,:,:].unsqueeze(0)], dim=1)
            attention_mask2 = torch.cat([atts_llm[3,32:64].unsqueeze(0), llm_tokens['attention_mask'][3,:].unsqueeze(0)], dim=1)
            inputs_embeds = torch.cat([inputs_embeds1, inputs_embeds2], dim=0)
            attention_mask = torch.cat([attention_mask1, attention_mask2], dim=0)

            outputs = self.llm_model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                do_sample=use_nucleus_sampling,
                top_p=top_p,
                temperature=temperature,
                num_beams=num_beams,
                max_length=max_length,
                min_length=min_length,
                # eos_token_id=self.eos_token_id,
                repetition_penalty=repetition_penalty,
                length_penalty=length_penalty,
                num_return_sequences=num_captions,
            )

        outputs[outputs == 0] = 2  # convert output id 0 to 2 (eos_token_id)
        outputs[outputs == -1] = 2
        outputs[outputs == 1] = 2

        output_text = self.llm_tokenizer.batch_decode(outputs, skip_special_tokens=True)

        output_text = [text.strip() for text in output_text]

        #######################determine output##########################
        answer_dict = EasyDict()
        answer_dict[output_text[3]] = 1.1
        
        print_sig = False
        for i in range(3):
        
            # if knowledge_scores_sigmoid[i] > 0.5:
                    
            print_sig = True
            
            if output_text[i] not in answer_dict.keys():
                answer_dict[output_text[i]] = 1
            else:
                answer_dict[output_text[i]] += 1

        # if print_sig:
        print(output_text)
        # print(samples["text_input"])
        # print(samples["passages"])
        print(samples["answers_list"])
        print(answer_dict)

        output_text = max(zip(answer_dict.values(), answer_dict.keys()))
        output_text = [output_text[1]]
        
        # output_text = [output_text[0]]
        #################################################

        return output_text

    @torch.no_grad()
    def generate_incontext(
            self,
            samples,
            use_nucleus_sampling=False,
            num_beams=5,
            max_length=256,
            min_length=1,
            top_p=0.9,
            repetition_penalty=1.5,
            length_penalty=1,
            num_captions=1,
            temperature=1,
    ):

        self.llm_tokenizer.padding_side = "left"

        self.qformer_text_input = False

        if "prompt" in samples.keys():
            prompt = samples["prompt"]
        else:
            prompt = self.prompt

        image = samples["image"]

        bs = image.size(0)

        if isinstance(prompt, str):
            prompt = [prompt] * bs
        else:
            assert len(prompt) == bs, "The number of prompts must be equal to the batch size."

        # For TextCaps
        if "ocr_tokens" in samples.keys() and "{}" in prompt[0]:
            prompt = [p.format(', '.join(samples['ocr_tokens'][i][:30])) for i, p in enumerate(prompt)]

        query_tokens = self.query_tokens.expand(bs, -1, -1)
        if self.qformer_text_input:
            # remove ocr tokens in q_former (for eval textvqa)
            # qformer_prompt = prompt
            # qformer_prompt = ['Question: ' + qp.split(' Question: ')[1] for qp in qformer_prompt]

            text_Qformer = self.tokenizer(
                prompt,
                padding='longest',
                truncation=True,
                max_length=self.max_txt_len,
                return_tensors="pt",
            ).to(image.device)
            query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(image.device)
            Qformer_atts = torch.cat([query_atts, text_Qformer.attention_mask], dim=1)

        # For video data
        if image.dim() == 5:
            inputs_llm, atts_llm = [], []
            for j in range(image.size(2)):
                this_frame = image[:, :, j, :, :]
                with self.maybe_autocast():
                    frame_embeds = self.ln_vision(self.visual_encoder(this_frame))
                frame_atts = torch.ones(frame_embeds.size()[:-1], dtype=torch.long).to(image.device)

                if self.qformer_text_input:
                    frame_query_output = self.Qformer.bert(
                        text_Qformer.input_ids,
                        attention_mask=Qformer_atts,
                        query_embeds=query_tokens,
                        encoder_hidden_states=frame_embeds,
                        encoder_attention_mask=frame_atts,
                        return_dict=True,
                    )
                else:
                    frame_query_output = self.Qformer.bert(
                        query_embeds=query_tokens,
                        encoder_hidden_states=frame_embeds,
                        encoder_attention_mask=frame_atts,
                        return_dict=True,
                    )
                frame_inputs_llm = self.llm_proj(frame_query_output.last_hidden_state[:, :query_tokens.size(1), :])
                frame_atts_llm = torch.ones(frame_inputs_llm.size()[:-1], dtype=torch.long).to(image.device)
                inputs_llm.append(frame_inputs_llm)
                atts_llm.append(frame_atts_llm)
            # inputs_llm = torch.cat(inputs_llm, dim=1)
            # atts_llm = torch.cat(atts_llm, dim=1)
        else:
            with self.maybe_autocast():
                image_embeds = self.ln_vision(self.visual_encoder(image))
            image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device)

            if self.qformer_text_input:
                query_output = self.Qformer.bert(
                    text_Qformer.input_ids,
                    attention_mask=Qformer_atts,
                    query_embeds=query_tokens,
                    encoder_hidden_states=image_embeds,
                    encoder_attention_mask=image_atts,
                    return_dict=True,
                )
            else:
                query_output = self.Qformer.bert(
                    query_embeds=query_tokens,
                    encoder_hidden_states=image_embeds,
                    encoder_attention_mask=image_atts,
                    return_dict=True,
                )

            inputs_llm = self.llm_proj(query_output.last_hidden_state[:, :query_tokens.size(1), :])
            atts_llm = torch.ones(inputs_llm.size()[:-1], dtype=torch.long).to(image.device)

        # llm_tokens = self.llm_tokenizer(
        #     prompt,
        #     padding="longest",
        #     return_tensors="pt"
        # ).to(image.device)

        llm_tokens_bs = []
        for p in prompt:
            llm_tokens = self.llm_tokenizer(
                p,
                padding="longest",
                return_tensors="pt"
            ).to(image.device)
            llm_tokens_bs.append(llm_tokens)

        with self.maybe_autocast():
            # inputs_embeds = self.llm_model.get_input_embeddings()(llm_tokens.input_ids)
            # inputs_embeds = torch.cat([inputs_llm, inputs_embeds], dim=1)
            # attention_mask = torch.cat([atts_llm, llm_tokens.attention_mask], dim=1)
            inputs_embeds = []
            attention_mask = []
            for j in range(image.size(2)):
                text_emb = self.llm_model.get_input_embeddings()(llm_tokens_bs[0].input_ids[j])
                visual_emb = inputs_llm[j][0]
                inputs_embeds.append(visual_emb)
                inputs_embeds.append(text_emb)

                text_att = llm_tokens_bs[0].attention_mask[j]
                visual_att = atts_llm[j][0]
                attention_mask.append(visual_att)
                attention_mask.append(text_att)
            inputs_embeds = torch.cat(inputs_embeds, dim=0).unsqueeze(0)
            attention_mask = torch.cat(attention_mask, dim=0).unsqueeze(0)

            outputs = self.llm_model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                do_sample=use_nucleus_sampling,
                top_p=top_p,
                temperature=temperature,
                num_beams=num_beams,
                max_length=max_length,
                min_length=min_length,
                # eos_token_id=self.eos_token_id,
                repetition_penalty=repetition_penalty,
                length_penalty=length_penalty,
                num_return_sequences=num_captions,
            )
        
        outputs[outputs == 0] = 2  # convert output id 0 to 2 (eos_token_id)
        
        output_text = self.llm_tokenizer.batch_decode(outputs, skip_special_tokens=True)
        output_text = [text.strip() for text in output_text]

        return output_text

    def predict_answers(
            self,
            samples,
            num_beams=5,
            inference_method="generate",
            max_len=10,
            min_len=1,
            num_ans_candidates=128,
            answer_list=None,
            prompt="",
            length_penalty=0,
            **kwargs
    ):
        
        # for i in range(len(samples["text_input"])):
        #     knowledges = samples["passages"][i].split("#")[:3]
        #     samples["text_input"][i] = samples["text_input"][i]+"#"+knowledges[0]  #samples["passages"][i][0]

        if isinstance(samples["text_input"], str):
            samples["text_input"] = [samples["text_input"]]

        if prompt:
            if prompt.count("{}") == 2:
                if 'ocr_tokens' in samples:
                    text_input = [
                        prompt.format(', '.join(samples['ocr_tokens'][i][:30]), samples["text_input"][i])
                        for i in range(len(samples["text_input"]))]
                elif 'choices' in samples:
                    text_input = []
                    for i in range(len(samples["text_input"])):
                        this_choices = [f"({string.ascii_lowercase[j]}) {ch}" for j, ch in
                                        enumerate(samples["choices"][i])]
                        this_choices = " ".join(this_choices)
                        text_input.append(prompt.format(samples["text_input"][i], this_choices))
            else:
                text_input = [prompt.format(question) for question in samples["text_input"]]
        else:
            text_input = samples["text_input"]

        # text_input_new = []
        # for i in range(len(text_input)):
        #     text_input_new.append(samples["passages"][i][0]+"\n"+samples["passages"][i][1]+"\n"+text_input[i])
        # samples["prompt"] = text_input_new
        samples["prompt"] = text_input

        output_text = self.generate(
            samples,
            num_beams=num_beams,
            max_length=max_len,
            min_length=min_len,
            length_penalty=length_penalty
        )


        if "apply_lemmatizer" in samples.keys() and samples["apply_lemmatizer"]:
            output_text = self._lemmatize(output_text)

        return output_text

    def predict_class(
            self,
            samples,
            candidates,
            n_segments=1,
    ):
        
        self.llm_tokenizer.padding_side = "left"

        # If candidates is a list of lists, each sample has its candidates, then we need to iterate one by one
        if type(candidates[0]) == list:
            results = []

            for i in range(samples["image"].size(0)):
                this_sample = {
                    "image": samples["image"][i].unsqueeze(0),
                    "prompt": samples["prompt"],
                }

                if "text_input" in samples.keys():
                    this_sample["text_input"] = [samples["text_input"][i]]

                if 'context' in samples.keys():
                    this_sample['context'] = [samples["context"][i]]

                if 'history' in samples.keys():
                    this_sample['history'] = [samples["history"][i]]

                if 'caption' in samples.keys():
                    this_sample['caption'] = [samples["caption"][i]]

                this_result = self._predict_class(this_sample, candidates[i], n_segments)
                results.append(this_result)

            try:
                results = torch.cat(results, dim=0)
            except:
                results = [res.tolist()[0] for res in results]

            return results

        return self._predict_class(samples, candidates, n_segments)

    def _predict_class(
            self,
            samples,
            candidates,
            n_segments=1,
    ):
        image = samples["image"]
        prompt = samples["prompt"]

        bs = image.size(0)

        if isinstance(prompt, str):
            prompt = [prompt] * bs
        else:
            assert len(prompt) == bs, "The number of prompts must be equal to the batch size."

        if "text_input" in samples.keys():
            if type(samples["text_input"][0]) == list:
                prompt = [prompt[i].format(*samples["text_input"][i]) for i in range(len(prompt))]
            else:
                prompt = [prompt[i].format(samples["text_input"][i]) for i in range(len(prompt))]

        # scienceqa
        if 'context' in samples.keys() and samples['context'] != '':
            prompt = [f'context: {samples["context"][i]}. {prompt[i]}' for i in range(len(prompt))]

        # visual dialog
        if 'history' in samples.keys() and samples['history'][0] != '':
            prompt = [f'dialog history: {samples["history"][i]}\n{prompt[i]}' for i in range(len(prompt))]

        if 'caption' in samples.keys() and samples['caption'][0] != '':
            prompt = [f'This image has the caption "{samples["caption"][i]}". {prompt[i]}' for i in range(len(prompt))]
        # print(prompt)

        query_tokens = self.query_tokens.expand(bs, -1, -1)
        if self.qformer_text_input:
            text_Qformer = self.tokenizer(
                prompt,
                padding='longest',
                truncation=True,
                max_length=self.max_txt_len,
                return_tensors="pt"
            ).to(image.device)
            query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(image.device)
            Qformer_atts = torch.cat([query_atts, text_Qformer.attention_mask], dim=1)

        if image.dim() == 5:
            inputs_llm, atts_llm = [], []
            for j in range(image.size(2)):
                this_frame = image[:, :, j, :, :]
                with self.maybe_autocast():
                    frame_embeds = self.ln_vision(self.visual_encoder(this_frame))
                    frame_atts = torch.ones(frame_embeds.size()[:-1], dtype=torch.long).to(image.device)

                if self.qformer_text_input:
                    frame_query_output = self.Qformer.bert(
                        text_Qformer.input_ids,
                        attention_mask=Qformer_atts,
                        query_embeds=query_tokens,
                        encoder_hidden_states=frame_embeds,
                        encoder_attention_mask=frame_atts,
                        return_dict=True,
                    )
                else:
                    frame_query_output = self.Qformer.bert(
                        query_embeds=query_tokens,
                        encoder_hidden_states=frame_embeds,
                        encoder_attention_mask=frame_atts,
                        return_dict=True,
                    )

                frame_inputs_llm = self.llm_proj(frame_query_output.last_hidden_state[:, :query_tokens.size(1), :])
                frame_atts_llm = torch.ones(frame_inputs_llm.size()[:-1], dtype=torch.long).to(image.device)
                inputs_llm.append(frame_inputs_llm)
                atts_llm.append(frame_atts_llm)
            inputs_llm = torch.cat(inputs_llm, dim=1)
            atts_llm = torch.cat(atts_llm, dim=1)
        else:
            with self.maybe_autocast():
                image_embeds = self.ln_vision(self.visual_encoder(image))
            image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device)

            if self.qformer_text_input:
                query_output = self.Qformer.bert(
                    text_Qformer.input_ids,
                    attention_mask=Qformer_atts,
                    query_embeds=query_tokens,
                    encoder_hidden_states=image_embeds,
                    encoder_attention_mask=image_atts,
                    return_dict=True,
                )
            else:
                query_output = self.Qformer.bert(
                    query_embeds=query_tokens,
                    encoder_hidden_states=image_embeds,
                    encoder_attention_mask=image_atts,
                    return_dict=True,
                )

            inputs_llm = self.llm_proj(query_output.last_hidden_state[:, :query_tokens.size(1), :])
            atts_llm = torch.ones(inputs_llm.size()[:-1], dtype=torch.long).to(image.device)

        self.llm_tokenizer.padding_side = "right"
        self.llm_tokenizer.truncation_side = 'left'
        text_input_tokens = self.llm_tokenizer(
            prompt,
            return_tensors="pt",
            padding="longest",
            # truncation=True,
            # max_length=self.max_txt_len,
        ).to(image.device)

        empty_targets = torch.ones(atts_llm.size(), dtype=torch.long).to(image.device).fill_(-100)

        # self.llm_tokenizer.padding_side = "right"
        self.llm_tokenizer.truncation_side = 'right'
        n_cands = len(candidates)
        with self.maybe_autocast(dtype=torch.bfloat16):
            all_losses = []
            for n in range(n_segments):
                seg_len = n_cands // n_segments
                if n == (n_segments - 1):
                    seg_len = n_cands - seg_len * (n_segments - 1)

                start_i = n * (n_cands // n_segments)
                end_i = start_i + seg_len

                this_output_tokens = self.llm_tokenizer(
                    candidates[start_i:end_i],
                    return_tensors="pt",
                    padding="longest",
                    # truncation=True,
                    # max_length=self.max_output_txt_len,
                ).to(image.device)

                this_input_tokens_ids = text_input_tokens.input_ids.repeat_interleave(seg_len, dim=0)
                this_input_tokens_atts = text_input_tokens.attention_mask.repeat_interleave(seg_len, dim=0)

                this_output_tokens_ids = this_output_tokens.input_ids.repeat(bs, 1)
                this_output_tokens_atts = this_output_tokens.attention_mask.repeat(bs, 1)

                this_llm_tokens, this_input_targets_len = self.concat_text_input_output(
                    this_input_tokens_ids,
                    this_input_tokens_atts,
                    this_output_tokens_ids,
                    this_output_tokens_atts
                )

                this_llm_input_ids = this_llm_tokens['input_ids']
                this_llm_atts = this_llm_tokens['attention_mask']
                # this_llm_input_ids = torch.cat([this_input_tokens_ids, this_output_tokens_ids], dim=1)
                # this_llm_atts = torch.cat([this_input_tokens_atts, this_output_tokens_atts], dim=1)

                inputs_embeds = self.llm_model.get_input_embeddings()(this_llm_input_ids)
                inputs_embeds = torch.cat([inputs_llm.repeat_interleave(seg_len, dim=0), inputs_embeds], dim=1)
                attention_mask = torch.cat([atts_llm.repeat_interleave(seg_len, dim=0), this_llm_atts], dim=1)

                this_targets = this_llm_input_ids.masked_fill(this_llm_input_ids == self.llm_tokenizer.pad_token_id,
                                                              -100)
                # this_targets[:, :this_input_tokens_ids.size(1)] = -100
                for i, l in enumerate(this_input_targets_len):
                    this_targets[i][:l] = -100

                this_targets = torch.cat([empty_targets.repeat_interleave(seg_len, dim=0), this_targets], dim=1)
                outputs = self.llm_model(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    return_dict=True,
                    labels=this_targets,
                    reduction="none",
                )

                loss = outputs.loss

                loss = loss.reshape(bs, seg_len)
                # output_class_ranks = torch.argsort(loss, dim=-1)
                all_losses.append(loss)

            all_losses = torch.cat(all_losses, dim=-1)
            output_class_ranks = torch.argsort(all_losses, dim=-1)


        return output_class_ranks

    def _lemmatize(self, answers):
        def apply(answer):
            doc = self.lemmatizer(answer)

            words = []
            for token in doc:
                if token.pos_ in ["NOUN", "VERB"]:
                    words.append(token.lemma_)
                else:
                    words.append(token.text)
            answer = " ".join(words)

            return answer

        return [apply(answer) for answer in answers]

    @property
    def lemmatizer(self):
        if self._lemmatizer is None:
            try:
                import spacy

                self._lemmatizer = spacy.load("en_core_web_sm")
            except ImportError:
                logging.error(
                    """
                    Please install spacy and en_core_web_sm model to apply lemmatization.
                    python -m spacy download en_core_web_sm
                    OR
                    import spacy.cli
                    spacy.cli.download("en_core_web_sm")
                    """
                )
                exit(1)

        return self._lemmatizer

    @classmethod
    def from_config(cls, cfg):
        vit_model = cfg.get("vit_model", "eva_clip_g")
        img_size = cfg.get("image_size")
        num_query_token = cfg.get("num_query_token")
        llm_model = cfg.get("llm_model")

        drop_path_rate = cfg.get("drop_path_rate", 0)
        use_grad_checkpoint = cfg.get("use_grad_checkpoint", False)
        vit_precision = cfg.get("vit_precision", "fp16")
        freeze_vit = cfg.get("freeze_vit", True)

        prompt = cfg.get("prompt", "")
        max_txt_len = cfg.get("max_txt_len", 128)
        max_output_txt_len = cfg.get("max_output_txt_len", 256)

        apply_lemmatizer = cfg.get("apply_lemmatizer", False)

        qformer_text_input = cfg.get("qformer_text_input", True)

        model = cls(
            vit_model=vit_model,
            img_size=img_size,
            drop_path_rate=drop_path_rate,
            use_grad_checkpoint=use_grad_checkpoint,
            vit_precision=vit_precision,
            freeze_vit=freeze_vit,
            num_query_token=num_query_token,
            llm_model=llm_model,
            prompt=prompt,
            max_txt_len=max_txt_len,
            max_output_txt_len=max_output_txt_len,
            apply_lemmatizer=apply_lemmatizer,
            qformer_text_input=qformer_text_input,
        )

        # if qformer_text_input:
        #     # Hard-coded to load from BLIP-2 stage-1 pre-trained model (not ideal)
        #     model.load_from_pretrained(
        #         url_or_filename="https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP2/blip2_pretrained.pth"
        #     )
        # print(model)
        # exit()

        model.load_checkpoint_from_config(cfg)

        model.Qformer2.load_state_dict(model.Qformer.state_dict())
        model.llm_proj2.load_state_dict(model.llm_proj.state_dict())
        model.Qformer3.load_state_dict(model.Qformer.state_dict())
        model.llm_proj3.load_state_dict(model.llm_proj.state_dict())
        model.Qformer4.load_state_dict(model.Qformer.state_dict())
        model.llm_proj4.load_state_dict(model.llm_proj.state_dict())



        return model