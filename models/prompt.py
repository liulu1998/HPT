import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from transformers import AutoTokenizer
from transformers.modeling_outputs import (
    MaskedLMOutput
)
from transformers.models.bert.modeling_bert import BertModel, BertPreTrainedModel, BertOnlyMLMHead

from .graph import GraphEncoder
from .loss import multilabel_categorical_crossentropy


class GraphEmbeddingWithSoftPrompt(nn.Module):
    def __init__(self, config, embedding, new_embedding, soft_prompt_embedding, graph_type='GAT', layer=1,
                 path_list=None, data_path=None):
        """

        :param config:
        :param embedding: 传入的是 input_embeddings, BERT 的原始 embedding
        :param new_embedding: 传入的是 label_emb, [n_label + n_depth + 1, hidden_size]
        :param soft_prompt_embedding: LiuLu 新增的参数
        :param graph_type:
        :param layer:
        :param path_list:
        :param data_path:
        """
        super(GraphEmbeddingWithSoftPrompt, self).__init__()
        self.graph_type = graph_type
        padding_idx = config.pad_token_id
        self.num_class = config.num_labels

        if self.graph_type != '':
            self.graph = GraphEncoder(config, graph_type, layer, path_list=path_list, data_path=data_path)

        self.padding_idx = padding_idx

        # input_embeddings
        self.original_embedding = embedding

        ## 在 new_embedding 前，拼接一个 zeroes
        # 在 label_embeddings 前面拼一个 zeroes
        new_embedding = torch.cat(
            [
                # shape: ( 1, hidden size )
                torch.zeros(
                    1, new_embedding.size(-1),
                    device=new_embedding.device, dtype=new_embedding.dtype
                ),
                new_embedding
            ],
            dim=0
        )
        self.new_embedding = nn.Embedding.from_pretrained(new_embedding, freeze=False, padding_idx=0)

        # new added, Soft Prompt token embeddings
        self.soft_prompt_embedding = nn.Embedding.from_pretrained(
            soft_prompt_embedding.weight, freeze=False, padding_idx=0
        )
        self.n_soft_token = self.soft_prompt_embedding.weight.size(0)

        self.size = self.original_embedding.num_embeddings + self.new_embedding.num_embeddings - 1
        self.depth = (self.new_embedding.num_embeddings - 2 - self.num_class)

    @property
    def weight(self):
        def foo():
            # label prompt MASK
            # # 不取第一个 zeroes
            edge_features = self.new_embedding.weight[1:, :]
            if self.graph_type != '':
                # label prompt
                edge_features = edge_features[:-1, :]
                edge_features = self.graph(edge_features, self.original_embedding)
                edge_features = torch.cat(
                    [edge_features, self.new_embedding.weight[-1:, :]],
                    dim=0
                )
            # 前面是 BERT 原始 emb, 后面是 label_emb, depth_emb, MASK emb, 注意 dim = 0
            # ! 确保 input_id 能正确索引, 不加入 soft prompt emb
            # BERT emb, label emb (depth emb, MASK emb)
            return torch.cat(
                [self.original_embedding.weight, edge_features],
                dim=0
            )

        # << foo
        return foo

    @property
    def raw_weight(self):
        def foo():
            # BERT emb, label emb (depth emb, MASK emb)
            # todo: 无需加入 soft prompt emb ? 加入后可能扰乱 id
            return torch.cat(
                [self.original_embedding.weight, self.new_embedding.weight[1:, :]],
                dim=0
            )

        return foo

    def forward(self, x):
        """

        :param x: supposed to be 2D tensor ?
        :return:
        """
        # print("Graph Embedding with Soft Prompt.forward called !")

        batch_size = x.size(0)

        soft_prompt_emb = self.soft_prompt_embedding.weight
        # 增广
        soft_prompt_emb = soft_prompt_emb.repeat(batch_size, 1, 1)

        other_emb = F.embedding(x[:, self.n_soft_token:], self.weight(), self.padding_idx)

        # return x
        # for each sample, [soft prompt token emb, BERT emb, label emb]
        # shape [batch size, seq len, hidden size], i.e. [batch size, 512, 768]
        return torch.cat([soft_prompt_emb, other_emb], dim=1)


class OutputEmbedding(nn.Module):
    def __init__(self, bias):
        super(OutputEmbedding, self).__init__()
        self.weight = None
        self.bias = bias

    def forward(self, x):
        return F.linear(x, self.weight(), self.bias)


# class SoftEmbedding(nn.Module):
#     def __init__(self,
#                  wte: nn.Embedding,
#                  n_tokens: int = 10,
#                  random_range: float = 0.5,
#                  initialize_from_vocab: bool = True):
#         """appends learned embedding to
#         Args:
#             wte (nn.Embedding): original transformer word embedding
#             n_tokens (int, optional): number of tokens for task. Defaults to 10.
#             random_range (float, optional): range to init embedding (if not initialize from vocab). Defaults to 0.5.
#             initialize_from_vocab (bool, optional): initializes from default vocab. Defaults to True.
#         """
#         super(SoftEmbedding, self).__init__()
#         # ! wte 是 GraphEmbedding 类
#         self.wte = wte
#         self.n_tokens = n_tokens
#         self.learned_embedding = nn.Parameter(
#             self.initialize_embedding(self.wte, self.n_tokens, random_range, initialize_from_vocab)
#         )
#
#     def initialize_embedding(self,
#                              wte: nn.Embedding,
#                              n_tokens: int = 10,
#                              random_range: float = 0.5,
#                              initialize_from_vocab: bool = True):
#         """initializes learned embedding
#         Args:
#             same as __init__
#         Returns:
#             torch.float: initialized using original schemes
#         """
#         if initialize_from_vocab:
#             return self.wte.weight[:n_tokens].clone().detach()
#         else:
#             return torch.FloatTensor(n_tokens, wte.weight.size(1)).uniform_(-random_range, random_range)
#
#     def forward(self, tokens):
#         """run forward pass
#         Args:
#             tokens (torch.long): input tokens before encoding
#         Returns:
#             torch.float: encoding of text concatenated with learned task specifc embedding
#         """
#         # todo: 这里需要修改
#         input_embedding = self.wte(tokens[:, self.n_tokens:])
#         learned_embedding = self.learned_embedding.repeat(input_embedding.size(0), 1, 1)
#         return torch.cat([learned_embedding, input_embedding], 1)


class Prompt(BertPreTrainedModel):
    # ! 下方 demo 展示了 Soft Prompt tuning
    # https://github.com/qhduan/mt5-soft-prompt-tuning/blob/main/mt5_soft_prompt_tuning_large.ipynb
    # 冻结了 BERT 的参数，只训练 Prompt 的参数 (Soft Prompt Embedding)
    """
    Prompt.forward called !
    Graph Embedding with Soft Prompt.forward called !
    """

    _keys_to_ignore_on_load_unexpected = [r"pooler"]
    _keys_to_ignore_on_load_missing = [r"position_ids", r"predictions.decoder.bias"]

    def __init__(self, config, graph_type='GAT', layer=1, path_list=None, data_path=None, depth2label=None,
                 n_soft_token: int = 100, **kwargs):
        super().__init__(config)

        self.bert = BertModel(config, add_pooling_layer=False)
        # # 冻结 BERT 参数
        # for name, param in self.bert.named_parameters():
        #     param.requires_grad = False

        self.tokenizer = AutoTokenizer.from_pretrained(self.name_or_path)
        self.cls = BertOnlyMLMHead(config)

        self.num_labels = config.num_labels
        self.multiclass_bias = nn.Parameter(torch.zeros(self.num_labels, dtype=torch.float32))
        bound = 1 / math.sqrt(768)
        nn.init.uniform_(self.multiclass_bias, -bound, bound)
        self.data_path = data_path
        self.graph_type = graph_type
        self.vocab_size = self.tokenizer.vocab_size
        self.path_list = path_list
        self.depth2label = depth2label
        self.layer = layer

        # soft prompt token 个数
        self.n_soft_token = n_soft_token
        self.init_weights()

    def get_output_embeddings(self):
        return self.cls.predictions.decoder

    def set_output_embeddings(self, new_embeddings):
        self.cls.predictions.decoder = new_embeddings

    def init_embedding(self):
        depth = len(self.depth2label)
        label_dict = torch.load(os.path.join(self.data_path, 'value_dict.pt'))
        tokenizer = AutoTokenizer.from_pretrained(self.name_or_path)
        label_dict = {i: tokenizer.encode(v) for i, v in label_dict.items()}

        # label 的表示，以 label name words 的词向量平均值 初始化
        label_emb = []

        # 从语料初始化 label emb
        input_embeds = self.get_input_embeddings()
        for i in range(len(label_dict)):
            label_emb.append(
                input_embeds.weight.index_select(0, torch.tensor(label_dict[i], device=self.device)).mean(dim=0)
            )

        # [MASK] 的词向量
        prefix = input_embeds(torch.tensor([tokenizer.mask_token_id], device=self.device, dtype=torch.long))

        # prefix soft prompt embeddings
        # [n_soft_prompt_tokens, hidden size]
        soft_prompt_embedding = nn.Embedding(self.n_soft_token, input_embeds.weight.size(1), 0)
        self._init_weights(soft_prompt_embedding)
        # todo: init from corpus OR follow PPT(ACL 2022) / SPoT (ACL 2022) to pre-train soft prompt

        # postfix prompt embeddings ( [V_i][Pred_i]... )
        # 虚拟标签词 的 embedding （即层次编码）, prompt_embedding[0] 是 Root 的编码 ?
        prompt_embedding = nn.Embedding(depth + 1, input_embeds.weight.size(1), 0)

        self._init_weights(prompt_embedding)
        # label prompt mask, [n_label + n_depth + 1, hidden size]
        label_emb = torch.cat(
            [torch.stack(label_emb), prompt_embedding.weight[1:, :], prefix],
            dim=0
        )

        # (config, embedding, new_embedding, graph_type='GAT', layer=1, path_list=None, data_path=None)
        embedding = GraphEmbeddingWithSoftPrompt(
            config=self.config, embedding=input_embeds, new_embedding=label_emb,
            soft_prompt_embedding=soft_prompt_embedding, graph_type=self.graph_type,
            path_list=self.path_list, layer=self.layer, data_path=self.data_path
        )

        # ! 在这将 input_embedding 设为了 GraphEmbedding
        self.set_input_embeddings(embedding)

        # todo: 这里需要修改
        output_embeddings = OutputEmbedding(self.get_output_embeddings().bias)
        self.set_output_embeddings(output_embeddings)
        output_embeddings.weight = embedding.raw_weight
        self.vocab_size = output_embeddings.bias.size(0)
        output_embeddings.bias.data = nn.functional.pad(
            output_embeddings.bias.data,
            (
                0,
                embedding.size - output_embeddings.bias.shape[0],
            ),
            "constant",
            0,
        )

    def get_layer_features(self, layer, prompt_feature=None):
        labels = torch.tensor(self.depth2label[layer], device=self.device) + 1
        label_features = self.get_input_embeddings().new_embedding(labels)
        label_features = self.transform(label_features)
        label_features = torch.dropout(F.relu(label_features), train=self.training, p=self.config.hidden_dropout_prob)
        return label_features

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            labels=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the masked language modeling loss. Indices should be in ``[-100, 0, ...,
            config.vocab_size]`` (see ``input_ids`` docstring) Tokens with indices set to ``-100`` are ignored
            (masked), the loss is only computed for the tokens with labels in ``[0, ..., config.vocab_size]``
        """
        # print("Prompt.forward called !")

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        multiclass_pos = input_ids == (self.get_input_embeddings().size - 1)

        # ! 即，不打算预测的，「label设置为 - 100」。一般只设置[MASK]
        # 位置对应的label，其它位置设置成 -100。这样只计算了[MASK], soft prompt id 固定为 1
        # 待预测位置的 token 对应的 loss
        single_labels = input_ids.masked_fill(
            multiclass_pos | (input_ids == self.config.pad_token_id) | (input_ids == 1),
            -100
        )

        # 文章保留了 MLM 任务，即：随机 mask 掉一部分 token，预测
        if self.training:
            # enable_mask = input_ids < self.tokenizer.vocab_size
            # ! 1 为 soft prompt 的 id
            # todo 还需要 input_ids != 1
            enable_mask = input_ids < self.tokenizer.vocab_size
            random_mask = torch.rand(input_ids.shape, device=input_ids.device) * attention_mask * enable_mask
            input_ids = input_ids.masked_fill(random_mask > 0.865, self.tokenizer.mask_token_id)
            random_ids = torch.randint_like(input_ids, 104, self.vocab_size)
            mlm_mask = random_mask > 0.985
            input_ids = input_ids * mlm_mask.logical_not() + random_ids * mlm_mask
            mlm_mask = random_mask < 0.85
            single_labels = single_labels.masked_fill(mlm_mask, -100)

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]
        prediction_scores = self.cls(sequence_output)

        masked_lm_loss = None

        if labels is not None:
            loss_fct = CrossEntropyLoss()  # -100 index = padding token
            # MLM 任务损失
            masked_lm_loss = loss_fct(prediction_scores.view(-1, prediction_scores.size(-1)),
                                      single_labels.view(-1))

            # 论文自定义的损失
            multiclass_logits = prediction_scores.masked_select(
                multiclass_pos.unsqueeze(-1).expand(-1, -1, prediction_scores.size(-1))).view(-1,
                                                                                              prediction_scores.size(
                                                                                                  -1))
            multiclass_logits = multiclass_logits[:,
                                self.vocab_size:self.vocab_size + self.num_labels] + self.multiclass_bias
            multiclass_loss = multilabel_categorical_crossentropy(labels.view(-1, self.num_labels), multiclass_logits)
            masked_lm_loss += multiclass_loss

        if not return_dict:
            output = (prediction_scores,) + outputs[2:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        # 结果包装一下，非运算
        ret = MaskedLMOutput(
            loss=masked_lm_loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
        return ret

    def prepare_inputs_for_generation(self, input_ids, attention_mask=None, **model_kwargs):
        input_shape = input_ids.shape
        effective_batch_size = input_shape[0]

        #  add a dummy token
        assert self.config.pad_token_id is not None, "The PAD token should be defined for generation"
        attention_mask = torch.cat([attention_mask, attention_mask.new_zeros((attention_mask.shape[0], 1))], dim=-1)
        dummy_token = torch.full(
            (effective_batch_size, 1), self.config.pad_token_id, dtype=torch.long, device=input_ids.device
        )
        input_ids = torch.cat([input_ids, dummy_token], dim=1)

        return {"input_ids": input_ids, "attention_mask": attention_mask}

    @torch.no_grad()
    def generate(self, input_ids, depth2label, **kwargs):
        attention_mask = input_ids != self.config.pad_token_id
        outputs = self(input_ids, attention_mask)
        multiclass_pos = input_ids == (self.get_input_embeddings().size - 1)
        prediction_scores = outputs['logits']
        prediction_scores = prediction_scores.masked_select(
            multiclass_pos.unsqueeze(-1).expand(-1, -1, prediction_scores.size(-1))).view(-1,
                                                                                          prediction_scores.size(
                                                                                              -1))
        prediction_scores = prediction_scores[:,
                            self.vocab_size:self.vocab_size + self.num_labels] + self.multiclass_bias
        prediction_scores = prediction_scores.view(-1, len(depth2label), prediction_scores.size(-1))
        predict_labels = []
        for scores in prediction_scores:
            predict_labels.append([])
            for i, score in enumerate(scores):
                for l in depth2label[i]:
                    if score[l] > 0:
                        predict_labels[-1].append(l)
        return predict_labels, prediction_scores

# class SoftPromptClassifier(nn.Module):
#     def __init__(self, backbone, n_soft_tokens: int = 100):
#         super(SoftPromptClassifier, self).__init__()
#
#         self.backbone = backbone
#         # 冻结 BERT 参数
#         for name, param in self.backbone.named_parameters():
#             param.requires_grad = False
#
#         self.n_soft_tokens = n_soft_tokens
#
#         self.soft_prompt_emb = SoftEmbedding(
#             wte=self.backbone.get_input_embeddings(),
#             n_tokens=self.n_soft_tokens,
#             initialize_from_vocab=False
#         )
#
#         self.backbone.set_input_embeddings(self.soft_prompt_emb)
#
#     def forward(self, **kwargs):
#         return self.backbone(**kwargs)
