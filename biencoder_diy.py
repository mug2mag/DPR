from transformers import AutoTokenizer, AutoModel
import torch


def get_last_hidden_state():
    # First we initialize our model and tokenizer:

    tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/bert-base-nli-mean-tokens')
    model = AutoModel.from_pretrained('sentence-transformers/bert-base-nli-mean-tokens')

    # Then we tokenize the sentences just as before:

    sentences = [
        "Three years later, the coffin was still full of Jello.",
        "The fish dreamed of escaping the fishbowl and into the toilet where he saw his friend go.",
        "The person box was packed with jelly many dozens of months later.",
        "He found a leprechaun in his walnut shell."
    ]

    # 初始化字典来存储
    tokens = {'input_ids': [], 'attention_mask': []}

    for sentence in sentences:
        # 编码每个句子并添加到字典
        new_tokens = tokenizer.encode_plus(sentence, max_length=128,
                                           truncation=True, padding='max_length',
                                           return_tensors='pt')
        tokens['input_ids'].append(new_tokens['input_ids'][0])
        tokens['attention_mask'].append(new_tokens['attention_mask'][0])

    # 将张量列表重新格式化为一个张量
    tokens['input_ids'] = torch.stack(tokens['input_ids'])
    tokens['attention_mask'] = torch.stack(tokens['attention_mask'])

    # We process these tokens through our model:

    outputs = model(**tokens)
    # outputs.keys()

    #odict_keys(['last_hidden_state', 'pooler_output'])

    #The dense vector representations of our text are contained within the outputs 'last_hidden_state' tensor, which we access like so:

    # embeddings = outputs.last_hidden_state
    embeddings = outputs[0]
    return embeddings, tokens


def compute_mean_pooled(tokens, embeddings):

    # To perform this operation, we first resize our attention_mask tensor:
    attention_mask = tokens['attention_mask']

    # attention_mask.shape
    # torch.Size([4, 128])

    mask = attention_mask.unsqueeze(-1).expand(embeddings.size()).float()

    # mask.shape
    # torch.Size([4, 128, 768])

    # 上面的每个向量表示一个单独token的掩码——现在每个token都有一个大小为768的向量，表示它的attention_mask状态。然后将两个张量相乘:

    masked_embeddings = embeddings * mask

    # masked_embeddings.shape
    # torch.Size([4, 128, 768])
    # 然后我们沿着轴1将剩余的嵌入项求和:

    summed = torch.sum(masked_embeddings, 1)
    # summed.shape
    # torch.Size([4, 768])

    # 然后将张量的每个位置上的值相加:

    summed_mask = torch.clamp(mask.sum(1), min=1e-9)
    # summed_mask.shape
    # torch.Size([4, 768])
    # 最后，我们计算平均值:

    mean_pooled = summed / summed_mask
    return mean_pooled


def compute_similarity(mean_pooled):
    """
    一旦我们有了密集向量，我们就可以计算每个向量之间的余弦相似性：
    """
    from sklearn.metrics.pairwise import cosine_similarity

    #让我们计算第0句的余弦相似度:

    # 将PyTorch张量转换为numpy数组
    mean_pooled = mean_pooled.detach().numpy()

    # 计算
    simi_list = cosine_similarity(
        [mean_pooled[0]],
        mean_pooled[1:]
    )
    # array([[0.33088905, 0.7219259, 0.55483633]], dtype=float32)

    # These similarities translate to:
    return simi_list


if __name__ == '__main__':
    embeddings, tokens = get_last_hidden_state()
    mean_pooled = compute_mean_pooled(tokens, embeddings)
    print(compute_similarity(mean_pooled))

    # 1. DPR 训练代码里计算相似度；看来不用：（相关工作指出，相比BERT直接的输出计算loss，后面再加一些基本MLP将会大有帮助，比如原Bert输出句子嵌入、直接计算cosine的结果会比GloVe还差，但是Bert输出后面简单再加一层再进行计算，结果会远远超过原来的）
    # 2. cat(还是stack?) query和answer时 也要有对应的attention_mask

