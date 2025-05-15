import networkx as nx
import torch
from collections import defaultdict
import numpy as np
import scipy.sparse as sp
import json
from tqdm import tqdm
import re
import string
from collections import Counter

def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)
        # return re.sub(f"[{re.escape(string.punctuation)}]", " ", text)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))    

def get_score_of_chunk(normalized_prediction, normalized_ground_truth):
    # normalized_ground_truth = normalize_answer(ground_truth)
    accuracy = 1 if normalized_ground_truth in normalized_prediction else 0
    return accuracy

from hashlib import md5
def mdhash_id(content, prefix: str = ""):
    return prefix + md5(content.encode()).hexdigest()

import torch
import networkx as nx
import scipy.sparse as sp
import numpy as np

import tiktoken
def encode_string_by_tiktoken(content: str, model_name: str = "cl100k_base"):
    ENCODER = tiktoken.get_encoding(model_name)
    tokens = ENCODER.encode(content)
    return tokens

def decode_string_by_tiktoken(tokens: list[int], model_name: str = "cl100k_base"):
    ENCODER = tiktoken.get_encoding(model_name)
    string = ENCODER.decode(tokens)
    return string

RESPONSE_PROMPT = """
Your goal is to give the best full answer to question the user input according to the given context below.

Given Context: {context_data}

Give the best full answer to question :{question}.

IMPORTANT! JUST OUTPUT THE ANSWER BELOW!

Answer: 
"""


import torch
import networkx as nx

def compute_similarity_and_multiply(matrix, threshold, chunk_size=1000):
    N, D = matrix.shape
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 归一化矩阵（直接在 GPU 进行）
    matrix = matrix.to(device)
    norm_matrix = matrix / torch.norm(matrix, dim=1, keepdim=True)

    # 创建一个稀疏存储的相似度矩阵
    indices = []
    values = []

    # 逐块计算余弦相似度
    from tqdm import tqdm
    for i in tqdm(range(0, N, chunk_size)):
        end = min(i + chunk_size, N)
        cos_sim_chunk = torch.mm(norm_matrix[i:end], norm_matrix.T)  # 计算当前块与整个矩阵的相似度
        mask = cos_sim_chunk > threshold  # 仅保留大于阈值的部分
        selected_indices = mask.nonzero(as_tuple=False).T  # 获取索引
        
        # 偏移行索引
        selected_indices[0] += i
        indices.append(selected_indices)
        values.append(cos_sim_chunk[mask])  # 仅保留符合阈值的值
    
    # 构造稀疏矩阵
    if indices:
        indices = torch.cat(indices, dim=1)
        values = torch.cat(values)
        sparse_binary = torch.sparse_coo_tensor(indices, values, (N, N), dtype=torch.float32, device=device).coalesce()
    else:
        sparse_binary = torch.sparse_coo_tensor(torch.empty((2, 0), dtype=torch.long, device=device),
                                                torch.empty(0, dtype=torch.float32, device=device),
                                                (N, N)).coalesce()

    # 生成下三角掩码
    row, col = sparse_binary.indices()
    mask = row > col  # 仅保留下三角部分
    lower_tri_indices = sparse_binary.indices()[:, mask]
    lower_tri_values = sparse_binary.values()[mask]

    # 结果矩阵
    result = torch.sparse_coo_tensor(lower_tri_indices, lower_tri_values, (N, N), dtype=torch.float32, device=device)

    return result

def recall_at_k(target_list, predict_list, k=5):
    """
    计算Recall@k
    
    参数:
    target_list: 真实相关的项目列表
    predict_list: 预测的项目列表
    k: 考虑的前k个预测项目
    
    返回:
    recall@k的值
    """
    # 将target_list转换为集合便于快速查找
    target_set = set(target_list)
    
    # 取前k个预测项目
    top_k_predictions = predict_list[:k]
    
    # 计算在前k个预测中有多少是相关的
    pre_right = set()
    for item in top_k_predictions:
        if item in target_set:
            pre_right.add(item)
    relevant_in_top_k = len(pre_right)
    
    # 计算总的相关项目数
    total_relevant = len(target_set)
    
    # 避免除以0的情况
    if total_relevant == 0:
        return 0.0
    
    
    return relevant_in_top_k / total_relevant, list(pre_right), list(target_set - pre_right)

import networkx as nx
from collections import defaultdict
import math

NER = """
Your task is to extract named entities from the given paragraph. 
Respond with a JSON list of entities.
Example:
Input: Radio City is India's first private FM radio station and was started on 3 July 2001. It plays Hindi, English and regional songs. Radio City recently forayed into New Media in May 2008 with the launch of a music portal - PlanetRadiocity.com that offers music related news, videos, songs, and other music-related features.
Output: {{"named_entities":["Radio City", "India", "3 July 2001", "Hindi", "English", "May 2008", "PlanetRadiocity.com"]}}

Input: {passage}
Output:
"""

NER_RELATION = """Your task is to construct an RDF (Resource Description Framework) graph from the given passages and named entity lists. 
Respond with a JSON list of triples, with each triple representing a relationship in the RDF graph. 

Pay attention to the following requirements:
- Each triple should contain at least one, but preferably two, of the named entities in the list for each passage.
- Clearly resolve pronouns to their specific names to maintain clarity.

# Here is an example for your reference:

[Example]

Paragraph:

Radio City
Radio City is India's first private FM radio station and was started on 3 July 2001.
It plays Hindi, English and regional songs.
Radio City recently forayed into New Media in May 2008 with the launch of a music portal - PlanetRadiocity.com that offers music related news, videos, songs, and other music-related features.

Here is the example entity list:

{{
"named_entities": ["Radio City", "India", "3 July 2001", "Hindi", "English", "May 2008", "PlanetRadiocity.com"]
}}

Based on the entity above, the triple list should be:

{{"triples": [
            ["Radio City", "located in", "India"],
            ["Radio City", "is", "private FM radio station"],
            ["Radio City", "started on", "3 July 2001"],
            ["Radio City", "plays songs in", "Hindi"],
            ["Radio City", "plays songs in", "English"]
            ["Radio City", "forayed into", "New Media"],
            ["Radio City", "launched", "PlanetRadiocity.com"],
            ["PlanetRadiocity.com", "launched in", "May 2008"],
            ["PlanetRadiocity.com", "is", "music portal"],
            ["PlanetRadiocity.com", "offers", "news"],
            ["PlanetRadiocity.com", "offers", "videos"],
            ["PlanetRadiocity.com", "offers", "songs"]
    ]
}}

Now, please output high quality, precise, and concise triples into a json format:
Paragraph:
```
{passage}
```
named_entity: {named_entity}
Based on the entity above, the triple list should be:

"""

NER_EXTRACTION = """
Task:
Extract all relevant entities from the given question. 
Return the results in a structured JSON format.

[ Example-1 ]
Input: How far is the Eiffel Tower from Charles de Gaulle Airport in Paris?
Output: {{"named_entities":["Eiffel Tower", "Charles de Gaulle Airport", "Paris"]}}
[ Example-2 ]
Input: Why did Lionel Messi leave Barcelona for Inter Miami in 2023?
Output: {{"named_entities":["Lionel Messi", "Barcelona", "Inter Miami", "2023"]}}

DON'T OUTPUT YOUR THOUGHTS OR THINKING PROCESS!
ONLY OUTPUT THE JSON FORMAT!
[ Real Case ]
Input: {passage}
Output: 
"""

PROPOSITION_NER_PROMPT = """
Decompose the "Content" into clear and simple propositions, ensuring they are interpretable out of context.
Then extracte the entities from the propositions.

1. Split compound sentence into simple sentences. Maintain the original phrasing from the input whenever possible.
2. For any named entity that is accompanied by additional descriptive information, separate this information into its own distinct proposition.
3. Pronoun Elimination: Replace ALL pronouns (it, they, this, etc.) with full taxonomic names or explicit references. Never use possessives (its, their), always use "[entity]'s [property]" construction.
4. Extract relevent entities from the output propositions.
5. Present the results as a list of strings, formatted in JSON, whose key is the extracted proposition, and value is the corresponding entity list.

Example-1:
Input: Jes\u00fas Aranguren. His 13-year professional career was solely associated with Athletic Bilbao, with which he played in nearly 400 official games, winning two Copa del Rey trophies.
Output: 
{{ 
"Jesús Aranguren had a 13-year professional career.":["Jesús Aranguren", "13-year professional career"],
"Jesús Aranguren's professional career was solely associated with Athletic Bilbao.": ["Jesús Aranguren", "Athletic Bilbao"],
"Athletic Bilbao is a football club.": ["Athletic Bilbao", "football club"],
"Jesús Aranguren played for Athletic Bilbao in nearly 400 official games.": ["Jesús Aranguren", "Athletic Bilbao"],
"Jesús Aranguren won two Copa del Rey trophies with Athletic Bilbao.": ["Jesús Aranguren", "Copa del Rey trophies", "Athletic Bilbao"],
}}

JUST OUTPUT THE RESULT IN JSON FORMAT! DON"T OUTPUT ANYTHING INRELEVENT!
Input: {passage}
Output:
"""

def read_json(file_path):
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    f.close()
    return data

def wrtie_json(data:list, file_path:str, mode="w"):
    with open(file_path, mode) as f:
        for item in data:
            f.write(json.dumps(item) + "\n")
    f.close()
    
    
def parse_value_from_string(value: str):
    """
    Parse a value from a string, attempting to convert it into the appropriate type.

    Args:
        value: The string value to parse.

    Returns:
        The value converted to its appropriate type (e.g., int, float, bool, str).
    """
    try:
        if value.lower() == 'true':
            return True
        elif value.lower() == 'false':
            return False
        elif value.isdigit():
            return int(value)
        else:
            return float(value) if '.' in value else value.strip('"')
    except ValueError:
        return value

import html
import re
def clean_str(input) -> str:
    """Clean an input string by removing HTML escapes, control characters, and other unwanted characters."""
    # If we get non-string input, just give it back
    if not isinstance(input, str):
        return input

    result = html.unescape(input.strip())
    # https://stackoverflow.com/questions/4324790/removing-control-characters-from-a-string-in-python
    result = re.sub(r"[\x00-\x1f\x7f-\x9f]", "", result)

    # Remove non-alphanumeric characters and convert to lowercase
    return re.sub('[^A-Za-z0-9 ]', ' ', result.lower()).strip()

def prase_json_from_response(response: str) -> dict:
    """
    Extract JSON data from a string response.

    This function attempts to extract the first complete JSON object from the response.
    If that fails, it tries to extract key-value pairs from a potentially malformed JSON string.

    Args:
        response: The string response containing JSON data.
    Returns:
        A dictionary containing the extracted JSON data.
    """
    stack = []
    first_json_start = None

    # Attempt to extract the first complete JSON object using a stack to track braces
    for i, char in enumerate(response):
        if char == '{':
            stack.append(i)
            if first_json_start is None:
                first_json_start = i
        elif char == '}':
            if stack:
                start = stack.pop()
                if not stack:
                    first_json_str = response[first_json_start:i + 1]
                    try:
                        # Attempt to parse the JSON string
                        return json.loads(first_json_str.replace("\n", ""))
                    except json.JSONDecodeError as e:
                        #logger.error(f"JSON decoding failed: {e}. Attempted string: {first_json_str[:50]}...")
                        break
                    finally:
                        first_json_start = None

    # If extraction of complete JSON failed, try extracting key-value pairs from a non-standard JSON string
    extracted_values = {}
    regex_pattern = r'(?P<key>"?\w+"?)\s*:\s*(?P<value>{[^}]*}|".*?"|[^,}]+)'

    for match in re.finditer(regex_pattern, response, re.DOTALL):
        key = match.group('key').strip('"')  # Strip quotes from key
        value = match.group('value').strip()

        # If the value is another nested JSON (starts with '{' and ends with '}'), recursively parse it
        if value.startswith('{') and value.endswith('}'):
            extracted_values[key] = prase_json_from_response(value)
        else:
            # Parse the value into the appropriate type (int, float, bool, etc.)
            extracted_values[key] = parse_value_from_string(value)

    if not extracted_values:
        #logger.warning("No values could be extracted from the string.")
        pass
    else:
        #logger.info("JSON data successfully extracted.")
        pass

    return extracted_values
    
import re
    
import math
from collections import defaultdict
import re
import networkx as nx

import html
import re
import torch
from dataclasses import dataclass, asdict, field
from collections import Counter

from collections import deque


PROPOSITION_PROMPT = """
Decompose the "Content" into clear and simple propositions, ensuring they are interpretable out of context.
1. Split compound sentence into simple sentences. Maintain the original phrasing from the input whenever possible.
2. For any named entity that is accompanied by additional descriptive information, separate this information into its own distinct proposition.
3. Pronoun Elimination: Replace ALL pronouns (it, they, this, etc.) with full taxonomic names or explicit references. Never use possessives (its, their), always use "[entity]'s [property]" construction.
4. Present the results as a list of strings, formatted in JSON.

Example-1:
Input: Jes\u00fas Aranguren. His 13-year professional career was solely associated with Athletic Bilbao, with which he played in nearly 400 official games, winning two Copa del Rey trophies.
Output: {{ "propositions": [ "Jesús Aranguren had a 13-year professional career.", "Jesús Aranguren's professional career was solely associated with Athletic Bilbao.", "Athletic Bilbao is a football club.", "Jesús Aranguren played for Athletic Bilbao in nearly 400 official games.", "Jesús Aranguren won two Copa del Rey trophies with Athletic Bilbao."]}}

Example-2:
Input: Ophrys apifera. Ophrys apifera grows to a height of 15 -- 50 centimetres (6 -- 20 in). This hardy orchid develops small rosettes of leaves in autumn. They continue to grow slowly during winter. Basal leaves are ovate or oblong - lanceolate, upper leaves and bracts are ovate - lanceolate and sheathing. The plant blooms from mid-April to July producing a spike composed from one to twelve flowers. The flowers have large sepals, with a central green rib and their colour varies from white to pink, while petals are short, pubescent, yellow to greenish. The labellum is trilobed, with two pronounced humps on the hairy lateral lobes, the median lobe is hairy and similar to the abdomen of a bee. It is quite variable in the pattern of coloration, but usually brownish - red with yellow markings. The gynostegium is at right angles, with an elongated apex.
Output: {{ "propositions": [ "Ophrys apifera grows to a height of 15-50 centimetres (6-20 in)", "Ophrys apifera is a hardy orchid", "Ophrys apifera develops small rosettes of leaves in autumn", "The leaves of Ophrys apifera continue to grow slowly during winter", "The basal leaves of Ophrys apifera are ovate or oblong-lanceolate", "The upper leaves and bracts of Ophrys apifera are ovate-lanceolate and sheathing", "Ophrys apifera blooms from mid-April to July", "Ophrys apifera produces a spike composed of one to twelve flowers", "The flowers of Ophrys apifera have large sepals with a central green rib", "The flowers of Ophrys apifera vary in colour from white to pink", "The petals of Ophrys apifera are short, pubescent, and yellow to greenish", "The labellum of Ophrys apifera is trilobed with two pronounced humps on the hairy lateral lobes", "The median lobe of Ophrys apifera's labellum is hairy and resembles a bee's abdomen", "The coloration pattern of Ophrys apifera is variable but usually brownish-red with yellow markings", "The gynostegium of Ophrys apifera is at right angles with an elongated apex" ]}}

JUST OUTPUT THE PROPOSITIONS IN JSON FORMAT! DON"T OUTPUT ANYTHING INRELEVENT!
Input: {passage}
Output:
"""
    
import os    

def search(client, collection_name, query:list[list[float]], output_fields=None, top_k=None, filter=None):
    if filter:
        res = client.search(
            collection_name=collection_name,
            data=query,
            limit=top_k,
            output_fields=output_fields,
            filter=filter,
        )
    else:
        res = client.search(
            collection_name=collection_name,
            data=query,
            limit=top_k,
            output_fields=output_fields,
        )
    return res


def create_edge_collections(client, collection_name, dimension=1024):
    from pymilvus import MilvusClient, DataType

    schema = MilvusClient.create_schema(enable_dynamic_field=True)
    
    schema.add_field(
        field_name="id",
        datatype=DataType.VARCHAR,
        is_primary=True,
        max_length=100
    )
    
    # schema.add_field(
    #     field_name="relation",
    #     datatype=DataType.ARRAY,
    #     element_type=DataType.VARCHAR,
    #     max_capacity=3
    # )
    
    schema.add_field(
        field_name="entity",
        datatype=DataType.ARRAY,
        element_type=DataType.VARCHAR,
        max_capacity=2
    )
    
    schema.add_field(field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=dimension)
    
    index_params = MilvusClient.prepare_index_params()
    index_params.add_index(field_name="vector",index_type="AUTOINDEX")
    
    if client.has_collection(collection_name=collection_name):
        client.drop_collection(collection_name=collection_name)
    
    client.create_collection(
        collection_name=collection_name,
        dimension=dimension,
        schema=schema,
    )
    client.create_index(collection_name=collection_name, index_params=index_params)
    
def create_edge_collections_hyper(client, collection_name, dimension=1024):
    from pymilvus import MilvusClient, DataType
    schema = MilvusClient.create_schema(enable_dynamic_field=True)
    
    schema.add_field(
        field_name="id",
        datatype=DataType.INT64,
        is_primary=True,
    )
    
    
    schema.add_field(
        field_name="entity",
        datatype=DataType.ARRAY,
        element_type=DataType.VARCHAR,
        max_capacity=200,
        max_length=150,
    )
    
    schema.add_field(field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=dimension)
    
    index_params = MilvusClient.prepare_index_params()
    index_params.add_index(field_name="vector",index_type="AUTOINDEX")
    
    if client.has_collection(collection_name=collection_name):
        client.drop_collection(collection_name=collection_name)
    
    client.create_collection(
        collection_name=collection_name,
        dimension=dimension,
        schema=schema,
    )
    client.create_index(collection_name=collection_name, index_params=index_params)
    
def create_node_collections(client, collection_name, dimension=1024):
    if client.has_collection(collection_name=collection_name):
        client.drop_collection(collection_name=collection_name)
    
    client.create_collection(
        collection_name=collection_name,
        dimension=dimension,
    )
    
def batch_insert(client, collection_name, data, BATCH_SIZE):
    total = len(data)
    with tqdm(total=total, desc=f"Inserting into {collection_name}") as pbar:
        for i in range(0, total, BATCH_SIZE):
            batch = data[i:i+BATCH_SIZE]
            client.insert(collection_name=collection_name, data=batch)
            pbar.update(len(batch))

    
def get_embedding_model(embedding_model_name):
    from sentence_transformers import SentenceTransformer
    
    if embedding_model_name == "models--BAAI--bge-m3":
        #     model = SentenceTransformer("/home/yaodong/codes/GNNRAG/models--BAAI--bge-m3/snapshots/5617a9f61b028005a4858fdac845db406aefb181")
        from FlagEmbedding import BGEM3FlagModel
        model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True, devices="cuda")
        
    elif embedding_model_name == "nvidia/NV-Embed-v2":
        model = SentenceTransformer('nvidia/NV-Embed-v2', trust_remote_code=True, model_kwargs={"torch_dtype": "float16"}, device="cuda")
        model.max_seq_length = 4096
        model.tokenizer.padding_side="right"
        
    else:
        pass
    
    return model

def get_rerank_model(rerank_model_name):
    # from sentence_transformers import CrossEncoder
    # if rerank_model_name == 'BAAI/bge-reranker-v2-m3':
    #     model_rerank = CrossEncoder(rerank_model_name, max_length=512, device="cuda", )
    # else:
    #     pass
    if rerank_model_name == 'BAAI/bge-reranker-v2-m3':
        from FlagEmbedding import FlagReranker
        model_rerank = FlagReranker('BAAI/bge-reranker-v2-m3', use_fp16=True, devices="cuda")
    # score = reranker.compute_score(['query', 'passage'])
    # score = reranker.compute_score(['query', 'passage'], normalize=True)

    return model_rerank

task_name_to_instruct = {"example": "Given a question, retrieve passages that answer the question"}
query_prefix = "Instruct: "+task_name_to_instruct["example"]+"\nQuery: "

def include_query_prefix(question, use_prompt=True):
    if use_prompt:
        return query_prefix + question
    else:
        return question
    
def get_question_embedding(args, model, question):
    if args.embedding_model_name == "models--BAAI--bge-m3":
        # question_emb = model.encode(
        #     include_query_prefix(question, args.use_prompt), convert_to_tensor=True, device="cuda").cpu().numpy().reshape(1,-1)
        question_emb = model.encode(include_query_prefix(question, args.use_prompt), return_dense=True, return_sparse=False, return_colbert_vecs=False)
        question_emb = question_emb["dense_vecs"].reshape(1,-1).astype(np.float32)
            
    elif args.embedding_model_name == "nvidia/NV-Embed-v2":
        question_emb = model.encode(add_eos(model, [include_query_prefix(question, args.use_prompt)]), normalize_embeddings=True)
        question_emb = question_emb.astype(np.float32)
    else:
        pass
    return question_emb

def get_corpus_embedding(args, model, texts):
    
    if args.embedding_model_name == "models--BAAI--bge-m3":
        with torch.no_grad():
            # texts_embeddings = model.encode(texts, convert_to_tensor=True, show_progress_bar=True, device="cuda", batch_size=args.batch_size)
            # texts_embeddings = model.encode(texts, convert_to_tensor=True, device="cuda", batch_size=args.batch_size)
            # texts_embeddings = texts_embeddings.cpu().numpy()
            texts_embeddings = model.encode(texts, return_dense=True, return_sparse=False, return_colbert_vecs=False)
            texts_embeddings = texts_embeddings["dense_vecs"].astype(np.float32)
    
        return texts_embeddings
    
    elif args.embedding_model_name == "nvidia/NV-Embed-v2":
        add_eos_texts = add_eos(model, texts)
        batch_res = []
        
        with torch.no_grad():
            # 使用 tqdm 显示进度条
            for i in tqdm(
                range(0, len(add_eos_texts), args.batch_size),
                desc="Encoding texts",
                unit="batch",
                total=(len(add_eos_texts) + args.batch_size - 1) // args.batch_size,  # 计算总批次数
            ):
                # 编码当前批次
                texts_embeddings = model.encode(
                    add_eos_texts[i:i + args.batch_size],
                    batch_size=args.batch_size,
                    normalize_embeddings=True,
                )
                batch_res.append(texts_embeddings)
        
        texts_embeddings = np.concatenate(batch_res, axis=0)
        texts_embeddings = texts_embeddings.astype(np.float32)
        return texts_embeddings
    
    else:
        pass

def add_eos(model, input_examples):
    input_examples = [input_example + model.tokenizer.eos_token for input_example in input_examples]
    return input_examples

def find_top_k_index_hyper(client, collection_name, q_emb, path_set, edgeid2relation, filter, args):
    count = 0
    fold = 1
    return_keys = []
    last_result = None
    
    while count < args.top_k:
        now_result = search(client=client, collection_name=collection_name, query=q_emb, 
                            output_fields=["id","vector","entity"], top_k=args.top_k*fold, filter=filter)
        if last_result == now_result:
            break
        last_result = now_result
        now_result = now_result[0][args.top_k*(fold-1):]
        
        for item in now_result:
            edge_id = item["entity"]["id"]
            #breakpoint()
            edge_key = edgeid2relation[edge_id]
            if edge_key not in path_set:
                next_entity = item["entity"]["entity"]
                # breakpoint()
                return_keys.append((edge_key, item["distance"], next_entity, np.array(item["entity"]["vector"]).reshape(1,-1)))
                count += 1
                if count >= args.top_k:
                    break
        fold += 1
    
    return return_keys

def find_neighbor_hyper(client, collection_name, model, model_rerank, edgeid2relation, origin_q, entity, next_q, path, args):
    
    return_neighbor = []
    #q_emb = get_question_embedding(args, model, next_q)
    q_emb = next_q
    filter='ARRAY_CONTAINS(entity, "{}")'.format(entity)
    relations_key = find_top_k_index_hyper(client, collection_name, q_emb, set(path), edgeid2relation, filter, args)
    
    # if entity == "southeast library":
    #     breakpoint()
    
    class_of_tmp_path = []
    class_of_relations_des = []
    if relations_key:
        relations_des = []
        for x in relations_key:
            tmp_path = path.copy()
            tmp_path.append(x[0])
            
            class_of_tmp_path.append(tuple(sorted(tmp_path)))
            relations_des.append((origin_q, " ".join(sorted(tmp_path))))
            #class_of_next_q_embs.append(q_emb - x[-1])
        class_of_relations_des.extend(relations_des)
            #relations_des = list(map(lambda x: (origin_q, path + ":" + x[0]), relations_key))
        #scores = model_rerank.predict(relations_des)

    for i, (key, _, next_entities, key_emb) in enumerate(relations_key):
        # tmp_path = path.copy()
        # tmp_path.append(key)
        # tmp_entities = [next_entity for next_entity in next_entities if next_entity != entity]
        tmp_entities = tuple(sorted(next_entities))
        tmp_path = class_of_tmp_path[i]
        #return_neighbor.append((tuple(tmp_entities), next_q + ":" + key, key, path+"<SEP>"+key, scores[i]))
        #return_neighbor.append((tmp_entities, "".join(tmp_path)+origin_q, key, tmp_path, scores[i]))
        # return_neighbor.append((tmp_entities, q_emb - key_emb, key, tmp_path, scores[i]))
        return_neighbor.append((tmp_entities, q_emb - key_emb, key, tmp_path, class_of_relations_des[i]))

    return return_neighbor


def expand_one_layer_hyper(client, collection_name, queue, model, model_rerank, edgeid2relation, origin_q, args):
    next_queue = []
    sorted_candidate = []
    while queue:
        entity, next_q, path  = queue.popleft()
        path = list(path)
        return_neighbor = find_neighbor_hyper(client, collection_name, model, model_rerank, edgeid2relation, origin_q, entity, next_q, path, args)
        next_queue.extend(return_neighbor)
    
    if next_queue:
        next_queue, sorted_candidate = choose_top_k_next_candidate(next_queue, edgeid2relation, model_rerank, args)
    
    return next_queue, sorted_candidate

def sort_by_score(scores, elements):
    paired = list(zip(scores, elements))
    paired_sorted = sorted(paired, key=lambda x: x[0], reverse=True)
    sorted_scores = [item[0] for item in paired_sorted]
    sorted_elements = [item[1] for item in paired_sorted]
    return sorted_scores, sorted_elements
    
def choose_top_k_next_candidate(candidate_list:list, already_find, model_rerank, args):
    candidate = {item[-2]: item for item in candidate_list}
    candidate_list = list(map(lambda x: candidate[x], candidate))
    #candidate_list_scores = model_rerank.predict()
    candidate_list_scores = model_rerank.compute_score(list(map(lambda x: x[-1], candidate_list)), normalize=True)
    # breakpoint()
    candidate_list_scores, candidate_list = sort_by_score(candidate_list_scores, candidate_list)
    
    sorted_candidate = candidate_list[:args.top_k_path]
    
    next_queue = []
    return_items = []
    for idx, item in enumerate(sorted_candidate):
        # next_entity, q_str + ":" + key, key, path+"<SEP>"+key, scores[i]
        next_entities, next_question, relation_key, path, _ = item
        return_items.append((next_entities, next_question, relation_key, path, candidate_list_scores[idx]))
        #already_find[relation_key] = next_question
        for next_entity in next_entities:
            next_queue.append((next_entity, next_question, path))
    
    return deque(next_queue), return_items

import gzip
import pickle
def load_gzipped_pickle(filename):
    """从gzip压缩的pickle文件中加载Python对象"""
    with gzip.open(filename, 'rb') as f:
        return pickle.load(f)
    
def load_question_ner(args):
    qid2ner = {}
    with open(f'./data/{args.dataset_name}/{args.folder_name}/Question_llm.json') as f:
        for line in tqdm(f):
            line = json.loads(line.strip())
            md5_id = line["question"]
            
            raw_string = line["generation"].strip()
            json_string = raw_string.strip('```json\n').strip('```').strip()
                
            try:
                ner_entity = json.loads(json_string)
                if ner_entity:
                    ner_entity = ner_entity["named_entities"]
                else:
                    ner_entity = []
                qid2ner[md5_id] = ner_entity
            except:
                breakpoint()
    return qid2ner

def load_question(args):
    #qid2ner = load_question_ner(args)
    questions_list = []
    with open(f'./data/{args.dataset_name}/{args.folder_name}/Question.json') as f:
        for line in tqdm(f):
            line = json.loads(line.strip())
            
            # md5_id = f"doc-{args.folder_name}-" + mdhash_id(line["question"])
            md5_id = mdhash_id(line["question"])
            question = line["question"]
            
            # questions_list.append((md5_id, question, qid2ner[mdhash_id(line["question"])]))
            questions_list.append((md5_id, question, 0))
    return questions_list

def load_context(args):
    cid_2_context = {}
    corpus_context = read_json(f'./data/{args.dataset_name}/{args.folder_name}/Corpus.json')
    for item in corpus_context:
        cid_2_context[item["question_id"]] = item["context"]
        
    return cid_2_context

def load_vdb(args):
    from pymilvus import MilvusClient
    client = MilvusClient(f'./data/{args.dataset_name}/{args.folder_name}/{clean_str(args.embedding_model_name).replace(" ", "")}_{args.vdb_name}')
    return client

def rerank_chunks(chunks, cid_2_context, model_rerank, question):
    context = list(map(lambda x: (question, cid_2_context[x]), chunks))
    # scores = model_rerank.predict(context)
    scores = model_rerank.compute_score(context, normalize=True)
    combined = sorted(zip(chunks, scores), key=lambda x: -x[1])
    sorted_chunks, scores = zip(*combined)

    return sorted_chunks, scores
        
def vallina_rag(client, question_emb, args):
    relevent_chunks = search(client, "chunk_collection", question_emb, ["chunk_id"], args.top_k_chunk)
    res = []
    for item in relevent_chunks[0]:
        res.append(item["entity"]["chunk_id"])
    
    return res

def prune_token(question, retrieved_context, prompt, max_token_len):
    question_token_len = len(encode_string_by_tiktoken(question))
    left_token_len = max_token_len - question_token_len
    user_prompt_token_len = len(encode_string_by_tiktoken(retrieved_context))
    
    if user_prompt_token_len >= left_token_len:
        left_prompt = decode_string_by_tiktoken(encode_string_by_tiktoken(retrieved_context)[:left_token_len])
    else:
        left_prompt = retrieved_context
    
    res = prompt.format(context_data=left_prompt, question=question)
    
    return res

def wrapper_generation(question: str, relevent_chunks: list, cid_2_context: dict, args):
    retrieved_context = list(map(lambda x: cid_2_context[x], relevent_chunks))[:args.top_k_chunk]
    retrieved_context = "\n\n".join(retrieved_context)
    return prune_token(question, retrieved_context, RESPONSE_PROMPT, args.max_generation_token)

def get_file_name(args):
    name = ""
    interest_name_fiedl = set(["embedding_model_name", "rerank_model_name", "top_k", "recall_k",
                               "top_k_path", "use_vallina_rag", "use_rerank_final", "llm_model"])
    
    for key, value in vars(args).items():
        if key in interest_name_fiedl and value:
            cleaned_key = key.replace(" ", "").replace("_", "")
            cleaned_value = str(value).replace("-","").replace("/","")
            name += f"{cleaned_key}-{cleaned_value}_"  
    return name

def split_tokens_with_overlap(
    text: str,
    max_length: int = 500,
    overlap: int = 100,
    model_name: str = "cl100k_base"
) -> list[list[int]]:
    """
    Split a long string into token chunks with overlap.
    
    Args:
        text: The input string to split.
        max_length: Maximum token length per chunk.
        overlap: Number of overlapping tokens between chunks.
        model_name: tiktoken encoding model (default: "cl100k_base").
    
    Returns:
        List of token chunks (each chunk is a list of token IDs).
    """
    if overlap >= max_length:
        raise ValueError("Overlap must be smaller than max_length.")
    
    tokens = encode_string_by_tiktoken(text, model_name)
    chunks = []
    start = 0
    
    while start < len(tokens):
        end = start + max_length
        chunk = tokens[start:end]
        stri = decode_string_by_tiktoken(chunk)
        chunks.append({"question_id": mdhash_id(stri), "context": stri})
        start += (max_length - overlap)  # Move forward with overlap
    
    return chunks

DATASET_NEED_TOKENIZER = set(["narrative", "multihoprag"])

def single_llm(args, data, save_path):
    import multiprocessing
    completed_questions = set()
    if os.path.exists(save_path):
        completed_questions = {item["question_id"] for item in read_json(save_path)}

    tasks = [
        item
        for item in data
        if item["question_id"] not in completed_questions
    ]

    total_tasks = len(tasks)
    
    with tqdm(total=total_tasks, desc="Processing") as pbar:
        with multiprocessing.Pool(processes=args.llm_parallel_nums) as pool:
            for result in pool.imap_unordered(entity_extraction_single, tasks):
                answer = result
                if answer is not None:
                    with open(save_path, "a") as f:
                        json.dump(answer, f)
                        f.write("\n")
                    pbar.update(1)

    print("All tasks completed!")

def get_propositions_by_llm(args):
    questions = []
    candidates_text = []
    from utils import PROPOSITION_PROMPT
    from utils import single_llm
    
    if args.dataset_name in DATASET_NEED_TOKENIZER:
        if not os.path.exists(f'./data/{args.dataset_name}/{args.folder_name}/Corpus_raw.json'):
            with open(f'./data/{args.dataset_name}/{args.folder_name}/Corpus.json', "r") as f:
                for line in f:
                    line = json.loads(line.strip())
                    long_text = line["context"]
            split_chunks = split_tokens_with_overlap(long_text)
            # breakpoint()
            os.rename(f'./data/{args.dataset_name}/{args.folder_name}/Corpus.json', 
                    f'./data/{args.dataset_name}/{args.folder_name}/Corpus_raw.json')
            
            wrtie_json(split_chunks, f'./data/{args.dataset_name}/{args.folder_name}/Corpus.json')
    
    with open(f'./data/{args.dataset_name}/{args.folder_name}/Corpus.json', "r") as f:
        for line in f:
            line = json.loads(line.strip())
            questions.append({
                "question_id": line["question_id"],
                "prompt": PROPOSITION_PROMPT.format(passage=line["context"]),
            })
            candidates_text.append(line["context"])
    f.close()
    save_path = f'./data/{args.dataset_name}/{args.folder_name}/Corpus_proposition.json'
    
    if not os.path.exists(f'./data/{args.dataset_name}/{args.folder_name}/chunks_choices.json'):
        chunks_score, chunks_cost = select_chunk(questions, candidates_text, args)
        
        total_cost = sum(chunks_cost)
        chunks_len = len(chunks_cost)
        constraint_weight = int(total_cost * 0.5)
        max_value, selected_items = backpack_memmap_with_selection(number=chunks_len, weight=constraint_weight, w=chunks_cost, v=chunks_score)
        
        with open(f'./data/{args.dataset_name}/{args.folder_name}/chunks_choices.json', "w") as f:
            # pickle.dump((chunks_score, chunks_cost, max_value, selected_items), f)
            f.write(json.dumps({
                "chunks_score": chunks_score,
                "chunks_cost": chunks_cost,
                "max_value": float(max_value),
                "selected_items": selected_items
            }) + "\n")
        f.close()
    else:
        with open(f'./data/{args.dataset_name}/{args.folder_name}/chunks_choices.json', "r") as f:      
            for line in f:
                dic = json.loads(line)
        selected_items = dic["selected_items"]
        max_value = dic["max_value"]
        selected_items = dic["selected_items"]
        
    print("max value", max_value)
    print("selected_items len", len(selected_items))
    
    chunks_id_0 = set(list(map(lambda x: questions[x]["question_id"], selected_items)))
    write_data = []
    
    import nltk
    from nltk.tokenize import sent_tokenize
    
    for idx, item in enumerate(questions):
        if item["question_id"] not in chunks_id_0:
            
            sentences = sent_tokenize(candidates_text[idx])
            sentences = [sen for sen in sentences if sen != ""]
            write_data.append({
                "question_id": item["question_id"],
                "generation": sentences
            })
    
    wrtie_json(write_data, save_path, "w")
    
    chunks_need_llm = [questions[i] for i in selected_items]
    single_llm(args, chunks_need_llm, save_path)
    
    import nltk
    from nltk.tokenize import sent_tokenize
        
    completed_questions = set()
    if os.path.exists(save_path):
        completed_questions = {item["question_id"] for item in read_json(save_path)}

    write_data = []
    for idx, item in enumerate(questions):
        if item["question_id"] not in completed_questions:
            sentences = sent_tokenize(candidates_text[idx])
            sentences = [sen for sen in sentences if sen != ""]
            write_data.append({
                "question_id": item["question_id"],
                "generation": sentences
            })
    wrtie_json(write_data, save_path, "a")
    
    print("Finish get_propositions_by_llm...")

def extract_entity_from_propositions(args):
    import os
    all_chunks = read_json(f'./data/{args.dataset_name}/{args.folder_name}/Corpus.json')
    #chunkid2context = {item["question_id"]:item["context"] for item in all_chunks}
    
    if not os.path.exists(f'./data/{args.dataset_name}/{args.folder_name}/chunkid2propositions.json'):
        import spacy
        corpus = read_json(f'./data/{args.dataset_name}/{args.folder_name}/Corpus_proposition.json')
        spacy.prefer_gpu()
        nlp = spacy.load("en_core_web_trf")
        interest_entity_set = set(["DATE", "EVENT", "FAC", "GPE", "LANGUAGE", "LAW", "LOC", "NORP", "ORG","PERSON","PRODUCT","WORK_OF_ART"])
        
        chunkid2propositions = defaultdict(lambda: defaultdict(dict))
        for cor in tqdm(corpus, total=len(corpus)):
            chunkid = cor["question_id"]
            propositions = cor["generation"]
                    
            for pro in propositions:
                doc = nlp(pro)
                reconizer_enttiy2type = {}
                for span in doc.ents:
                    if span.label_ in interest_entity_set:
                        reconizer_enttiy2type[span.text] = span.label_
                
                chunkid2propositions[chunkid][pro] = reconizer_enttiy2type
                
        chunkid2propositions = dict(chunkid2propositions)
        
        with open(f'./data/{args.dataset_name}/{args.folder_name}/chunkid2propositions.json', "w") as f:
            f.write(json.dumps(chunkid2propositions) + "\n")
        f.close()
        
    print("Finish extract_entity_from_propositions...")


def index(args):
    # from utils import (
    #     read_json, clean_str, load_context,get_embedding_model, get_corpus_embedding,batch_insert
    # )
    db_name = f'./data/{args.dataset_name}/{args.folder_name}/{clean_str(args.embedding_model_name).replace(" ", "")}_{args.vdb_name}'
    if not os.path.exists(db_name):
        import gzip
        import pickle
        
        rkey2cid = {}
        cid_2_context = load_context(args)
        model = get_embedding_model(args.embedding_model_name)
        
        # embedding edges
        chunkid2propositions = read_json(f'./data/{args.dataset_name}/{args.folder_name}/chunkid2propositions.json')
        chunkid2propositions = chunkid2propositions[0]
        edge_data = []
        edge_text = []
        edge_count = 0
        
        entity_text = []
        edgeid2relation = {}
        for chunkid, propositions_dict in chunkid2propositions.items():
            for key, value in propositions_dict.items():
                entity_list = [clean_str(key) for key, item in value.items() if item != "DATE"]
                entity = list(map(lambda x: clean_str(x), value))
                edge_dict = {
                    "id": edge_count, "entity": entity
                }
                edgeid2relation[edge_count] = key
                entity_text.extend(entity_list)
                edge_count += 1
                edge_data.append(edge_dict)
                edge_text.append(key)
                rkey2cid[key] = chunkid

        # embedding entity
        entity_text = list(set(entity_text))  
        entity_embeddings = get_corpus_embedding(args, model, entity_text)
        entity_data = [
            {"id": i, "vector": entity_embeddings[i].tolist(), "entity_name":entity_text[i]} 
            for i in range(len(entity_text))
        ]
        
        edge_embeddings= get_corpus_embedding(args, model, edge_text)
        for i in range(edge_count):
            edge_data[i]["vector"] = edge_embeddings[i].tolist()
        
        # embedding chunk
        chunks = list(map(lambda x: cid_2_context[x], cid_2_context))
        chunk_embedding = get_corpus_embedding(args, model, chunks)
        chunks_id = list(map(lambda x: x, cid_2_context))
        chunk_data = [
            {"id": i, "vector": chunk_embedding[i].tolist(), "chunk_id":chunks_id[i]}
            for i in range(len(chunks))
        ]
        
        # create vdb
        from pymilvus import MilvusClient
        from utils import create_node_collections, create_edge_collections_hyper
        
        client = MilvusClient(db_name)
        
        create_edge_collections_hyper(client=client, collection_name="edge_collection", dimension=edge_embeddings.shape[1])
        create_node_collections(client=client, collection_name="chunk_collection", dimension=edge_embeddings.shape[1])
        create_node_collections(client=client, collection_name="entity_collection", dimension=edge_embeddings.shape[1])
        
        batch_insert(client=client, collection_name="edge_collection", data=edge_data, BATCH_SIZE=1024)
        batch_insert(client=client, collection_name="chunk_collection", data=chunk_data, BATCH_SIZE=1024)  
        batch_insert(client=client, collection_name="entity_collection", data=entity_data, BATCH_SIZE=1024)   
        
        with gzip.open(f'./data/{args.dataset_name}/{args.folder_name}/rkey2cid.pkl.gz', "wb") as f:
            pickle.dump((rkey2cid, edgeid2relation), f)
            
    print("Finish Index...")
    
def vallina_retrieve(args):
    model = get_embedding_model(args.embedding_model_name)
    # cid_2_context = load_context(args)
    questions_list = load_question(args)
    client = load_vdb(args)
    jsonl_result = []
    
    for md5_id, question, ner in tqdm(questions_list):
        question_emb = get_question_embedding(args, model, question)
        if args.use_vallina_rag:
            from utils import vallina_rag
            relevent_chunks = vallina_rag(client, question_emb, args)
        
        jsonl_result.append({ "question_id": md5_id, "chunks_id": relevent_chunks})
    
    os.makedirs(f'./data/{args.dataset_name}/{args.folder_name}/{args.mode}', exist_ok=True)
    write_path = f'./data/{args.dataset_name}/{args.folder_name}/{args.mode}/{get_file_name(args)}retrieve.json'
    wrtie_json(jsonl_result, write_path)

def load_extract_entity(args):
    qid_2_entity = {}
    entity = read_json(f'./data/{args.dataset_name}/{args.folder_name}/Question_result.json')
    for item in entity:
        try:
            qid_2_entity[item["question_id"]] = item["generation"]["named_entities"]
        except:
            breakpoint()
            qid_2_entity[item["question_id"]] = []
    return qid_2_entity

def hyper_graph(args):
    from collections import defaultdict,deque
    
    model = get_embedding_model(args.embedding_model_name)
    model_rerank = get_rerank_model(args.rerank_model_name)
    rkey2cid, edgeid2relation = load_gzipped_pickle(f'./data/{args.dataset_name}/{args.folder_name}/rkey2cid.pkl.gz')
    cid_2_context = load_context(args)
    qid_2_entity = load_extract_entity(args)
    questions_list = load_question(args)
    client = load_vdb(args)
    jsonl_result = []
    
    for md5_id, question, ner in tqdm(questions_list):
        # if md5_id != "bf55136b8364d440dd263956256d1ece":
        #     continue
        # breakpoint()
        question_emb = get_question_embedding(args, model, question)
        relevent_edges = search(client, "edge_collection", question_emb, ["entity"], args.top_k)
        relevent_edges = relevent_edges[0]
        
        link_entities = []
        ner_entities = qid_2_entity[md5_id]
        if ner_entities:
            ner_entity_embeddings = get_corpus_embedding(args, model, ner_entities)
            similar_entities = search(client, "entity_collection", ner_entity_embeddings.tolist(), top_k=3, output_fields=["entity_name"])
            
            for idx, _ in enumerate(ner_entities):
                link_entities.extend(list(map(lambda x: x["entity"]["entity_name"], similar_entities[idx])))
        
        for i in range(len(relevent_edges)):
            link_entities.extend(relevent_edges[i]["entity"]["entity"])
        link_entities = list(set(link_entities))
        
        queue = deque(
            list(zip(link_entities, [question_emb for _ in range(len(link_entities))], [[] for _ in range(len(link_entities))]))
        )
        depth = 0
        all_items = []
        # breakpoint()
        while depth < 3:
            queue, items = expand_one_layer_hyper(client, "edge_collection", queue, model, model_rerank, edgeid2relation, question, args)
            all_items.extend(items)
            depth += 1
            
        chunk_scores = defaultdict(dict)
        
       # all_items = sorted(all_items, key=lambda x: -x[-1])[:args.top_k_path]
        
        for item in all_items:
            #path_list = item[3].split("<SEP>")[1:]
            path_list = item[3]
                
            d_value = item[4]
            for path in path_list:
                cid = rkey2cid[path]
            
                if "score" not in chunk_scores[cid]:
                    chunk_scores[cid]["score"] = []
                if "count" not in chunk_scores[cid]:
                    chunk_scores[cid]["count"] = 0.0
                    
                chunk_scores[cid]["score"].append(d_value)
                chunk_scores[cid]["count"] += 1
                
        for cid in chunk_scores:
            chunk_scores[cid]["avg_score"] = sum(chunk_scores[cid]["score"]) / chunk_scores[cid]["count"]
        
        cid_avg_score = list(map(lambda x: (x, chunk_scores[x]["avg_score"]), chunk_scores))
        
        list_chunks = sorted(cid_avg_score, key=lambda x: -x[1])
        
        relevent_chunks = []
        for j in list_chunks:
            relevent_chunks.append(j[0])
        sorted_chunks = relevent_chunks
        #breakpoint()
        ################################################################ 
        if args.use_vallina_rag:
            from utils import vallina_rag
            tmp = vallina_rag(client, question_emb, args)
            relevent_chunks.extend(tmp)
            
            relevent_chunks = list(set(relevent_chunks))
        
        if args.use_rerank_final:
            from utils import rerank_chunks
            if not relevent_chunks:
                print("Out")
                continue
            sorted_chunks, _ = rerank_chunks(relevent_chunks, cid_2_context, model_rerank, question)
            sorted_chunks = sorted_chunks[:args.top_k_chunk]
        else:
            sorted_chunks = relevent_chunks[:args.top_k_chunk]
        ################################################################ 
        
        final_prompt = wrapper_generation(question, sorted_chunks, cid_2_context, args)
        jsonl_result.append({ "question_id": md5_id, "prompt": final_prompt, "chunks_id": sorted_chunks})
    
    
    # import multiprocessing
    # model = get_embedding_model(args.embedding_model_name)
    # model_rerank = get_rerank_model(args.rerank_model_name)
    # rkey2cid = load_gzipped_pickle(f'./data/{args.dataset_name}/{args.folder_name}/rkey2cid.pkl.gz')
    # cid_2_context = load_context(args)
    # questions_list = load_question(args)
    # #client = load_vdb(args)
    # jsonl_result = []
    
    # # md5_id, question, model, model_rerank, rkey2cid, cid_2_context, args
    # tasks = [(md5_id, question, model, model_rerank, rkey2cid, cid_2_context, args) for md5_id, question, _ in questions_list]

    # total_tasks = len(tasks)
    
    # with tqdm(total=total_tasks, desc="Processing") as pbar:
    #     with multiprocessing.Pool(processes=32) as pool:
    #         for result in pool.imap_unordered(single_retrieve, tasks):
    #             answer = result
    #             jsonl_result.append(answer)
    #             pbar.update(1)
    
    os.makedirs(f'./data/{args.dataset_name}/{args.folder_name}/{args.mode}', exist_ok=True)
    write_path = f'./data/{args.dataset_name}/{args.folder_name}/{args.mode}/{get_file_name(args)}retrieve.json'
    wrtie_json(jsonl_result, write_path)
    print("Finish Retrieve...")

DATASET_NEED_RETRIEVE_RECALL = set(["nq", "musique-1000", "popqa", "2wiki", "hotpotqa", "test"])

def generation(args):
    
    if args.dataset_name in DATASET_NEED_RETRIEVE_RECALL:
        calculate_recall(args)
    
    if 0:
        from single_llm import single_llm
        retrieve_result = read_json(f'./data/{args.dataset_name}/{args.folder_name}/{args.mode}/{get_file_name(args)}retrieve.json')
        
        save_path=f'./data/{args.dataset_name}/{args.folder_name}/{args.mode}/{get_file_name(args)}generation.json'
        # generate answer using llm
        single_llm(args, retrieve_result, save_path)

        # get the result and evaluate them
        data_generations = read_json(save_path)
        ground_truth_answer = get_ground_truth_answer(args)
        save_path=f'./data/{args.dataset_name}/{args.folder_name}/{args.mode}/{get_file_name(args)}generation_result.json'
        eval(data_generations, ground_truth_answer, save_path)
        
    print("Finish Generation...")
        
def get_ground_truth_answer(args):
    dic = {}
    questions = read_json(f'./data/{args.dataset_name}/{args.folder_name}/Question.json')
    for item in questions:
        if isinstance(item["answer"], str):
            value = [item["answer"]]
        else:
            value = item["answer"]
        dic[f"doc-{args.folder_name}-"+mdhash_id(item["question"])] = value
    return dic

def get_grount_truth_corpus(args):
    dic = {}
    data = read_json(f'./data/{args.dataset_name}/{args.folder_name}/Corpus_gt.json')
    for item in data:
        # dic[f"doc-{args.folder_name}-"+item["question_id"]] = item["ground_truth"]
        dic[item["question_id"]] = item["ground_truth"]
    return dic

def calculate_recall(args):
    res = 0
    output = []
    ground_truth = get_grount_truth_corpus(args)
    retrieve_result = read_json(f'./data/{args.dataset_name}/{args.folder_name}/{args.mode}/{get_file_name(args)}retrieve.json')
    for item in retrieve_result:
        v, pre_right, pre_false = recall_at_k(ground_truth[item["question_id"]], item["chunks_id"], k=args.recall_k)
        res += v
        
        output.append({
            "question_id":item["question_id"],
            f"recall@{args.recall_k}": v,
            "true":pre_right,
            "false": pre_false,
        })
        
    print("Total number of questions: ", len(retrieve_result))
    print(f"Recall @ {args.recall_k}: {res/len(retrieve_result)}")
    
    wrtie_json(output, f'./data/{args.dataset_name}/{args.folder_name}/{args.mode}/{get_file_name(args)}retrieve_result.json')
    
from eval_hippo2 import calculate_metric_scores
def eval(data_generation, data_answer, data_path):
    golden_list = []
    pre_list = []
    index = []
    
    for item in data_generation:
        golden_list.append(data_answer[item["question_id"]])
        pre_list.append(item["generation"])
        index.append(item["question_id"]) 
    pooled_eval_results = calculate_metric_scores(golden_list,pre_list)
    print(f"{len(data_generation)} tasks: ", pooled_eval_results)
    wrtie_json([pooled_eval_results],data_path)

from multiprocessing import Pool

def calc_bleu_single(id, references, hypothesis):
    from sacrebleu import sentence_bleu
    #return (id, nltk.translate.bleu_score.sentence_bleu(reference, hypothesis, weight,smoothing_function=SmoothingFunction().method1))
    score = sentence_bleu(hypothesis, references, lowercase=True, smooth_method="floor")
    
    return id, score.score * 0.01


def calculate_selfbleu(candidates_text: list[str], parallel_nums=2, gram=3):

    result = []
    sentence_num = len(candidates_text)
    pool = Pool(parallel_nums)
    
    # 使用 tqdm 包装循环
    for cid in tqdm(range(sentence_num), desc="Calculating Self-BLEU"):
        hypothesis = candidates_text[cid]
        other = candidates_text[:cid] + candidates_text[cid+1:]
        # result.append(pool.apply_async(calc_bleu_single, args=(cid, other, hypothesis, weight)))
        result.append(pool.apply_async(calc_bleu_single, args=(cid, other, hypothesis)))
    
    sorted_result = [0.0 for _ in range(sentence_num)]
    for res in tqdm(result, desc="Collecting results"):
        cid, value = res.get()
        sorted_result[cid] = value
        
    pool.close()
    pool.join()
    
    return sorted_result

def select_chunk(questions, candidates_text, args):
    scores = calculate_selfbleu(candidates_text=candidates_text, parallel_nums=64, gram=args.gram)
    #top_k_indices = sorted(range(len(scores)), key=lambda i: -scores[i])[:args.select_topk_chunks]
    #return list(map(lambda k: questions[k], top_k_indices))
    costs = list(map(lambda x: len(encode_string_by_tiktoken(x)), candidates_text))
    
    return scores, costs

def propositions2graph(chunkid2propositions):
    import networkx as nx
    from itertools import combinations
    G = nx.Graph()

    nodes = []
    edges = []
    for cid, propositions in chunkid2propositions.items():
        for key, value in propositions.items():
            entities = list(map(lambda x: clean_str(x), value))
            nodes.extend(entities)
            pairs = list(map(lambda x: tuple(sorted(x)), combinations(entities, 2)))
            edges.extend(pairs)
    
    nodes = list(set(nodes))
    edges = list(set(edges))
    
    G.add_edges_from(edges)
    G.add_nodes_from(nodes)
    
    print(G.number_of_edges())
    print(G.number_of_nodes())
    connected_components = list(nx.connected_components(G))
    num_components = len(connected_components)
    print(f"连通分量数量: {num_components}")  
    
    largest_component = max(connected_components, key=len)
    # print(f"最大连通分量的节点: {largest_component}")  # 输出: {1, 2, 3, 4}
    largest_subgraph = G.subgraph(largest_component)
    print(largest_subgraph.number_of_edges())
    print(largest_subgraph.number_of_nodes())
    
    breakpoint()

def entity_alignment(args, entity_list: list[str], round):
    import networkx as nx
    
    model = get_embedding_model(args.embedding_model_name)
    texts_embeddings = get_corpus_embedding(args, model, entity_list)
    texts_embeddings = torch.from_numpy(texts_embeddings)
    
    threshold = dynamic_threshold(round)
    res = compute_similarity_and_multiply(texts_embeddings, threshold)
    res = res.coalesce()
    res = res.indices().tolist()
    
    item_len = len(res[0])
    G = nx.Graph()
    edges = [sorted([entity_list[res[0][idx]], entity_list[res[1][idx]]]) for idx in range(item_len)]
    G.add_edges_from(edges)
    
    connected_components = list(nx.connected_components(G))
    num_components = len(connected_components)
    #print(f"连通分量数量: {num_components}")  
    
    res = []
    for item in connected_components: 
        item = str(sorted(list(item)))
        qid = mdhash_id(item)
        try:
            prompt = ENTITY_ALIGNMENT_PROMPT.format(input_entity_list=item)
        except:
            breakpoint()
        res.append({"question_id": qid, "prompt":prompt})
    return res
    
# import ollama
# import time
from typing import Dict, Optional, Callable

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from typing import Dict, List
from pydantic import BaseModel, ValidationError
import time
import json

from typing import Dict, List
try:
    from pydantic import RootModel
    class EntityAlignmentResult(RootModel):
        root: Dict[str, List[str]]  # 键是规范化的实体名称，值是别名列表

    class EntityExtractionResult(RootModel):
        root: Dict[str, List[str]]
except:
    class EntityAlignmentResult:
        pass
    
    class EntityExtractionResult:
        pass

def entity_alignment_single(task: dict[str], max_retries: int = 3) -> EntityAlignmentResult:
    from ollama import chat
    retries = 0
    question_id = task["question_id"]
    prompt = task["prompt"]
    while retries < max_retries:
        try:
            # 调用模型
            response = chat(
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                model="llama3.1:8b4k",
                format=EntityAlignmentResult.model_json_schema(),
            )

            # 验证结果
            result = EntityAlignmentResult.model_validate_json(response.message.content)
            return {"question_id": question_id, "generation": result.root}

        except (ValidationError, json.JSONDecodeError) as e:
            print(f"Attempt {retries + 1} failed: {str(e)}")
            retries += 1
            # if retries < max_retries:
            #     pass
        except Exception as e:
            print(f"Unexpected error: {str(e)}")
            retries += 1

    print(f"Failed after {max_retries} retries.")
    return None

def entity_extraction_single(task: dict[str], max_retries: int = 10) -> EntityExtractionResult:
    import ollama
    client = ollama.Client(host="http://localhost:5001/forward")
    
    retries = 0
    question_id = task["question_id"]
    prompt = task["prompt"]
    while retries < max_retries:
        try:
            response = client.chat(
                model="llama3.1:8b4k",
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": 0},
                # format=EntityExtractionResult.model_json_schema(),
            )
            # 验证结果
            result = EntityExtractionResult.model_validate_json(response.message.content)
            if not result.root:
                print("Null, retry!")
                continue
            # if "named_entities" not in result.root:
            #     continue
            
            return {"question_id": question_id, "generation": result.root["propositions"]}

        except (ValidationError, json.JSONDecodeError) as e:
            print(f"Attempt {retries + 1} failed: {str(e)}")
            retries += 1
            continue
            # if retries < max_retries:
            #     pass
        except Exception as e:
            print(f"Unexpected error: {str(e)}")
            retries += 1

    print(f"Failed after {max_retries} retries.")
    return None

def propositions2entities(args, chunkid2propositions, round):
    entity_list = []
    for chunkid, propositions_dict in chunkid2propositions.items():
        for key, value in propositions_dict.items():
            tmp = []
            for name, typ in value.items():
                if typ != "DATE":
                    tmp.append(name)
                    #cleanname2origin[clean_name] = name
            entity_list.extend(tmp)
    
    entity_list = list(set(entity_list))
    
    data = entity_alignment(args, entity_list, round)
    
    print(f"Round {round}: {len(data)} pairs")
    return data

def save_propositions2entities(args, chunkid2propositions, round):
    save_path = f'./data/{args.dataset_name}/{args.folder_name}/alignment/round_{round}.json'
    alignment_result = read_json(save_path)
    
    oldentity2newentity = {}
    for res in alignment_result:
        gen = res["generation"]
        for key, value in gen.items():
            for val in value:
                oldentity2newentity[val] = key

    new_chunkid2propositions = defaultdict(lambda: defaultdict(dict))

    for chunkid, propositions_dict in chunkid2propositions.items():
        for key, value in propositions_dict.items():
            for name, typ in value.items():
                if name in oldentity2newentity:
                    new_chunkid2propositions[chunkid][key][oldentity2newentity[name]] = typ
                else:
                    new_chunkid2propositions[chunkid][key][name] = typ
    
    wrtie_json([new_chunkid2propositions],f'./data/{args.dataset_name}/{args.folder_name}/alignment/chunkid2propositions_{round}.json')
    
    return new_chunkid2propositions

def test(args):
    import multiprocessing
    chunkid2propositions = read_json(f'./data/{args.dataset_name}/{args.folder_name}/chunkid2propositions.json')
    chunkid2propositions = chunkid2propositions[0]
    
    os.makedirs(f'./data/{args.dataset_name}/{args.folder_name}/alignment/', exist_ok=True)
    round = 0
    while True:
        data = propositions2entities(args, chunkid2propositions, round)
        if len(data) == 0:
            break
        
        save_path = f'./data/{args.dataset_name}/{args.folder_name}/alignment/round_{round}.json'
        completed_questions = set()
        if os.path.exists(save_path):
            completed_questions = {item["question_id"] for item in read_json(save_path)}

        print(f"Already complete {len(completed_questions)}")
        tasks = [
            item
            for item in data
            if item["question_id"] not in completed_questions
        ]

        total_tasks = len(tasks)
        
        with tqdm(total=total_tasks, desc="Processing") as pbar:
            with multiprocessing.Pool(processes=args.llm_parallel_nums) as pool:
                for result in pool.imap_unordered(entity_alignment_single, tasks):
                    answer = result
                    if answer is not None:
                        with open(save_path, "a") as f:
                            json.dump(answer, f)
                            f.write("\n")
                        pbar.update(1)
                        
        chunkid2propositions = save_propositions2entities(args, chunkid2propositions, round)
        round += 1
    print("END")
    
def extract_entity_from_question(args):
    import multiprocessing
    raw_questions = read_json(f'./data/{args.dataset_name}/{args.folder_name}/Question.json')
    data = [{"question_id": mdhash_id(item["question"]), "prompt": NER_EXTRACTION.format(passage=item["question"])} for item in raw_questions]
    
    save_path = f'./data/{args.dataset_name}/{args.folder_name}/Question_result.json'
    
    completed_questions = set()
    if os.path.exists(save_path):
        completed_questions = {item["question_id"] for item in read_json(save_path)}

    print(f"Already complete {len(completed_questions)}")
    tasks = [
        item
        for item in data
        if item["question_id"] not in completed_questions
    ]

    total_tasks = len(tasks)
    
    with tqdm(total=total_tasks, desc="Processing extracting entities...") as pbar:
        with multiprocessing.Pool(processes=args.llm_parallel_nums) as pool:
            for result in pool.imap_unordered(entity_extraction_single, tasks):
                answer = result
                if answer is not None:
                    with open(save_path, "a") as f:
                        json.dump(answer, f)
                        f.write("\n")
                    pbar.update(1)
                    

def extract_propositions_entity_from_chunks(args):
    import multiprocessing
    raw_chunks= read_json(f'./data/{args.dataset_name}/{args.folder_name}/Corpus.json')
    data = [{"question_id": item["question_id"], "prompt": PROPOSITION_NER_PROMPT.format(passage=item["context"])} for item in raw_chunks]
    
    save_path = f'./data/{args.dataset_name}/{args.folder_name}/chunkid2propositions.json'
    
    completed_questions = set()
    if os.path.exists(save_path):
        completed_questions = {item["question_id"] for item in read_json(save_path)}

    print(f"Already complete {len(completed_questions)}")
    tasks = [
        item
        for item in data
        if item["question_id"] not in completed_questions
    ]

    total_tasks = len(tasks)
    
    with tqdm(total=total_tasks, desc="Processing extracting entities...") as pbar:
        with multiprocessing.Pool(processes=args.llm_parallel_nums) as pool:
            for result in pool.imap_unordered(entity_extraction_single, tasks):
                answer = result
                if answer is not None:
                    with open(save_path, "a") as f:
                        json.dump(answer, f)
                        f.write("\n")
                    pbar.update(1)
    
    
ENTITY_ALIGNMENT_PROMPT = """
You are an expert entity disambiguation system. 
Your task is to analyze a list of potentially ambiguous entity mentions and group them by their underlying real-world referents.
Rules:
- Prefer official names over colloquial ones.
- ONLY Output the formal json format including your results!
- Always use double quotes ("") for string literals. Avoid using single quotes ('') entirely. 
[ Example Input ]
["The Minnesota Vikings", "Vikings", "Minnesota Vikings", "The Twin Towers"]

[ Example Output ]
{{"Minnesota Vikings": ["The Minnesota Vikings", "Vikings", "Minnesota Vikings"], "The Twin Towers": ["The Twin Towers"]}}

[ Input ]
{input_entity_list}
[ Output ]

"""

def dynamic_threshold(
    depth: int,
    base_threshold: float = 0.85,
    alpha: float = 0.04,
    max_threshold: float = 1
) -> float:
    threshold = base_threshold * (1 + alpha) ** depth
    return min(threshold, max_threshold)




import concurrent
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

def hyper_graph_pool(args):
    questions_list = load_question(args)
    jsonl_result = []
    tasks = [(md5_id, question, args) for md5_id, question, _ in questions_list]
    total_tasks = len(tasks)

    # 初始化模型（全局变量，所有线程共享）
    init_models(args)

    # 使用线程池替代进程池
    with ThreadPoolExecutor(max_workers=64) as executor:
        # 提交所有任务
        futures = [executor.submit(single_retrieve, task) for task in tasks]
        
        # 使用tqdm显示进度
        jsonl_result = []
        for future in tqdm(
            concurrent.futures.as_completed(futures),
            total=total_tasks,
            desc="Retrieving",
            unit="task"
        ):
            jsonl_result.append(future.result())

    os.makedirs(f'./data/{args.dataset_name}/{args.folder_name}/{args.mode}', exist_ok=True)
    write_path = f'./data/{args.dataset_name}/{args.folder_name}/{args.mode}/{get_file_name(args)}retrieve.json'
    wrtie_json(jsonl_result, write_path)
    print("Finish Retrieve...")


def get_propositions_by_split(args):
    import nltk
    nltk.download('punkt')  # 第一次使用需要下载数据
    from nltk.tokenize import sent_tokenize

    if args.dataset_name in DATASET_NEED_TOKENIZER:
        if not os.path.exists(f'./data/{args.dataset_name}/{args.folder_name}/Corpus_raw.json'):
            with open(f'./data/{args.dataset_name}/{args.folder_name}/Corpus.json', "r") as f:
                for line in f:
                    line = json.loads(line.strip())
                    long_text = line["context"]
            split_chunks = split_tokens_with_overlap(long_text)
            # breakpoint()
            os.rename(f'./data/{args.dataset_name}/{args.folder_name}/Corpus.json', 
                    f'./data/{args.dataset_name}/{args.folder_name}/Corpus_raw.json')
            
            wrtie_json(split_chunks, f'./data/{args.dataset_name}/{args.folder_name}/Corpus.json')
    
    write_data = []
    with open(f'./data/{args.dataset_name}/{args.folder_name}/Corpus.json', "r") as f:
        for line in f:
            
            line = json.loads(line.strip())
            
            sentences = sent_tokenize(line["context"])
            sentences = [sen for sen in sentences if sen != ""]
            
            write_data.append(
                {
                    "question_id": line["question_id"],
                    "generation": sentences
                }
            )
    f.close()
    save_path = f'./data/{args.dataset_name}/{args.folder_name}/Corpus_proposition.json'
    wrtie_json(write_data, save_path)
    
    print("Finish get_propositions_by_split...")

def backpack_memmap_with_selection(number, weight, w, v):
    temp_file = "temp_knapsack.npy"
    if os.path.exists(temp_file):
        os.remove(temp_file)

    # 初始化 memmap
    result = np.memmap(temp_file, dtype=np.float16, mode="w+", shape=(number+1, weight+1))
    result[0, :] = 0  # 初始条件

    # 动态规划（向量化）
    for i in tqdm(range(1, number+1), total=number, desc="Dynamic calculating..."):
        prev_row = result[i-1, :]
        current_row = prev_row.copy()  # 默认继承上一行的值

        # 向量化计算可能更新的位置
        j_values = np.arange(1, weight+1)
        mask = j_values >= w[i-1]  # 找到满足 j >= w[i-1] 的位置
        update_indices = j_values[mask]

        if len(update_indices) > 0:
            # 计算新值：prev_row[j - w[i-1]] + v[i-1]
            new_values = prev_row[update_indices - w[i-1]] + v[i-1]
            # 取最大值
            current_row[update_indices] = np.maximum(prev_row[update_indices], new_values)

        result[i, :] = current_row
        result.flush()
    # 回溯选择的物品（与原代码相同）
    selected_items = []
    j = weight
    for i in tqdm(range(number, 0, -1), total=number, desc="Choosing..."):
        if result[i, j] != result[i-1, j]:
            selected_items.append(i-1)
            j -= w[i-1]

    max_value = result[number, weight]
    del result
    os.remove(temp_file)
    return max_value, selected_items