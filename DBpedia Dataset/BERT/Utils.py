from datasets import load_dataset
dbpedia = load_dataset('dbpedia_14')
from transformers import BertTokenizer
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from sklearn.model_selection import train_test_split



def encode(docs, tokenizer):
    '''
    This function takes a list of texts and returns input_ids and attention_mask of texts
    '''
    encoded_dict = tokenizer.batch_encode_plus(
        docs, add_special_tokens=True, max_length=128, padding='max_length',
        return_attention_mask=True, truncation=True, return_tensors='pt'
    )
    input_ids = encoded_dict['input_ids']
    attention_masks = encoded_dict['attention_mask']
    return input_ids, attention_masks

def get_trainvalidtest_loaders(BATCH_SIZE=16):
    label_names = dbpedia["train"].features['label'].names
    print(label_names)
    dbpedia.set_format(type="pandas")
    train_df_val = dbpedia['train'][:]
    test_df = dbpedia['test'][:]
    train_df, valid_df = train_test_split(train_df_val, test_size=0.1, stratify=train_df_val['label'], random_state=42)
    print(train_df.head())
    print("---------counts before taking subset of data------------")
    print(train_df['label'].value_counts())
    print(valid_df['label'].value_counts())
    print(test_df['label'].value_counts())
    print("---------------------------------------------------------")

    num_train = 500  # number of data items to use for training
    num_validate = 80
    num_test = 65
    train_df = train_df.groupby('label').apply(lambda x: x.sample(min(len(x), num_train))).reset_index(drop=True)
    valid_df = valid_df.groupby('label').apply(lambda x: x.sample(min(len(x), num_train))).reset_index(drop=True)
    test_df = test_df.groupby('label').apply(lambda x: x.sample(min(len(x), num_train))).reset_index(drop=True)
    print("---------counts after taking subset of data------------")
    print(train_df['label'].value_counts())
    print(valid_df['label'].value_counts())
    print(test_df['label'].value_counts())
    print("---------------------------------------------------------")

    # Use BERT pretrained tokenizer
    tokenizer = BertTokenizer.from_pretrained("bert-base-cased", do_lower_case=True)
    print(tokenizer)
    print(train_df.columns)
    train_list = train_df['content'].values.tolist()
    train_input_ids, train_att_masks = encode(train_list, tokenizer)
    valid_input_ids, valid_att_masks = encode(valid_df['content'].values.tolist(), tokenizer)
    test_input_ids, test_att_masks = encode(test_df['content'].values.tolist(), tokenizer)

    num_check = 5
    print(train_list[num_check])
    print(train_input_ids[num_check])
    print(train_att_masks[num_check])

    # Get the labels
    train_y = torch.LongTensor(train_df['label'].values.tolist())
    valid_y = torch.LongTensor(valid_df['label'].values.tolist())
    test_y = torch.LongTensor(test_df['label'].values.tolist())
    print(train_y.size(), valid_y.size(), test_y.size())

    train_dataset = TensorDataset(train_input_ids, train_att_masks, train_y)
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=BATCH_SIZE)

    valid_dataset = TensorDataset(valid_input_ids, valid_att_masks, valid_y)
    valid_sampler = SequentialSampler(valid_dataset)
    valid_dataloader = DataLoader(valid_dataset, sampler=valid_sampler, batch_size=BATCH_SIZE)

    test_dataset = TensorDataset(test_input_ids, test_att_masks, test_y)
    test_sampler = SequentialSampler(test_dataset)
    test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=BATCH_SIZE)

    return train_dataloader, valid_dataloader, test_dataloader, train_df, valid_df, label_names
