#コメント追加してコミット&プッシュする

import csv
import pandas as pd
from datasets import Dataset
import torch
from transformers import AutoTokenizer
from dataclasses import dataclass
from transformers.tokenization_utils_base import PreTrainedTokenizerBase, PaddingStrategy
from typing import Optional, Union
from transformers import AutoModelForMultipleChoice, TrainingArguments, Trainer

file_path = "task/LLM/train.csv"
train = pd.read_csv(file_path)

train.head()
train = train[[]]


train_ds = Dataset.from_pandas(train)
print(train_ds[1])

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

options = "ABCDE"
indices = list(range(5))

option_to_index = {option: index for option, index in zip(options, indices)}
index_to_option = {index: option for option, index in zip(options, indices)}

def preprocess(example):
    
    first_sentence = [example['prompt']] * 5
    second_sentence = []
    for option in options:
        second_sentence.append(example[option])

    tokenized_example = tokenizer(first_sentence, second_sentence, truncation=True)
    tokenized_example['label'] = option_to_index[example['answer']]
    return tokenized_example

tokenized_train_ds = train_ds.map(preprocess, 
                                  batched=False, 
                                  remove_columns=['prompt', 'A', 'B', 'C', 'D', 'E', 'answer'])

@dataclass
class DataCollatorForMultipleChoice:
    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    
    def __call__(self, features):
        label_name = "label" if 'label' in features[0].keys() else 'labels'
        labels = [feature.pop(label_name) for feature in features]
        batch_size = len(features)
        num_choices = len(features[0]['input_ids'])
        flattened_features = [
            [{k: v[i] for k, v in feature.items()} for i in range(num_choices)] for feature in features
        ]
        flattened_features = sum(flattened_features, [])
        
        batch = self.tokenizer.pad(
            flattened_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors='pt',
        )
        batch = {k: v.view(batch_size, num_choices, -1) for k, v in batch.items()}
        batch['labels'] = torch.tensor(labels, dtype=torch.int64)
        return batch


model = AutoModelForMultipleChoice.from_pretrained("bert-base-cased")

model_dir = 'finetuned_bert'
training_args = TrainingArguments(
    output_dir=model_dir,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    learning_rate=5e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=3,
    weight_decay=0.01,
    report_to='none'
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_ds,
    eval_dataset=tokenized_train_ds,
    tokenizer=tokenizer,
    data_collator=DataCollatorForMultipleChoice(tokenizer=tokenizer),
)

trainer.train()

predictions = trainer.predict(tokenized_train_ds)

import numpy as np
def predictions_to_map_output(predictions):
    sorted_answer_indices = np.argsort(-predictions)
    top_answer_indices = sorted_answer_indices[:,:3] # Get the first three answers in each row
    top_answers = np.vectorize(index_to_option.get)(top_answer_indices)
    return np.apply_along_axis(lambda row: ' '.join(row), 1, top_answers)

predictions_to_map_output(predictions.predictions)

test_df = pd.read_csv('task/LLM/test.csv')
test_df.head()
test_df['answer'] = 'A'

# Other than that we'll preprocess it in the same way we preprocessed test.csv
test_ds = Dataset.from_pandas(test_df)
tokenized_test_ds = test_ds.map(preprocess, batched=False, remove_columns=['prompt', 'A', 'B', 'C', 'D', 'E', 'answer'])
test_predictions = trainer.predict(tokenized_test_ds)

submission_df = test_df[['id']]
submission_df['prediction'] = predictions_to_map_output(test_predictions.predictions)

submission_df.head()

submission_df.to_csv('submission.csv', index=False)