from transformers import Pipeline
from torch import Tensor
import torch
import torch.nn.functional as F

class MyPipeline(Pipeline):
    def _sanitize_parameters(self, **kwargs):
        preprocess_kwargs = {}
        if "maybe_arg" in kwargs:
            preprocess_kwargs["maybe_arg"] = kwargs["maybe_arg"]
        return preprocess_kwargs, {}, {}

    def preprocess(self, inputs, maybe_arg=2):
        max_length = 4096
        # Tokenize the input texts
        # batch_dict = self.tokenizer(inputs, max_length=max_length - 1, return_attention_mask=False, padding=False,
        #                        truncation=True)
        inputs = inputs + self.tokenizer.eos_token
        batch_dict = self.tokenizer(inputs, max_length=max_length - 1, return_attention_mask=True, padding=False,
                                                           truncation=True, return_tensors='pt')
        # print('batch_dict',batch_dict)
        # append eos_token_id to every input_ids
        # batch_dict['input_ids'] = [input_ids + [self.tokenizer.eos_token_id] for input_ids in batch_dict['input_ids']]
        # batch_dict['input_ids'] =  batch_dict['input_ids'] + [self.tokenizer.eos_token_id]
        # print('batch_dict', batch_dict)
        # batch_dict = self.tokenizer.pad(batch_dict, padding=True, return_attention_mask=True, return_tensors='pt')

        # self.batch_dict = batch_dict
        # model_input = Tensor(inputs["input_ids"])
        # return {"model_input": model_input}
        # return  {"model_input": batch_dict}
        return batch_dict

    def _forward(self, model_inputs):
        # model_inputs == {"model_input": model_input}
        # print('padding', model_inputs)
        outputs = self.model(**model_inputs)
        embeddings = last_token_pool(outputs.last_hidden_state, model_inputs['attention_mask'])
        # Maybe {"logits": Tensor(...)}
        # return {'outputs': outputs, 'attention_mask': model_inputs['attention_mask']}
        return embeddings

    def postprocess(self, model_outputs):
        # best_class = model_outputs["logits"].softmax(-1)

        # embeddings = last_token_pool(model_outputs[0].last_hidden_state, model_outputs[1]['attention_mask'])
        # embeddings = last_token_pool(model_outputs.last_hidden_state, model_outputs['attention_mask'])
        # embeddings = model_outputs.last_hidden_state[:, -1] # might be useful


        # normalize embeddings
        embeddings = F.normalize(model_outputs, p=2, dim=1)
        # scores = (embeddings[:1] @ embeddings[1:].T) * 100
        # scores = embeddings[:1] @ embeddings[1:].T
        # print('size', embeddings.size())
        # print('size', scores.size())

        return embeddings

    # def run_single(self, inputs, preprocess_params, forward_params, postprocess_params):
    #     model_inputs = self.preprocess(inputs, **preprocess_params)
    #     # print('padding', model_inputs)
    #     model_outputs = self.forward(model_inputs, **forward_params)
    #     outputs = self.postprocess(model_outputs, **postprocess_params)
    #     return outputs


def last_token_pool(last_hidden_states: Tensor,
                 attention_mask: Tensor) -> Tensor:
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]


if __name__ == "__main__":
    from transformers.pipelines import PIPELINE_REGISTRY
    from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel
    from transformers import pipeline

    tokenizer = AutoTokenizer.from_pretrained('intfloat/e5-mistral-7b-instruct')#, padding_side='left')
    PIPELINE_REGISTRY.register_pipeline(
        "text-similarity",
        pipeline_class=MyPipeline,
        pt_model=AutoModel
    )


    pipe = pipeline("text-similarity", model='intfloat/e5-mistral-7b-instruct', tokenizer=tokenizer, device_map="auto")
    text = ['Given a conversation scenario, retrieve a text that is similar to the given scenario from the aspect of topic, semantics and theme.',
            'I want to travel to Beijing',
            'I like to go hiking on weekends.',
             'retrieve a text that is similar to the given scenario from the aspect of topic, semantics and theme.',
             'I want to travel',
             'I like to go hiking on weekends.',
            'Given a conversation scenario, retrieve a text that is similar to the given scenario from the aspect of topic, semantics and theme.',
            'I want to travel to Beijing',
            'I like to go hiking on weekends.',
            'retrieve a text that is similar to the given scenario from the aspect of topic, semantics and theme.'
            ]
    result = pipe(text)
    print('len', len(result))
    for res in result:
        print('shape', res.shape)
