import math
import torch
import numpy as np
from torch.utils.data import DataLoader
from transformers.trainer_callback import TrainerCallback

class ISTCallback(TrainerCallback):
    def __init__(self, model, dataset, data_collator):
        super().__init__()
        self.batch_size = 16
        self.model = model.get_base_model()
        self.dataset = dataset
        self.data_collator = data_collator
        self.dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, collate_fn=self.data_collator)
        self.dataloader = iter(self.dataloader)
        # Determine the way to access layers based on the model type
        class_to_layers_map = {
            'LlamaForCausalLM': 'model.model.layers',
            'Qwen2ForCausalLM': 'model.model.layers',
            'MistralForCausalLM': 'model.model.layers',
            'MixtralForCausalLM': 'model.model.layers',
            'GemmaForCausalLM': 'model.model.layers',
            'GPT2LMHeadModel': 'model.transformer.h',
        }
        model_class_name = self.model.__class__.__name__
        if model_class_name in class_to_layers_map:
            self.layers_attribute = class_to_layers_map[model_class_name]
        else:
            print(model_class_name)
            raise NotImplementedError

        self.total_layers = len(eval('self.' + self.layers_attribute))  # Dynamically execute to get the number of layers
        self.importance_score = torch.zeros(self.total_layers)

        ### hyper parameters
        self.rl_step = 3
        self.rl_lr = 10

        self.response_suppression_factor = 0.25
        self.update_importance_interval_steps = 10

        self.n_layers_updated = int(self.total_layers * 0.25)
        self.n_layers_suppressed = int(self.total_layers * 0.5)
        ###

        self.active_layers_indices = []
        self.trainable_module_name = []
        self.raw_scaling = None
        layers = eval('self.' + self.layers_attribute)
        for idx in range(self.total_layers):
            for name, module in layers[idx].named_modules():
                if hasattr(module, 'scaling'):
                    self.raw_scaling = module.scaling
                if hasattr(module, 'adapter_scaling'):
                    self.raw_scaling = module.adapter_scaling
                if hasattr(module, 'disable_adapters'):
                    for name, param in module.named_parameters():
                        if param.requires_grad and name not in self.trainable_module_name:
                            self.trainable_module_name.append(name)

        if self.raw_scaling is not None:
            print(f'default scaling is {self.raw_scaling}')
        else:
            raise NotImplementedError

    def sampling_less_important_selection(self, num):
        prob = self.importance_score.sigmoid()
        select = torch.sort(torch.multinomial(prob, num))[0]
        return select

    def sampling_more_important_selection(self, num):
        prob = (-self.importance_score).sigmoid()
        select = torch.sort(torch.multinomial(prob, num))[0]
        return select

    def tensor_in_list(self, tensor_list, new_tensor):
        for tensor in tensor_list:
            if torch.equal(tensor, new_tensor):
                return True
        return False

    def freeze_all_layers(self):
        layers = eval('self.' + self.layers_attribute)  # Dynamically execute to get layers
        for layer in layers:
            for param in layer.parameters():
                param.requires_grad = False

    def on_step_begin(self, args, state, control, **kwargs):
        # Check if it's time to switch active layers, including at step 0
        if state.global_step % self.update_importance_interval_steps == 0 and state.global_step > 0:
            selects = []
            rets = []
            try:
                val_batch = next(self.dataloader)
            except:
                self.dataloader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True,
                                             collate_fn=self.data_collator)
                self.dataloader = iter(self.dataloader)
                val_batch = next(self.dataloader)

            for k, v in val_batch.items():
                val_batch[k] = v.cuda()

            for k in range(self.rl_step):
                select = self.sampling_less_important_selection(self.n_layers_suppressed)
                while self.tensor_in_list(selects, select):
                    select = self.sampling_less_important_selection(self.n_layers_suppressed)
                selects.append(select)
                self.switch_active_adapter(select)

                self.model.eval()
                with torch.inference_mode():
                    outputs = self.model(**val_batch)
                self.model.train()
                loss = outputs.loss
                rets.append(loss.item())

            rewards = []
            for i in range(self.rl_step):
                rewards.append(math.exp(-rets[i]))

            _mean = np.mean(rewards)

            rewards = np.array([(r - _mean) for r in rewards]).tolist()

            prob = self.importance_score.sigmoid()

            for k in range(self.rl_step):
                for i in range(self.total_layers):
                    if i in selects[k]:
                        self.importance_score[i] += rewards[k] * prob[i] * (1 - prob[i]) * self.rl_lr
                    # else:
                    #     self.importance_score[i] -= rewards[k] * prob[i] * (1 - prob[i]) * self.rl_lr
            if state.global_step % 100==0:
                print(prob)
            self.active_all_adapter()
        self.switch_active_layers()

    def active_all_adapter(self):
        self.model.train()
        layers = eval('self.' + self.layers_attribute)
        for idx in range(self.total_layers):
            for name, module in layers[idx].named_modules():
                if hasattr(module, 'scaling'):
                    module.scaling = self.raw_scaling
                if hasattr(module, 'adapter_scaling'):
                    module.adapter_scaling = self.raw_scaling

    def switch_active_adapter(self, select):
        layers = eval('self.' + self.layers_attribute)
        for idx in range(self.total_layers):
            if idx in select:  # disable lora
                for name, module in layers[idx].named_modules():
                    if hasattr(module, 'scaling'):
                        module.scaling = self.raw_scaling * self.response_suppression_factor
                    if hasattr(module, 'adapter_scaling'):
                        module.adapter_scaling = self.raw_scaling * self.response_suppression_factor
            else:
                for name, module in layers[idx].named_modules():
                    if hasattr(module, 'scaling'):
                        module.scaling = self.raw_scaling
                    if hasattr(module, 'adapter_scaling'):
                        module.adapter_scaling = self.raw_scaling

    def switch_active_layers(self):
        # First, disable gradients for all layers
        self.freeze_all_layers()

        # Randomly select n_layers to activate
        layers = eval('self.' + self.layers_attribute)  # Re-fetch layer references
        self.active_layers_indices = self.sampling_more_important_selection(self.n_layers_updated)
        print(
            f"Total layers: {self.total_layers}, Activating layers at indices: {self.active_layers_indices} for the next steps.",
            flush=True)

        # Enable gradients only for the selected layers
        for idx in self.active_layers_indices:
            for name, module in layers[idx].named_modules():
                if hasattr(module, 'disable_adapters'):
                    for name, param in module.named_parameters():
                        if name in self.trainable_module_name:
                            param.requires_grad = True