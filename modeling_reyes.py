#!/usr/bin/env python
# _*_coding:utf-8_*_
# Author   :    Junhui Yu
# Time     :    2025/1/14 16:07

import warnings
from typing import Any, List, Optional, Tuple, Union
import torch.utils.checkpoint
import transformers
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers import GenerationConfig
from transformers import Qwen2ForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import ModelOutput, logging
import torch.nn.functional as F

from .configuration_reyes import ReyesConfig
from .conversation import get_conv_template
from .modeling_intern_vit import InternVisionModel, has_flash_attn

logger = logging.get_logger(__name__)


def version_cmp(v1, v2, op='eq'):
    import operator

    from packaging import version
    op_func = getattr(operator, op)
    return op_func(version.parse(v1), version.parse(v2))


class ReyesModel(PreTrainedModel):
    config_class = ReyesConfig
    main_input_name = 'pixel_values'
    _supports_flash_attn_2 = True
    _no_split_modules = ['InternVisionModel', 'Qwen2DecoderLayer']

    def __init__(self, config: ReyesConfig, vision_model=None, language_model=None, use_flash_attn=True):
        super().__init__(config)

        assert version_cmp(transformers.__version__, '4.44.2', 'ge')
        image_size = config.force_image_size or config.vision_config.image_size
        patch_size = config.vision_config.patch_size
        self.patch_size = patch_size
        self.select_layer = config.select_layer
        self.llm_arch_name = config.llm_config.architectures[0]
        self.template = config.template
        self.num_image_token = int((image_size // patch_size) ** 2 * (config.downsample_ratio ** 2))
        self.downsample_ratio = config.downsample_ratio
        self.ps_version = config.ps_version
        use_flash_attn = use_flash_attn if has_flash_attn else False
        config.vision_config.use_flash_attn = True if use_flash_attn else False
        config.llm_config._attn_implementation = 'flash_attention_2' if use_flash_attn else 'eager'

        logger.info(f'num_image_token: {self.num_image_token}')
        logger.info(f'ps_version: {self.ps_version}')
        if vision_model is not None:
            self.vision_model = vision_model
        else:
            self.vision_model = InternVisionModel(config.vision_config)
        if language_model is not None:
            self.language_model = language_model
        else:
            if config.llm_config.architectures[0] == 'Qwen2ForCausalLM':
                self.language_model = Qwen2ForCausalLM(config.llm_config)
                # self.language_model = AutoLigerKernelForCausalLM(config.llm_config)
            else:
                raise NotImplementedError(f'{config.llm_config.architectures[0]} is not implemented.')

        vit_hidden_size = config.vision_config.hidden_size
        llm_intermediate_size = config.llm_config.intermediate_size
        llm_hidden_size = config.llm_config.hidden_size

        self.mlp1 = nn.Sequential(
            nn.LayerNorm(vit_hidden_size * int(1 / self.downsample_ratio) ** 2),
            nn.Linear(vit_hidden_size * int(1 / self.downsample_ratio) ** 2, llm_intermediate_size, bias=False),
            nn.GELU(),
            nn.Linear(llm_intermediate_size, llm_hidden_size, bias=False)
        )

        self.img_context_token_id = None
        self.conv_template = get_conv_template(self.template)
        self.system_message = self.conv_template.system_message

    def forward(
            self,
            pixel_values: torch.FloatTensor,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            image_flags: Optional[torch.LongTensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # image_flags = image_flags.squeeze(-1)
        input_embeds = self.language_model.get_input_embeddings()(input_ids)

        vit_embeds = self.extract_feature(pixel_values)
        # vit_embeds = vit_embeds[image_flags == 1]
        vit_batch_size = pixel_values.shape[0]

        B, N, C = input_embeds.shape
        input_embeds = input_embeds.reshape(B * N, C)

        # if torch.distributed.get_rank() == 0:
        #     print(f'dynamic ViT batch size: {vit_batch_size}, images per sample: {vit_batch_size / B}, dynamic token length: {N}')

        input_ids = input_ids.reshape(B * N)
        selected = (input_ids == self.img_context_token_id)
        try:
            input_embeds[selected] = input_embeds[selected] * 0.0 + vit_embeds.reshape(-1, C)
        except Exception as e:
            vit_embeds = vit_embeds.reshape(-1, C)
            print(f'warning: {e}, input_embeds[selected].shape={input_embeds[selected].shape}, '
                  f'vit_embeds.shape={vit_embeds.shape}')
            n_token = selected.sum()
            input_embeds[selected] = input_embeds[selected] * 0.0 + vit_embeds[:n_token]

        input_embeds = input_embeds.reshape(B, N, C)

        outputs = self.language_model(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        logits = outputs.logits

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.language_model.config.vocab_size)
            shift_labels = shift_labels.view(-1)

            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def pixel_shuffle(self, x, scale_factor=0.5):
        n, w, h, c = x.size()
        # N, W, H, C --> N, W, H * scale, C // scale
        x = x.view(n, w, int(h * scale_factor), int(c / scale_factor))
        # N, W, H * scale, C // scale --> N, H * scale, W, C // scale
        x = x.permute(0, 2, 1, 3).contiguous()
        # N, H * scale, W, C // scale --> N, H * scale, W * scale, C // (scale ** 2)
        x = x.view(n, int(h * scale_factor), int(w * scale_factor),
                   int(c / (scale_factor * scale_factor)))
        if self.ps_version == 'v1':
            warnings.warn("In ps_version 'v1', the height and width have not been swapped back, "
                          'which results in a transposed image.')
        else:
            x = x.permute(0, 2, 1, 3).contiguous()
        return x

    def extract_feature(self, pixel_values):
        if self.select_layer == -1:
            vit_embeds = self.vision_model(
                pixel_values=pixel_values,
                output_hidden_states=False,
                return_dict=True).last_hidden_state
        else:
            vit_embeds = self.vision_model(
                pixel_values=pixel_values,
                output_hidden_states=True,
                return_dict=True).hidden_states[self.select_layer]
        vit_embeds = vit_embeds[:, 1:, :]

        h = w = int(vit_embeds.shape[1] ** 0.5)
        vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], h, w, -1)
        vit_embeds = self.pixel_shuffle(vit_embeds, scale_factor=self.downsample_ratio)
        vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], -1, vit_embeds.shape[-1])
        vit_embeds = self.mlp1(vit_embeds)
        return vit_embeds

    def chat(self, tokenizer, pixel_values, question, generation_config, history=None, return_history=False,
             num_patches_list=None, IMG_START_TOKEN='<|vision_start|>', IMG_END_TOKEN='<|vision_end|>',
             IMG_CONTEXT_TOKEN='<|vision_pad|>', verbose=False, visual_features=None):

        if history is None and pixel_values is not None and '<image>' not in question:
            question = '<image>\n' + question

        if num_patches_list is None:
            num_patches_list = [pixel_values.shape[0]] if pixel_values is not None else []
        assert pixel_values is None or len(pixel_values) == sum(num_patches_list)

        img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
        self.img_context_token_id = img_context_token_id

        template = get_conv_template(self.template)
        template.system_message = self.system_message
        eos_token_id = tokenizer.convert_tokens_to_ids(template.sep)

        history = [] if history is None else history
        for (old_question, old_answer) in history:
            template.append_message(template.roles[0], old_question)
            template.append_message(template.roles[1], old_answer)
        template.append_message(template.roles[0], question)
        template.append_message(template.roles[1], None)
        query = template.get_prompt()
        # print('query:', query)

        if verbose and pixel_values is not None:
            image_bs = pixel_values.shape[0]
            print(f'dynamic ViT batch size: {image_bs}')

        for num_patches in num_patches_list:
            tile_pos_identifiers = [f"<tile_{i}>" for i in range(1, num_patches)] + ["<tile_global_thumbnail>"]
            image_tokens = ''
            for tile_pos_identifier in tile_pos_identifiers:
                image_tokens += tile_pos_identifier + IMG_CONTEXT_TOKEN * self.num_image_token
            image_tokens = IMG_START_TOKEN + image_tokens + IMG_END_TOKEN
            query = query.replace('<image>', image_tokens, 1)

        model_inputs = tokenizer(query, return_tensors='pt')
        input_ids = model_inputs['input_ids'].cuda()
        attention_mask = model_inputs['attention_mask'].cuda()
        generation_config['eos_token_id'] = eos_token_id
        generation_output = self.generate(
            pixel_values=pixel_values,
            visual_features=visual_features,
            input_ids=input_ids,
            attention_mask=attention_mask,
            **generation_config
        )
        response = tokenizer.batch_decode(generation_output, skip_special_tokens=True)[0]
        response = response.split(template.sep)[0].strip()
        history.append((question, response))
        if return_history:
            return response, history
        else:
            query_to_print = query.replace(IMG_CONTEXT_TOKEN, '')
            query_to_print = query_to_print.replace(f'{IMG_START_TOKEN}{IMG_END_TOKEN}', '<image>')
            if verbose:
                print(query_to_print, response)
            return response

    def chat_without_sys_prompt(self, tokenizer, pixel_values, question, generation_config, history=None,
                                return_history=False,
                                num_patches_list=None, IMG_START_TOKEN='<|vision_start|>',
                                IMG_END_TOKEN='<|vision_end|>',
                                IMG_CONTEXT_TOKEN='<|vision_pad|>', verbose=False, visual_features=None):

        if history is None and pixel_values is not None and '<image>' not in question:
            question = '<image>\n' + question

        if num_patches_list is None:
            num_patches_list = [pixel_values.shape[0]] if pixel_values is not None else []
        assert pixel_values is None or len(pixel_values) == sum(num_patches_list)

        img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
        self.img_context_token_id = img_context_token_id

        template = get_conv_template(self.template)
        system_prompt = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>"  # override dummy system prompt
        template.system_message = system_prompt
        eos_token_id = tokenizer.convert_tokens_to_ids(template.sep)

        history = [] if history is None else history
        for (old_question, old_answer) in history:
            template.append_message(template.roles[0], old_question)
            template.append_message(template.roles[1], old_answer)
        template.append_message(template.roles[0], question)
        template.append_message(template.roles[1], None)
        query = template.get_prompt()

        if verbose and pixel_values is not None:
            image_bs = pixel_values.shape[0]
            print(f'dynamic ViT batch size: {image_bs}')

        query = query[len(system_prompt):]

        for num_patches in num_patches_list:
            tile_pos_identifiers = [f"<tile_{i}>" for i in range(1, num_patches)] + ["<tile_global_thumbnail>"]
            image_tokens = ''
            for tile_pos_identifier in tile_pos_identifiers:
                image_tokens += tile_pos_identifier + IMG_CONTEXT_TOKEN * self.num_image_token
            image_tokens = IMG_START_TOKEN + image_tokens + IMG_END_TOKEN
            query = query.replace('<image>', image_tokens, 1)

        model_inputs = tokenizer(query, return_tensors='pt')
        input_ids = model_inputs['input_ids'].cuda()
        attention_mask = model_inputs['attention_mask'].cuda()
        generation_config['eos_token_id'] = eos_token_id
        generation_output = self.generate(
            pixel_values=pixel_values,
            visual_features=visual_features,
            input_ids=input_ids,
            attention_mask=attention_mask,
            **generation_config
        )
        response = tokenizer.batch_decode(generation_output, skip_special_tokens=True)[0]
        response = response.split(template.sep)[0].strip()
        history.append((question, response))
        if return_history:
            return response, history
        else:
            query_to_print = query.replace(IMG_CONTEXT_TOKEN, '')
            query_to_print = query_to_print.replace(f'{IMG_START_TOKEN}{IMG_END_TOKEN}', '<image>')
            if verbose:
                print(query_to_print, response)
            return response

    def chat_without_chat_prompt(self, tokenizer, pixel_values, question, generation_config,
                                 num_patches_list=None, IMG_START_TOKEN='<|vision_start|>',
                                 IMG_END_TOKEN='<|vision_end|>',
                                 IMG_CONTEXT_TOKEN='<|vision_pad|>', verbose=False, visual_features=None):

        if pixel_values is not None and '<image>' not in question:
            question = '<image>\n' + question

        if num_patches_list is None:
            num_patches_list = [pixel_values.shape[0]] if pixel_values is not None else []
        assert pixel_values is None or len(pixel_values) == sum(num_patches_list)

        img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
        self.img_context_token_id = img_context_token_id

        template = get_conv_template(self.template)
        template.system_message = self.system_message
        eos_token_id = tokenizer.convert_tokens_to_ids(template.sep)

        if verbose and pixel_values is not None:
            image_bs = pixel_values.shape[0]
            print(f'dynamic ViT batch size: {image_bs}')

        query = question

        for num_patches in num_patches_list:
            tile_pos_identifiers = [f"<tile_{i}>" for i in range(1, num_patches)] + ["<tile_global_thumbnail>"]
            image_tokens = ''
            for tile_pos_identifier in tile_pos_identifiers:
                image_tokens += tile_pos_identifier + IMG_CONTEXT_TOKEN * self.num_image_token
            image_tokens = IMG_START_TOKEN + image_tokens + IMG_END_TOKEN
            query = query.replace('<image>', image_tokens, 1)

        model_inputs = tokenizer(query, return_tensors='pt')
        input_ids = model_inputs['input_ids'].cuda()
        attention_mask = model_inputs['attention_mask'].cuda()
        generation_config['eos_token_id'] = eos_token_id
        generation_output = self.generate(
            pixel_values=pixel_values,
            visual_features=visual_features,
            input_ids=input_ids,
            attention_mask=attention_mask,
            **generation_config
        )
        response = tokenizer.batch_decode(generation_output, skip_special_tokens=True)[0]
        response = response.split(template.sep)[0].strip()

        query_to_print = query.replace(IMG_CONTEXT_TOKEN, '')
        query_to_print = query_to_print.replace(f'{IMG_START_TOKEN}{IMG_END_TOKEN}', '<image>')
        if verbose:
            print(query_to_print, response)
        return response

    @torch.no_grad()
    def generate(
            self,
            pixel_values: Optional[torch.FloatTensor] = None,
            input_ids: Optional[torch.FloatTensor] = None,
            attention_mask: Optional[torch.LongTensor] = None,
            visual_features: Optional[torch.FloatTensor] = None,
            generation_config: Optional[GenerationConfig] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            **generate_kwargs,
    ) -> torch.LongTensor:

        # assert self.img_context_token_id is not None
        if pixel_values is not None:
            if visual_features is not None:
                vit_embeds = visual_features.cuda()
                vit_embeds = self.mlp1(vit_embeds)
            else:
                vit_embeds = self.extract_feature(pixel_values)

            input_embeds = self.language_model.get_input_embeddings()(input_ids)
            B, N, C = input_embeds.shape
            input_embeds = input_embeds.reshape(B * N, C)

            input_ids = input_ids.reshape(B * N)
            selected = (input_ids == self.img_context_token_id)
            assert selected.sum() != 0
            input_embeds[selected] = vit_embeds.reshape(-1, C).to(input_embeds.device)

            input_embeds = input_embeds.reshape(B, N, C)
        else:
            input_embeds = self.language_model.get_input_embeddings()(input_ids)

        outputs = self.language_model.generate(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            generation_config=generation_config,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            use_cache=True,
            **generate_kwargs,
        )

        return outputs

    def chat_batch(
            self,
            tokenizer,
            pixel_values_list,
            questions,
            generation_config,
            histories=None,
            return_histories=False,
            num_patches_lists=None,
            IMG_START_TOKEN='<|vision_start|>',
            IMG_END_TOKEN='<|vision_end|>',
            IMG_CONTEXT_TOKEN='<|vision_pad|>',
            verbose=False,
            visual_features_list=None
    ):

        if histories is None:
            histories = [[] for _ in questions]

        img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
        self.img_context_token_id = img_context_token_id
        # Get eos_token_id from the template
        template = get_conv_template(self.template)
        template.system_message = self.system_message
        eos_token_id = tokenizer.convert_tokens_to_ids(template.sep)
        generation_config['eos_token_id'] = eos_token_id

        queries = []
        input_ids_list = []
        attention_mask_list = []

        for idx in range(len(questions)):
            question = questions[idx]
            history = histories[idx]
            pixel_values = pixel_values_list[idx] if pixel_values_list[idx] is not None else None
            num_patches_list = [pixel_values.shape[0]] if pixel_values is not None else []

            if not history and pixel_values is not None and '<image>' not in question:
                question = '<image>\n' + question

            template_i = get_conv_template(self.template)
            template_i.system_message = self.system_message
            for (old_question, old_answer) in history:
                template_i.append_message(template_i.roles[0], old_question)
                template_i.append_message(template_i.roles[1], old_answer)
            template_i.append_message(template_i.roles[0], question)
            template_i.append_message(template_i.roles[1], None)
            query = template_i.get_prompt()
            # Handle image tokens
            if pixel_values is not None:
                for num_patches in num_patches_list:
                    tile_pos_identifiers = [f"<tile_{i}>" for i in range(1, num_patches)] + ["<tile_global_thumbnail>"]
                    image_tokens = ''
                    for tile_pos_identifier in tile_pos_identifiers:
                        image_tokens += tile_pos_identifier + IMG_CONTEXT_TOKEN * self.num_image_token
                    image_tokens = IMG_START_TOKEN + image_tokens + IMG_END_TOKEN
                    query = query.replace('<image>', image_tokens, 1)

            model_inputs = tokenizer(
                query,
                return_tensors='pt',
                padding=True,
                truncation=True
            )
            input_ids = model_inputs['input_ids'].cuda()
            attention_mask = model_inputs['attention_mask'].cuda()
            input_ids_list.append(input_ids)
            attention_mask_list.append(attention_mask)

        # Call the generate function
        generation_output = self.generate_batch(
            pixel_values_list=pixel_values_list,
            input_ids_list=input_ids_list,
            attention_mask_list=attention_mask_list,
            **generation_config
        )
        responses = tokenizer.batch_decode(generation_output, skip_special_tokens=True)

        outputs = []
        for idx, response in enumerate(responses):
            response = response.split(template.sep)[0].strip()
            histories[idx].append((questions[idx], response))
            outputs.append(response)

        if return_histories:
            return outputs, histories
        else:
            if verbose:
                for idx, query in enumerate(queries):
                    query_to_print = query.replace(IMG_CONTEXT_TOKEN, '')
                    query_to_print = query_to_print.replace(f'{IMG_START_TOKEN}{IMG_END_TOKEN}', '<image>')
                    print(query_to_print, outputs[idx])
            return outputs

    @torch.no_grad()
    def generate_batch(
            self,
            pixel_values_list: Optional[List[torch.FloatTensor]] = None,
            input_ids_list: Optional[List[torch.FloatTensor]] = None,
            attention_mask_list: Optional[List[torch.LongTensor]] = None,
            visual_features: Optional[torch.FloatTensor] = None,
            generation_config: Optional[GenerationConfig] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            **generate_kwargs,
    ) -> torch.LongTensor:
        input_embeds_list = []
        attention_mask_padded_list = []

        max_seq_length = max(input_ids.shape[1] for input_ids in input_ids_list)

        for pixel_values, input_ids, attention_mask in zip(pixel_values_list, input_ids_list, attention_mask_list):
            if pixel_values is not None:
                if visual_features is not None:
                    vit_embeds = visual_features.cuda()
                    vit_embeds = self.mlp1(vit_embeds)
                else:
                    vit_embeds = self.extract_feature(pixel_values)

                input_embeds = self.language_model.get_input_embeddings()(input_ids)
                B, N, C = input_embeds.shape
                input_embeds = input_embeds.reshape(B * N, C)

                input_ids = input_ids.reshape(B * N)
                selected = (input_ids == self.img_context_token_id)
                assert selected.sum() != 0, "No valid image context token IDs found."
                input_embeds[selected] = vit_embeds.reshape(-1, C).to(input_embeds.device)

                input_embeds = input_embeds.reshape(B, N, C)
            else:
                input_embeds = self.language_model.get_input_embeddings()(input_ids)

            seq_length = input_embeds.shape[1]
            if seq_length < max_seq_length:
                pad_size = max_seq_length - seq_length
                input_embeds = F.pad(input_embeds, (0, 0, 0, pad_size))
                attention_mask = F.pad(attention_mask, (0, pad_size))

            input_embeds_list.append(input_embeds)
            attention_mask_padded_list.append(attention_mask)

        input_embeds = torch.cat(input_embeds_list, dim=0)
        attention_mask = torch.cat(attention_mask_padded_list, dim=0)

        outputs = self.language_model.generate(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            generation_config=generation_config,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            use_cache=True,
            **generate_kwargs,
        )

        return outputs
