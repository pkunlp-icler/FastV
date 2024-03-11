class LlamaModel:
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")

        seq_length_with_past = seq_length
        past_key_values_length = 0

        if past_key_values is not None:
            past_key_values_length = past_key_values[0][0].shape[2]
            seq_length_with_past = seq_length_with_past + past_key_values_length

        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
            )
            position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
        else:
            position_ids = position_ids.view(-1, seq_length).long()

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        # embed positions
        if attention_mask is None:
            attention_mask = torch.ones(
                (batch_size, seq_length_with_past), dtype=torch.bool, device=inputs_embeds.device
            )
        attention_mask = self._prepare_decoder_attention_mask(
            attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length
        )

        hidden_states = inputs_embeds

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = () if use_cache else None

        for idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            past_key_value = past_key_values[idx] if past_key_values is not None else None
            if self.gradient_checkpointing and self.training:
                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        # None for past_key_value
                        return module(*inputs, output_attentions, None)

                    return custom_forward
                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(decoder_layer),
                    hidden_states,
                    attention_mask,
                    position_ids,
                    None,
                )
            else:

                # FASTV START --------------------------------------------------

                USE_FAST_V = self.use_fast_v # 是否使用fastv ，True
                SYS_LENGTH= self.fast_v_sys_length # system prompt长度，就是image token前面token的数量，用来定位image tokens，llava是36
                IMAGE_TOKEN_LENGTH = self.fast_v_image_token_length # image tokens 的数量 llava是576
                ATTENTION_RANK = self.fast_v_attention_rank # 裁剪后image tokens 的数量，（1-R）x576 ， 如 R=50% 就是 288
                AGG_LAYER = self.fast_v_agg_layer # K的值，从第几层开始裁剪token，K=2 default

                if USE_FAST_V:
                    if idx<AGG_LAYER:
                        new_attention_mask = torch.ones(
                            (batch_size, seq_length_with_past), dtype=torch.bool, device=inputs_embeds.device
                        )
                        new_attention_mask = self._prepare_decoder_attention_mask(
                            new_attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length
                        )
                        
                    elif idx==AGG_LAYER:
                        if idx!=0:
                            last_layer_attention = layer_outputs[1]
                            # compute average attention over different head
                            last_layer_attention_avg = torch.mean(last_layer_attention, dim=1)[0]
                            # generate new attention mask based on the average attention, sample the top 144 tokens with highest attention
                            last_layer_attention_avg_last_tok = last_layer_attention_avg[-1]
                            # get the attention in image token
                            last_layer_attention_avg_last_tok_image = last_layer_attention_avg_last_tok[SYS_LENGTH:SYS_LENGTH+IMAGE_TOKEN_LENGTH]
                            # get the indexs of the top ATTENTION_RANK tokens
                            top_attention_rank_index = last_layer_attention_avg_last_tok_image.topk(ATTENTION_RANK).indices + SYS_LENGTH
                            # generate new attention mask
                            gen_attention_mask = torch.ones((batch_size, seq_length_with_past), dtype=torch.bool, device=inputs_embeds.device)
                            gen_attention_mask[:,SYS_LENGTH:SYS_LENGTH+IMAGE_TOKEN_LENGTH] = False
                            gen_attention_mask[:,top_attention_rank_index] = True

                            gen_attention_mask = self._prepare_decoder_attention_mask(
                                gen_attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length
                            )
                            new_attention_mask = gen_attention_mask
                        
                        else:
                            # idx==0 , random attention mask
                            # import pdb
                            # pdb.set_trace()
                            gen_attention_mask = torch.ones((batch_size, seq_length_with_past), dtype=torch.bool, device=inputs_embeds.device)

                            rand_image_attention_mask = [1]*ATTENTION_RANK + [0]*(IMAGE_TOKEN_LENGTH-ATTENTION_RANK)
                            random.shuffle(rand_image_attention_mask)

                            # import pdb
                            # pdb.set_trace()
                            gen_attention_mask[:, SYS_LENGTH:SYS_LENGTH+IMAGE_TOKEN_LENGTH] = torch.tensor(rand_image_attention_mask, dtype=attention_mask.dtype, device=inputs_embeds.device)
                            gen_attention_mask = self._prepare_decoder_attention_mask(
                                gen_attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length
                            )
                            new_attention_mask = gen_attention_mask


                    else:
                        new_attention_mask = gen_attention_mask
                
                else: 
                    new_attention_mask = attention_mask



                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=new_attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )