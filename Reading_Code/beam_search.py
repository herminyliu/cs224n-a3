def beam_search(self, src_sent: List[str], beam_size: int = 5, max_decoding_time_step: int = 70) -> List[Hypothesis]:
    """
    给定一个源句子，使用 beam search 生成目标语言的翻译。
    @param src_sent (List[str]): 一个源句子（单词列表）
    @param beam_size (int): beam size（保留的候选序列数）
    @param max_decoding_time_step (int): 解码RNN的最大步数
    @returns hypotheses (List[Hypothesis]): 一个包含翻译假设的列表，每个假设有以下字段：
            value: List[str]: 翻译生成的目标句子（单词列表）
            score: float: 该目标句子的对数似然得分
    """

    # 将源句子转换为模型输入格式
    src_sents_var = self.vocab.src.to_input_tensor([src_sent], self.device)

    # 编码源句子，得到编码表示和解码器初始状态
    src_encodings, dec_init_vec = self.encode(src_sents_var, [len(src_sent)])

    # 通过线性投影对编码表示进行处理，为注意力机制做准备
    src_encodings_att_linear = self.att_projection(src_encodings)

    # 初始化隐状态为解码器的初始状态
    h_tm1 = dec_init_vec
    # 初始化注意力上下文为零张量
    att_tm1 = torch.zeros(1, self.hidden_size, device=self.device)

    # 获取词典中的句子结束标记 ID
    eos_id = self.vocab.tgt['</s>']

    # 初始化候选序列和对应的得分
    hypotheses = [['<s>']]  # 每个序列从起始标记 '<s>' 开始
    hyp_scores = torch.zeros(len(hypotheses), dtype=torch.float, device=self.device)  # 对应的对数似然得分
    completed_hypotheses = []  # 存储已完成的候选序列

    t = 0
    # 开始解码循环，直到找到足够的候选序列或达到最大步数
    while len(completed_hypotheses) < beam_size and t < max_decoding_time_step:
        t += 1
        hyp_num = len(hypotheses)  # 当前候选序列的数量

        # 扩展源句子编码表示以适应多个候选序列
        exp_src_encodings = src_encodings.expand(hyp_num, src_encodings.size(1), src_encodings.size(2))
        exp_src_encodings_att_linear = src_encodings_att_linear.expand(hyp_num,
                                                                       src_encodings_att_linear.size(1),
                                                                       src_encodings_att_linear.size(2))

        # 获取每个候选序列的最后一个词，并查找其在目标词典中的 ID
        y_tm1 = torch.tensor([self.vocab.tgt[hyp[-1]] for hyp in hypotheses], dtype=torch.long, device=self.device)
        # 将这些词转换为嵌入表示
        y_t_embed = self.model_embeddings.target(y_tm1)

        # 将词嵌入与上一时间步的注意力上下文拼接在一起，作为当前输入
        x = torch.cat([y_t_embed, att_tm1], dim=-1)

        # 通过解码器进行一步计算，获取新的隐状态、注意力上下文和其他输出
        (h_t, cell_t), att_t, _ = self.step(x, h_tm1, exp_src_encodings, exp_src_encodings_att_linear, enc_masks=None)

        # 计算目标词的对数概率分布
        log_p_t = F.log_softmax(self.target_vocab_projection(att_t), dim=-1)

        # 计算新候选序列的得分，保持 beam_size 数量的候选序列
        live_hyp_num = beam_size - len(completed_hypotheses)
        contiuating_hyp_scores = (hyp_scores.unsqueeze(1).expand_as(log_p_t) + log_p_t).view(-1)
        top_cand_hyp_scores, top_cand_hyp_pos = torch.topk(contiuating_hyp_scores, k=live_hyp_num)

        # 找出最优的前 beam_size 个候选序列的前序 ID 和词 ID
        prev_hyp_ids = torch.div(top_cand_hyp_pos, len(self.vocab.tgt), rounding_mode='floor')
        hyp_word_ids = top_cand_hyp_pos % len(self.vocab.tgt)

        # 初始化新的候选序列、前序 ID 和得分
        new_hypotheses = []
        live_hyp_ids = []
        new_hyp_scores = []

        # 为每个候选序列更新其新生成的单词和对应的得分
        for prev_hyp_id, hyp_word_id, cand_new_hyp_score in zip(prev_hyp_ids, hyp_word_ids, top_cand_hyp_scores):
            prev_hyp_id = prev_hyp_id.item()
            hyp_word_id = hyp_word_id.item()
            cand_new_hyp_score = cand_new_hyp_score.item()

            # 从词 ID 中获取对应的单词
            hyp_word = self.vocab.tgt.id2word[hyp_word_id]
            new_hyp_sent = hypotheses[prev_hyp_id] + [hyp_word]  # 更新候选序列
            if hyp_word == '</s>':  # 如果生成了句子结束标记，保存该序列
                completed_hypotheses.append(Hypothesis(value=new_hyp_sent[1:-1], score=cand_new_hyp_score))
            else:  # 否则，继续扩展该候选序列
                new_hypotheses.append(new_hyp_sent)
                live_hyp_ids.append(prev_hyp_id)
                new_hyp_scores.append(cand_new_hyp_score)

        # 如果已找到足够的候选序列，提前结束
        if len(completed_hypotheses) == beam_size:
            break

        # 更新当前候选序列的状态，准备下一步解码
        live_hyp_ids = torch.tensor(live_hyp_ids, dtype=torch.long, device=self.device)
        h_tm1 = (h_t[live_hyp_ids], cell_t[live_hyp_ids])
        att_tm1 = att_t[live_hyp_ids]

        # 更新候选序列和其得分
        hypotheses = new_hypotheses
        hyp_scores = torch.tensor(new_hyp_scores, dtype=torch.float, device=self.device)

    # 如果没有完成的候选序列，返回当前最好的候选
    if len(completed_hypotheses) == 0:
        completed_hypotheses.append(Hypothesis(value=hypotheses[0][1:], score=hyp_scores[0].item()))

    # 将完成的候选序列按得分从高到低排序
    completed_hypotheses.sort(key=lambda hyp: hyp.score, reverse=True)

    # 返回排序后的候选序列列表
    return completed_hypotheses
