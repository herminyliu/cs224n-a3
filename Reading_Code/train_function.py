def train(args: Dict):
    """ 训练NMT模型。
    @param args (Dict): 来自命令行的参数
    """

    # 从文件中读取源语言训练数据，限制词汇表大小为21000
    train_data_src = read_corpus(args['--train-src'], source='src', vocab_size=21000)  # 编辑: 设置新的词汇表大小

    # 从文件中读取目标语言训练数据，限制词汇表大小为8000
    train_data_tgt = read_corpus(args['--train-tgt'], source='tgt', vocab_size=8000)

    # 从文件中读取源语言开发集数据，限制词汇表大小为3000
    dev_data_src = read_corpus(args['--dev-src'], source='src', vocab_size=3000)

    # 从文件中读取目标语言开发集数据，限制词汇表大小为2000
    dev_data_tgt = read_corpus(args['--dev-tgt'], source='tgt', vocab_size=2000)

    # 选取开发集的第三个例子作为示例句子
    example_sentence_src = dev_data_src[3]
    example_sentence_tgt = dev_data_tgt[3]

    # 将源语言和目标语言的数据配对
    train_data = list(zip(train_data_src, train_data_tgt))
    dev_data = list(zip(dev_data_src, dev_data_tgt))

    # 从命令行参数中获取批量大小
    train_batch_size = int(args['--batch-size'])

    # 从命令行参数中获取梯度裁剪阈值
    clip_grad = float(args['--clip-grad'])

    # 从命令行参数中获取验证的间隔迭代次数
    valid_niter = int(args['--valid-niter'])

    # 从命令行参数中获取日志打印间隔迭代次数
    log_every = int(args['--log-every'])

    # 模型保存路径
    model_save_path = args['--save-to']

    # 从文件加载词汇表
    vocab = Vocab.load(args['--vocab'])

    # 创建NMT模型实例
    model = NMT(embed_size=1024, hidden_size=768, dropout_rate=float(args['--dropout']), vocab=vocab)

    # 使用TensorBoard记录训练过程
    tensorboard_path = "nmt" if args['--cuda'] else "nmt_local"
    writer = SummaryWriter(log_dir=f"./runs/{tensorboard_path}")
    model.train()  # 设置模型为训练模式

    # 如果指定了初始化范围，按范围进行参数均匀初始化
    uniform_init = float(args['--uniform-init'])
    if np.abs(uniform_init) > 0.:
        print('均匀初始化参数 [-%f, +%f]' % (uniform_init, uniform_init), file=sys.stderr)
        for p in model.parameters():
            p.data.uniform_(-uniform_init, uniform_init)

    # 生成目标语言词汇表的掩码，并将<pad>符号位置设置为0
    vocab_mask = torch.ones(len(vocab.tgt))
    vocab_mask[vocab.tgt['<pad>']] = 0

    # 设置设备为cuda或cpu
    device = torch.device("cuda:0" if args['--cuda'] else "cpu")
    print('使用设备: %s' % device, file=sys.stderr)

    # 将模型移动到设备上
    model = model.to(device)

    # 使用Adam优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=float(args['--lr']))

    # 初始化相关变量
    num_trial = 0
    train_iter = patience = cum_loss = report_loss = cum_tgt_words = report_tgt_words = 0
    cum_examples = report_examples = epoch = valid_num = 0
    hist_valid_scores = []
    train_time = begin_time = time.time()
    print('开始最大似然估计训练')

    # 进入训练循环
    while True:
        epoch += 1

        # 批量迭代源和目标句子
        for src_sents, tgt_sents in batch_iter(train_data, batch_size=train_batch_size, shuffle=True):
            train_iter += 1

            optimizer.zero_grad()  # 清空优化器的梯度

            batch_size = len(src_sents)

            # 计算每个样本的损失
            example_losses = -model(src_sents, tgt_sents)  # (batch_size,)

            # 计算批量损失的和
            batch_loss = example_losses.sum()

            # 计算平均损失
            loss = batch_loss / batch_size

            # 反向传播计算梯度
            loss.backward()

            # 梯度裁剪
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)

            # 更新模型参数
            optimizer.step()

            # 记录损失值
            batch_losses_val = batch_loss.item()
            report_loss += batch_losses_val
            cum_loss += batch_losses_val

            # 计算需要预测的目标词数，忽略前导的<s>
            tgt_words_num_to_predict = sum(len(s[1:]) for s in tgt_sents)
            report_tgt_words += tgt_words_num_to_predict
            cum_tgt_words += tgt_words_num_to_predict
            report_examples += batch_size
            cum_examples += batch_size

            # 每隔log_every次迭代打印一次日志
            if train_iter % log_every == 0:
                writer.add_scalar("loss/train", report_loss / report_tgt_words, train_iter)
                writer.add_scalar("perplexity/train", math.exp(report_loss / report_tgt_words), train_iter)
                print('epoch %d, iter %d, avg. loss %.2f, avg. ppl %.2f ' \
                      '累计样本数 %d, 速度 %.2f 词/秒, 已经过时间 %.2f 秒' % (epoch, train_iter,
                                                                              report_loss / report_tgt_words,
                                                                              math.exp(report_loss / report_tgt_words),
                                                                              cum_examples,
                                                                              report_tgt_words / (
                                                                                          time.time() - train_time),
                                                                              time.time() - begin_time),
                      file=sys.stderr)

                train_time = time.time()
                report_loss = report_tgt_words = report_examples = 0.

            # 每隔valid_niter次迭代进行验证
            if train_iter % valid_niter == 0:
                writer.add_scalar("loss/val", cum_loss / cum_tgt_words, train_iter)
                print('epoch %d, iter %d, 累计损失 %.2f, 累计困惑度 %.2f 累计样本数 %d' % (epoch, train_iter,
                                                                                           cum_loss / cum_tgt_words,
                                                                                           np.exp(
                                                                                               cum_loss / cum_tgt_words),
                                                                                           cum_examples),
                      file=sys.stderr)

                cum_loss = cum_examples = cum_tgt_words = 0.
                valid_num += 1

                print('开始验证 ...', file=sys.stderr)

                # 计算开发集的困惑度
                dev_ppl = evaluate_ppl(model, dev_data, batch_size=128)  # 开发集批量大小可以稍大

                # 使用beam search进行示例句子的解码
                example_hypothesis_beam = beam_search(
                    model, [example_sentence_src],
                    beam_size=10,
                    max_decoding_time_step=int(args['--max-decoding-time-step'])
                )[0]

                # 记录验证集困惑度
                valid_metric = -dev_ppl
                writer.add_scalar("perplexity/val", dev_ppl, train_iter)
                writer.add_text(
                    f'example_translation_with_beam_search',
                    format_example_sentence(example_sentence_src, example_sentence_tgt, example_hypothesis_beam,
                                            train_iter),
                    train_iter
                )
                print('验证: iter %d, 开发集困惑度 %f' % (train_iter, dev_ppl), file=sys.stderr)

                # 检查是否是最佳模型
                is_better = len(hist_valid_scores) == 0 or valid_metric > max(hist_valid_scores)
                hist_valid_scores.append(valid_metric)

                if is_better:
                    patience = 0
                    print('保存当前最好的模型到 [%s]' % model_save_path, file=sys.stderr)
                    model.save(model_save_path)

                    # 同时保存优化器的状态
                    torch.save(optimizer.state_dict(), model_save_path + '.optim')
                elif patience < int(args['--patience']):
                    patience += 1
                    print('耐心计数 %d' % patience, file=sys.stderr)

                    if patience == int(args['--patience']):
                        num_trial += 1
                        print('达到第 %d 次尝试' % num_trial, file=sys.stderr)
                        if num_trial == int(args['--max-num-trial']):
                            print('提前停止!', file=sys.stderr)
                            exit(0)

                        # 学习率衰减，并从之前的最佳模型恢复
                        lr = optimizer.param_groups[0]['lr'] * float(args['--lr-decay'])
                        print('加载之前的最佳模型并衰减学习率到 %f' % lr, file=sys.stderr)

                        # 加载模型
                        params = torch.load(model_save_path, map_location=lambda storage, loc: storage)
                        model.load_state_dict(params)

                        print('恢复模型参数', file=sys.stderr)

                        # 恢复优化器状态并设置新的学习率
                        optimizer.load_state_dict(torch.load(model_save_path + '.optim'))
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = lr

                        # 重置训练状态
                        patience = 0
