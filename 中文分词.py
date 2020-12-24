import re
import joblib

#数据处理：get_data()函数将段落数据划分为以“，。？！”标点符号结束的大量句子
def get_data(filepath,newpath):
    split_article_content = []
    for line in open(filepath, encoding='utf-8'):
        split_article_content += re.split("(?<=[。！？ ，]\s\s)", line)

    f = open(newpath, 'w')

    for i in range(len(split_article_content)):
        f.write(split_article_content[i]+'\n')
    f.close()
get_data("E:/icwb2-data/icwb2-data/training/pku_training.utf8", "D:/datasets/train.utf8")
get_data("E:/icwb2-data/icwb2-data/gold/msr_test_gold.utf8", "D:/datasets/verify.utf8")
get_data("E:/icwb2-data/icwb2-data/gold/pku_test_gold.utf8", "D:/datasets/test.utf8")

#标记字符状态：通过状态集合S：{S,B,M,E}，对每个词的字符状态进行标记
def get_word_status(word):
    '''
    S:单字词single
    B:词的开头Begin
    M:词的中间
    E:词的结尾

    '''
    word_status = []
    if len(word) == 1:
        word_status.append('S')
    elif len(word) == 2:
        word_status = ['B', 'E']
    else:
        mid_num = len(word) - 2
        mid_list = ['M'] * mid_num
        word_status.append('B')
        word_status.extend(mid_list)
        word_status.append('E')

    return word_status


# H M M模型：使用测试数据计算初始状态概率分布、状态转移概率分布及发射概率分布并保存到models目录下
# 得出模型的三个参数：初始概率+转移概率+发射概率
def train():
    train_filepath = 'D:/datasets/train.utf8'
    line_index = 0
    trans_dict = {}  # 存储状态转移概率
    emit_dict = {}  # 发射概率
    Count_dict = {}  # 存储所有状态在训练集的出现次数
    start_dict = {}  # 存储状态的初始概率
    state_list = ['B', 'M', 'E', 'S']  # 状态序列

    # 初始化各概率分布
    for state in state_list:
        trans_dict[state] = {}
        for state1 in state_list:
            trans_dict[state][state1] = 0.0

    for state in state_list:
        start_dict[state] = 0.0
        emit_dict[state] = {}
        Count_dict[state] = 0
    for line in open(train_filepath):
        line_index += 1

        line = line.strip()
        if not line:
            continue

        char_list = []
        for i in range(len(line)):
            if line[i] == " ":
                continue
            char_list.append(line[i])

        word_list = line.split("  ")
        line_status = []  # 统计状态序列

        for word in word_list:
            line_status.extend(get_word_status(word))
        if len(char_list) == len(line_status):  # 保证状态和字符对应
            for i in range(len(line_status)):
                if i == 0:
                    start_dict[line_status[0]] += 1  # 句子第一个字的状态
                    Count_dict[line_status[0]] += 1  # 记录每一个状态的出现次数
                else:  # 统计上一个状态到下一个状态，两个状态之间的转移概率
                    trans_dict[line_status[i - 1]][line_status[i]] += 1
                    Count_dict[line_status[i]] += 1
                    # 统计发射概率
                    if char_list[i] in emit_dict[line_status[i]]:
                        emit_dict[line_status[i]][char_list[i]] += 1
                    else:
                        emit_dict[line_status[i]][char_list[i]] = 0.0
        else:
            continue
    for key in start_dict:  # 状态的初始概率
        start_dict[key] = start_dict[key] * 1.0 / line_index
    for key in trans_dict:  # 状态转移概率
        for key1 in trans_dict[key]:
            trans_dict[key][key1] = trans_dict[key][key1] / Count_dict[key]
    for key in emit_dict:  # 发射概率
        for key1 in emit_dict[key]:
            emit_dict[key][key1] = emit_dict[key][key1] / Count_dict[key]
    joblib.dump(start_dict, "D:/models/prob_start.joblib")
    joblib.dump(trans_dict, "D:/models/prob_trans.joblib")
    joblib.dump(emit_dict, "D:/models/prob_emit.joblib")

#Viterbi算法:求出测试集的最优状态序列
def viterbi(obs, states, start_p, trans_p, emit_p):       #维特比算法（一种递归算法）
    L = [{}]
    path = {}
    for y in states:  #初始值
        L[0][y] = start_p[y] * (emit_p[y].get(obs[0], 0))    #在位置0，以y状态为末尾的状态序列的最大概率
        path[y] = [y]

    for t in range(1, len(obs)):
        L.append({})
        newpath = {}
        for y in states:    #从y0 -> y状态的递归
            state_path = ([(L[t - 1][y0] * trans_p[y0].get(y, 0) * emit_p[y].get(obs[t], 0), y0) for y0 in states if L[t - 1][y0] > 0])
            if state_path == []:
                (prob, state) = (0.0, 'S')
            else:
                (prob, state) = max(state_path)
            L[t][y] = prob
            newpath[y] = path[state] + [y]
        path = newpath  # 记录状态序列
    (prob, state) = max([(L[len(obs) - 1][y], y) for y in states])    #在最后一个位置，以y状态为末尾的状态序列的最大概率
    return (prob, path[state])   #返回概率和状态序列

#cut()函数能针对给出的句子得到分词结果
def cut(sent):
    prob_start = joblib.load("D:/models/prob_start.joblib")
    prob_trans = joblib.load("D:/models/prob_trans.joblib")
    prob_emit = joblib.load("D:/models/prob_emit.joblib")

    prob, pos_list = viterbi(sent, ('B', 'M', 'E', 'S'), prob_start, prob_trans, prob_emit)
    seglist = list()
    word = list()
    for index in range(len(pos_list)):
        if pos_list[index] == 'S':
            word.append(sent[index])
            seglist.append(word)
            word = []
        elif pos_list[index] in ['B', 'M']:
            word.append(sent[index])
        elif pos_list[index] == 'E':
            word.append(sent[index])
            seglist.append(word)
            word = []
    seglist = [''.join(tmp) for tmp in seglist]

    return seglist

#分词模型训练  调用train()函数，训练HMM模型并保存模型结果
train()

#分词模型评估

#验证集准确率
#通过caculate()计算P,R,F值衡量模型在验证集上的性能，并将分词结果保存到指定位置
#  P值表示预测为正的样本中有多少是真正的样本；
#  R表示的是样本中的正例有多少被预测正确；
#  F值用F1值来对Precision和Recall进行整体评价。
def caculate(testfile, resultfile):
    count_right = 0    #分词器正确标注单词数
    count_split = 0    #分词器标注单词总数(预测为正类数）
    count_gold = 0   #黄金分割单词数
    process_count = 0   #段落数
    fw = open(resultfile, 'w')
    with open(testfile) as f:
        for line in f:
            if line == "\n":
                continue
            process_count += 1
            line = line.strip()
            goldlist = line.split('  ')
            sentence = line.replace('  ', '')
            inlist = cut(sentence)
            for word in inlist:
                fw.write(word + '  ')
            fw.write('\n')

            count_split += len(inlist)
            count_gold += len(goldlist)
            tmp_in = inlist
            tmp_gold = goldlist
            for key in tmp_in:
                if key in tmp_gold:
                    count_right += 1
                    tmp_gold.remove(key)
            if process_count > 100:  # 避免运行时间过长
                break

        P = count_right / count_split  # P = TP/(TP+FP)
        R = count_right / count_gold  # R = TP/(TP+FN)
        F = 2 * P * R / (P + R)
        ER=(count_split - count_right)/count_gold
    fw.close()
    return P, R, F, ER
# P,R,F,ER = caculate('D:/datasets/verify.utf8', 'D:/result/verifyResult.txt')
# print(u'P: {} ,R: {}, F:{}, ER:{}'.format(P,R,F,ER))


# 在测试集上使用caculate()方法
P,R,F,ER = caculate('D:/datasets/test.utf8', 'D:/result/testResult.txt')
print(u'P: {} ,R: {}, F:{}, ER:{}'.format(P,R,F,ER))
with open('testResult.txt') as f:
    for i in range(10):
        print(f.readline())