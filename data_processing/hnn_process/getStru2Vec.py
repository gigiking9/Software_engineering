import pickle
from multiprocessing.pool import ThreadPool


# 多进程解析Python数据
def multipro_python_context(data):
    acont1_cut = [i[0][0] for i in data]
    return acont1_cut


def multipro_python_code(data):
    code_cut = [i[1][0] for i in data]
    return code_cut


def multipro_python_query(data):
    query_cut = [i[2][0] for i in data]
    return query_cut


def parse_python(python_list, split_num):
    # 解析acont1数据
    acont1_data = [i[0][0] for i in python_list]
    acont1_split_list = [acont1_data[i:i + split_num] for i in range(0, len(acont1_data), split_num)]
    pool = ThreadPool(10)
    acont1_list = pool.map(multipro_python_context, acont1_split_list)
    pool.close()
    pool.join()
    acont1_cut = [item for sublist in acont1_list for item in sublist]
    print(f'acont1条数：{len(acont1_cut)}')

    # 解析code数据
    code_data = [i[1][0] for i in python_list]
    code_split_list = [code_data[i:i + split_num] for i in range(0, len(code_data), split_num)]
    pool = ThreadPool(10)
    code_list = pool.map(multipro_python_code, code_split_list)
    pool.close()
    pool.join()
    code_cut = [item for sublist in code_list for item in sublist]
    print(f'code条数：{len(code_cut)}')

    # 解析query数据
    query_data = [i[2][0] for i in python_list]
    query_split_list = [query_data[i:i + split_num] for i in range(0, len(query_data), split_num)]
    pool = ThreadPool(10)
    query_list = pool.map(multipro_python_query, query_split_list)
    pool.close()
    pool.join()
    query_cut = [item for sublist in query_list for item in sublist]
    print(f'query条数：{len(query_cut)}')

    # 获取qids
    qids = [i[0] for i in python_list]

    return acont1_cut, code_cut, query_cut, qids


# 多进程解析SQL数据
def multipro_sqlang_context(data):
    acont1_cut = [i[1][0][0] for i in data]
    return acont1_cut


def multipro_sqlang_code(data):
    code_cut = [i[2][0][0] for i in data]
    return code_cut


def multipro_sqlang_query(data):
    query_cut = [i[3][0] for i in data]
    return query_cut


def parse_sqlang(sqlang_list, split_num):
    # 解析acont1数据
    acont1_data = [i[1][0][0] for i in sqlang_list]
    acont1_split_list = [acont1_data[i:i + split_num] for i in range(0, len(acont1_data), split_num)]
    pool = ThreadPool(10)
    acont1_list = pool.map(multipro_sqlang_context, acont1_split_list)
    pool.close()
    pool.join()
    acont1_cut = [item for sublist in acont1_list for item in sublist]
    print(f'acont1条数：{len(acont1_cut)}')

    # 解析code数据
    code_data = [i[2][0][0] for i in sqlang_list]
    code_split_list = [code_data[i:i + split_num] for i in range(0, len(code_data), split_num)]
    pool = ThreadPool(10)
    code_list = pool.map(multipro_sqlang_code, code_split_list)
    pool.close()
    pool.join()
    code_cut = [item for sublist in code_list for item in sublist]
    print(f'code条数：{len(code_cut)}')

    # 解析query数据
    query_data = [i[3][0] for i in sqlang_list]
    query_split_list = [query_data[i:i + split_num] for i in range(0, len(query_data), split_num)]
    pool = ThreadPool(10)
    query_list = pool.map(multipro_sqlang_query, query_split_list)
    pool.close()
    pool.join()
    query_cut = [item for sublist in query_list for item in sublist]
    print(f'query条数：{len(query_cut)}')

    # 获取qids
    qids = [i[0] for i in sqlang_list]

    return acont1_cut, code_cut, query_cut, qids


def main(lang_type, split_num, input_path, save_path):
    with open(input_path, "rb") as f:
        total_data = pickle.load(f)

    if lang_type == 'python':
        acont1_cut, code_cut, query_cut, qids = parse_python(total_data, split_num)
    elif lang_type == 'sql':
        acont1_cut, code_cut, query_cut, qids = parse_sqlang(total_data, split_num)
    else:
        print("Unsupported language type.")
        return

    # 将解析后的数据保存到文件
    with open(save_path, "w") as f:
        f.write(str(total_data))


if __name__ == '__main__':
    # 解析Python数据
    staqc_python_path = '../hnn_process/ulabel_data/python_staqc_qid2index_blocks_unlabeled.txt'
    staqc_python_save = '../hnn_process/ulabel_data/staqc/python_staqc_unlabeled_data.txt'
    main('python', 1000, staqc_python_path, staqc_python_save)

    # 解析SQL数据
    staqc_sql_path = '../hnn_process/ulabel_data/sql_staqc_qid2index_blocks_unlabeled.txt'
    staqc_sql_save = '../hnn_process/ulabel_data/staqc/sql_staqc_unlabeled_data.txt'
    main('sql', 1000, staqc_sql_path, staqc_sql_save)

    # 解析大型Python数据
    large_python_path = '../hnn_process/ulabel_data/large_corpus/multiple/python_large_multiple.pickle'
    large_python_save = '../hnn_process/ulabel_data/large_corpus/multiple/python_large_multiple_unlabeled.txt'
    main('python', 1000, large_python_path, large_python_save)

    # 解析大型SQL数据
    large_sql_path = '../hnn_process/ulabel_data/large_corpus/multiple/sql_large_multiple.pickle'
    large_sql_save = '../hnn_process/ulabel_data/large_corpus/multiple/sql_large_multiple_unlabeled.txt'
    main('sql', 1000, large_sql_path, large_sql_save)
