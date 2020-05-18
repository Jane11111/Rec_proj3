# -*- coding: utf-8 -*-
# @Time    : 2020-05-15 22:52
# @Author  : zxl
# @FileName: test.py

import configargparse
# CONFIG_FILE='model.yml'
def preclean_opt(parse):
    group = parse.add_argument_group("Preclean")
    group.add("--vocab_size", "-vocab_size", type=int)
    group.add("--embedding_size", "-embedding_size", type=int)
    group.add("--feature_size", "-feature_size", type=int)
    group.add("--window_size", "-window_size", type=int)
    group.add("--max_text_len", "-max_text_len", type=int)
    group.add("--xu", "-xu", type=int)
    group.add("--yi", "-yi", type=int)
    group.add("--k", "-k", type=int)
    group.add("--batch_size", "-batch_size", type=int)
    group.add("--epoch", "-epoch", type=int)
    group.add("--learning_rate", "-learning_rate", type=float)
    group.add("--embedding_path", "-embedding_path", type=str)
    group.add("--save_path", "-save_path", type=str)
    return parse


def get_config(path):
    parse = configargparse.ArgumentParser(default_config_files=[path],
                                              config_file_parser_class=configargparse.YAMLConfigFileParser)
    parse = preclean_opt(parse)
    config,unknown = parse.parse_known_args()
    # print(type(config.vocab_size))
    return config