#
# Copyright (C) 2015  Ilya Markov
#
# Full copyright notice can be found in LICENSE.
#

from __future__ import print_function

import sys
import time

from pyclick.click_models.Evaluation import LogLikelihood, Perplexity, PerplexityCond, CTRPrediction, RankingPerformance
from pyclick.click_models.UBM import UBM
from pyclick.click_models.DBN import DBN
from pyclick.click_models.SDBN import SDBN
from pyclick.click_models.DCM import DCM
from pyclick.click_models.CCM import CCM
from pyclick.click_models.CTR import DCTR, RCTR, GCTR
from pyclick.click_models.CM import CM
from pyclick.click_models.PBM import PBM
from pyclick.utils.Utils import Utils
from pyclick.utils.YandexRelPredChallengeParser import YandexRelPredChallengeParser


__author__ = 'Ilya Markov'


if __name__ == "__main__":
    print("===============================")
    print("This is an example of using PyClick for training and testing click models.")
    print("===============================")

    if len(sys.argv) < 4:
        print("USAGE: %s <click_model> <dataset> <sessions_max>" % sys.argv[0])
        print("\tclick_model - the name of a click model to use.")
        print("\tdataset - the path to the dataset from Yandex Relevance Prediction Challenge")
        print("\tsessions_max - the maximum number of one-query search sessions to consider")
        print("\tanswerset - the path to the answer dataset for true relevance value")
        print("\toptional: forgetting rate - forgetting rate for EM algorithm")
        print("")
        sys.exit(1)

    click_model = globals()[sys.argv[1]]()
    search_sessions_path = sys.argv[2]
    search_sessions_num = int(sys.argv[3])
    true_relevance_path = sys.argv[4]
    forget_rate= 0 if len(sys.argv) == 5 else float(sys.argv[5])

    true_rel = {}
    with open(true_relevance_path) as f:
        for line in f:
            (querykey , urlkey, val) = line.split("\t")
            true_rel[querykey]={}
            true_rel[querykey][urlkey]=int(val)

    search_sessions = YandexRelPredChallengeParser().parse(search_sessions_path, search_sessions_num)

    train_test_split = int(len(search_sessions) * 0.75)
    train_sessions = search_sessions[:train_test_split]
    train_queries = Utils.get_unique_queries(train_sessions)

    test_sessions = search_sessions[train_test_split:]
    test_queries = Utils.get_unique_queries(test_sessions)

    print("===============================")
    print("Training on %d search sessions (%d unique queries)." % (len(train_sessions), len(train_queries)))
    print("===============================")

    start = time.time()
    click_model.train(train_sessions,forget_rate)
    end = time.time()
    print("\tTrained %s click model in %i secs:\n%r" % (click_model.__class__.__name__, end - start, click_model))

    print("-------------------------------")
    print("Testing on %d search sessions (%d unique queries)." % (len(test_sessions), len(test_queries)))
    print("-------------------------------")

    loglikelihood = LogLikelihood()
    perplexity = Perplexity()
    perplexitycond = PerplexityCond()
    ctrprediction = CTRPrediction()
    


    start = time.time()
    ll_value = loglikelihood.evaluate(click_model, test_sessions)
    end = time.time()
    print("\tlog-likelihood: %f; time: %i secs" % (ll_value, end - start))

    start = time.time()
    perp_value = perplexity.evaluate(click_model, test_sessions)
    end = time.time()
    print('\tperplexity: {0}; perplexity@rank: {1};  time: {2} secs'.format(perp_value[0], perp_value[1:], end - start))

    start = time.time()
    perp_value = perplexitycond.evaluate(click_model, test_sessions)
    end = time.time()
    print('\tcl perplexity: {0}; perplexity@rank: {1};  time: {2} secs'.format(perp_value[0], perp_value[1:], end - start))

    start = time.time()
    rank = RankingPerformance(true_rel)
    ndcg_1 = rank.evaluate(click_model, test_sessions,1)
    ndcg_2 = rank.evaluate(click_model, test_sessions, 2)
    ndcg_5 = rank.evaluate(click_model, test_sessions, 5)
    ndcg_10 = rank.evaluate(click_model, test_sessions, 10)
    end = time.time()
    print('\tndcg@1: {0}; ndcg@2: {1}; ndcg@5: {2}; ndcg@10: {3}; time: {4} secs'.format(ndcg_1, ndcg_2, ndcg_5, ndcg_10, end - start))


