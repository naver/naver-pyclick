# naver-pyclick : Additional work on PyClick by NAVER.

naver-pyclick implemented several additional click models based on the Pyclick repository and added its own dataset CLARA 2.

## Additional Function on the Pyclick

### Forget Rate and NDCG Calculation

You can train the Click Model through the following command.
```
python examples/Example.py $CLICK_MODEL $CLICK_LOG_DIR $SESSION_NUM $ANSWER_DIR $REL_TYPE $FORGET_RATE
```

Here, each environment variable means the following.
- ```$CLICK_MODEL```: click model 
   - PBM (position-based model) 
   - UBM (user browsing model)
   - DBN (dynamic Bayesian network) 
   - SDBN (simplified DBN)
- ```$CLICK_LOG_DIR```: directory for click log data  
- ```$SESSION_NUM``` : number of search sessions to consider
- ```$ANSWER_DIR```: (Optional) directory for true relevance data; if this is omitted, then NDCG will not be shown.
- ```$REL_TYPE```: (Optional) relevance parameter for NDCG calculation
   - A: `attr` (Default)
   - AE: `attr` * `exam`
   - AS: `attr` * `sat`
   - AES: `attr` * `exam` * `sat`
- ```$FORGET_RATE```: (Optional) forgetting rate for EM algorithm (Default:0)



### CLARA 2

CLARA 2 is a  click-dataset created by Naver, and the configuration is as follows.
It can be found in the path below.

```
/examples/data/clara2/.
```

- CLARA2SearchLog : Main click log data set.
- CLARA2QueryURLRelevance : relevance between query and url
- CLARA2QueryCount : query count

Currently, CLARA2 has only distributed beta versions and plans to release an official version later.


### Third-party SW

This repository was created based on the software below.

https://github.com/markovi/PyClick




