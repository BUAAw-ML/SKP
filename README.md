# Knowledge-enhanced Multi-modal Model

This project builds upon [LAVIS](https://github.com/salesforce/LAVIS) library.


## Installation

```bash
cd LAVIS
pip install -e .
```

##  Improve External Knowledge Utilization

The main idea focuses on making better use of external knowledge to instruct MLLM to output more accurate generation. The detail can refer to [`lavis\models\blip2_models\blip2_vicuna_instruct_okvqa.py`](lavis\models\blip2_models\blip2_vicuna_instruct_okvqa.py) 

### Quick Start

```python
bash run_scripts/blip2/train/train_okvqa.sh
```

## Notes

This publication version was made in a rush due to intensive workload that the author currently have. We will add follow-up patches to make codes more readible and ensure reproducibility. (of course, the speed depends on the number of people who are interested in using this framework.)



## License
[BSD 3-Clause License](LICENSE.txt)

