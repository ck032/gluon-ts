# _*_ coding:utf-8 _*_

# @Author      : chenkai<chenkai15@geely.com>
# @Created Time: 2020/7/30 上午10:29
# @File        : 均值预测.py


from gluonts.model.trivial.mean import MeanPredictor

context_length = 5
prediction_length = 6
num_samples = 4
freq = '1min'

hyperparameters = {
    "context_length": context_length,
    "prediction_length": prediction_length,
    "num_samples": num_samples,
    "freq": freq,
    "num_workers": 3,
    "num_prefetch": 4,
    "shuffle_buffer_length": 256,
    "epochs": 3,
}

predictor = MeanPredictor.from_hyperparameters(**hyperparameters)
print(predictor)
