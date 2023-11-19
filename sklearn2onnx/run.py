"""sklearn to onnx conversion demo"""

import numpy as np
import onnxruntime as rt
from onnxmltools.convert.xgboost.operator_converters.XGBoost import convert_xgboost
from skl2onnx import convert_sklearn, update_registered_converter
from skl2onnx.common.data_types import FloatTensorType
from skl2onnx.common.shape_calculator import calculate_linear_classifier_output_shapes
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier


def load_data(num_feats):
    data = load_iris()
    X = data.data[:, :num_feats]
    y = data.target

    ind = np.arange(X.shape[0])
    np.random.shuffle(ind)
    X = X[ind, :].copy()
    y = y[ind].copy()

    return X, y


def load_model(model_name):
    return {
        'xgb': lambda: Pipeline(
            [('scaler', StandardScaler()), (model_name, XGBClassifier(n_estimators=3))]
        ),  # 可以用任何函数代替
        'rf': lambda: Pipeline([('scaler', StandardScaler()), (model_name, RandomForestClassifier(n_estimators=3))]),
    }.get(model_name, lambda: None)()


def sklearn2onnx(model_name, num_feats):
    pipe = load_model(model_name)
    X, y = load_data(num_feats)

    pipe.fit(X, y)

    if model_name == 'xgb':  # Register the converter for XGBClassifier
        update_registered_converter(
            model=XGBClassifier,
            alias='XGBoostXGBClassifier',
            shape_fct=calculate_linear_classifier_output_shapes,
            convert_fct=convert_xgboost,
            options={'nocl': [True, False], 'zipmap': [True, False, 'columns']},
        )

    # convert

    # ai.onnx.ml: 内置机器学习操作符集
    # initial_types:
    # None 表示 batch_size 未知
    # num_feats表示输入的前num_feats个元素是 FloatTensorType 这个类型 (输入特征维度)
    model_onnx = convert_sklearn(
        model=pipe,
        name=f'pipeline_{model_name}',
        initial_types=[('input', FloatTensorType([None, num_feats]))],
        target_opset={'': 12, 'ai.onnx.ml': 2},
    )

    # save converted model
    with open(f'../model/pipeline_{model_name}.onnx', 'wb') as f:
        f.write(model_onnx.SerializeToString())

    # compare the predictions
    print('predict', pipe.predict(X[:5]))
    print('predict_proba', pipe.predict_proba(X[:1]))

    # initialize onnx session
    sess = rt.InferenceSession(f'../model/pipeline_{model_name}.onnx', providers=['CPUExecutionProvider'])

    # 模型有1个输入节点，通过 netron 可以看到
    input_name = sess.get_inputs()[0].name

    # 模型有2个输出节点，通过 netron 可以看到
    label_name = sess.get_outputs()[0].name
    probability_name = sess.get_outputs()[1].name

    # run inference
    pred_onx = sess.run(output_names=[label_name, probability_name], input_feed={'input': X[:5].astype(np.float32)})

    print(f'input_name: {input_name}')
    print(f'label_name: {label_name}')
    print(f'probability_name: {probability_name}')
    print('predict', pred_onx[0])
    print('predict_proba', pred_onx[1][:1])

    # save data
    X[:1].tofile('../input/array_data.bin')


if __name__ == '__main__':
    model_name = 'rf'
    num_feats = 4

    sklearn2onnx(model_name, num_feats)
