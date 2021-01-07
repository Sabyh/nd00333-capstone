# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------
import json
import logging
import os
import pickle
import numpy as np
import pandas as pd
import joblib

import azureml.automl.core
from azureml.automl.core.shared import logging_utilities, log_server
from azureml.telemetry import INSTRUMENTATION_KEY

from inference_schema.schema_decorators import input_schema, output_schema
from inference_schema.parameter_types.numpy_parameter_type import NumpyParameterType
from inference_schema.parameter_types.pandas_parameter_type import PandasParameterType


input_sample = pd.DataFrame({"Column1": pd.Series([0.0], dtype="float64"), "0": pd.Series([0.0], dtype="float64"), "1": pd.Series([0.0], dtype="float64"), "2": pd.Series([0.0], dtype="float64"), "3": pd.Series([0.0], dtype="float64"), "4": pd.Series([0.0], dtype="float64"), "5": pd.Series([0.0], dtype="float64"), "6": pd.Series([0.0], dtype="float64"), "7": pd.Series([0.0], dtype="float64"), "8": pd.Series([0.0], dtype="float64"), "9": pd.Series([0.0], dtype="float64"), "10": pd.Series([0.0], dtype="float64"), "11": pd.Series([0.0], dtype="float64"), "12": pd.Series([0.0], dtype="float64"), "13": pd.Series([0.0], dtype="float64"), "14": pd.Series([0.0], dtype="float64"), "15": pd.Series([0.0], dtype="float64"), "16": pd.Series([0.0], dtype="float64"), "17": pd.Series([0.0], dtype="float64"), "18": pd.Series([0.0], dtype="float64"), "19": pd.Series([0.0], dtype="float64"), "20": pd.Series([0.0], dtype="float64"), "21": pd.Series([0.0], dtype="float64"), "22": pd.Series([0.0], dtype="float64"), "23": pd.Series([0.0], dtype="float64"), "24": pd.Series([0.0], dtype="float64"), "25": pd.Series([0.0], dtype="float64"), "26": pd.Series([0.0], dtype="float64"), "27": pd.Series([0.0], dtype="float64"), "28": pd.Series([0.0], dtype="float64"), "29": pd.Series([0.0], dtype="float64"), "30": pd.Series([0.0], dtype="float64"), "31": pd.Series([0.0], dtype="float64"), "32": pd.Series([0.0], dtype="float64"), "33": pd.Series([0.0], dtype="float64"), "34": pd.Series([0.0], dtype="float64"), "35": pd.Series([0.0], dtype="float64"), "36": pd.Series([0.0], dtype="float64"), "37": pd.Series([0.0], dtype="float64"), "38": pd.Series([0.0], dtype="float64"), "39": pd.Series([0.0], dtype="float64"), "40": pd.Series([0.0], dtype="float64"), "41": pd.Series([0.0], dtype="float64"), "42": pd.Series([0.0], dtype="float64"), "43": pd.Series([0.0], dtype="float64"), "44": pd.Series([0.0], dtype="float64"), "45": pd.Series([0.0], dtype="float64"), "46": pd.Series([0.0], dtype="float64"), "47": pd.Series([0.0], dtype="float64"), "48": pd.Series([0.0], dtype="float64"), "49": pd.Series([0.0], dtype="float64"), "50": pd.Series([0.0], dtype="float64"), "51": pd.Series([0.0], dtype="float64"), "52": pd.Series([0.0], dtype="float64"), "53": pd.Series([0.0], dtype="float64"), "54": pd.Series([0.0], dtype="float64"), "55": pd.Series([0.0], dtype="float64"), "56": pd.Series([0.0], dtype="float64"), "57": pd.Series([0.0], dtype="float64"), "58": pd.Series([0.0], dtype="float64"), "59": pd.Series([0.0], dtype="float64"), "60": pd.Series([0.0], dtype="float64"), "61": pd.Series([0.0], dtype="float64"), "62": pd.Series([0.0], dtype="float64"), "63": pd.Series([0.0], dtype="float64")})
output_sample = np.array([0])
try:
    log_server.enable_telemetry(INSTRUMENTATION_KEY)
    log_server.set_verbosity('INFO')
    logger = logging.getLogger('azureml.automl.core.scoring_script')
except:
    pass


def init():
    global model
    # This name is model.id of model that we want to deploy deserialize the model file back
    # into a sklearn model
    model_path = os.path.join(os.getenv('AZUREML_MODEL_DIR'), 'model.pkl')
    path = os.path.normpath(model_path)
    path_split = path.split(os.sep)
    log_server.update_custom_dimensions({'model_name': path_split[1], 'model_version': path_split[2]})
    try:
        logger.info("Loading model from path.")
        model = joblib.load(model_path)
        logger.info("Loading successful.")
    except Exception as e:
        logging_utilities.log_traceback(e, logger)
        raise


@input_schema('data', PandasParameterType(input_sample))
@output_schema(NumpyParameterType(output_sample))
def run(data):
    try:
        result = model.predict(data)
        return json.dumps({"result": result.tolist()})
    except Exception as e:
        result = str(e)
        return json.dumps({"error": result})
