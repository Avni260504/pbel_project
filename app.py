from ibm_watsonx_ai.helpers import DataConnection
from ibm_watsonx_ai.helpers import ContainerLocation

training_data_references = [
    DataConnection(
        data_asset_id='b679e740-5044-47bc-b89f-d044bd3da790'
    ),
]

training_result_reference = DataConnection(
    location=ContainerLocation(
        path='auto_ml/.../data/automl',
        model_location='auto_ml/.../data/automl/model.zip',
        training_status='auto_ml/.../training-status.json'
    )
)

experiment_metadata = dict(
    prediction_type='regression',
    prediction_column='age',
    holdout_size=0.1,
    scoring='neg_root_mean_squared_error',
    csv_separator=',',
    random_state=33,
    max_number_of_estimators=2,
    training_data_references=training_data_references,
    training_result_reference=training_result_reference
)

# Likely follows with code to run experiment and deploy model
