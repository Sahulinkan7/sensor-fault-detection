from flask import Flask,render_template,request,send_from_directory
from sensor.pipeline.training_pipeline import TrainPipeline
from sensor.ml.model.estimator import ModelResolver,TargetValueMapping
from sensor.constant.training_pipeline import SAVED_MODEL_DIR
from sensor.utils.main_utils import load_object
from sensor.logger import logging
import pandas as pd
from datetime import datetime

from sensor.utils.main_utils import read_yaml_file
import os

PRED_DIR="predictions"
app=Flask(__name__)

@app.route('/',methods=['GET','POST'])
def index():
    try:
        return render_template('index.html')
    except Exception as e:
        return str(e)
    
@app.route('/train',methods=['GET','POST'])
def train():
    pipeline = TrainPipeline()
    if request.method=='POST':
        if not pipeline.experiment.running_status:
            message="Training Started"
            pipeline.start()
        else:
            message = "Training is already in progress "
        return render_template('train.html',msg=message)
    return render_template('train.html')
    
@app.route('/prediction',methods=['GET','POST'])
def predict():
    if request.method == 'POST':
        file=request.files['upload_file']
        df=pd.read_csv(file)
        model_resolver = ModelResolver(model_dir=SAVED_MODEL_DIR)
        if not model_resolver.is_model_exists():
            message = f"No model is present to predict"
            return render_template('predictions.html',message = message)
        best_model_path = model_resolver.get_best_model_path()
        model = load_object(file_path=best_model_path)
        y_pred=model.predict(df)
        df['predict_col']=y_pred
        df['predict_col'].replace(TargetValueMapping().reverse_mapping(),inplace=True)
        os.makedirs(PRED_DIR,exist_ok=True)
        file_name = f"prediction_{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.csv"
        file_path=os.path.join(PRED_DIR,file_name)
        df.to_csv(file_path)
        return send_from_directory(PRED_DIR,file_name)
    return render_template('predictions.html')


@app.route('/experiments',methods=['GET','POST'])
def experiments():
    try:
        pipeline = TrainPipeline()
        df = pipeline.get_experiments_status()
        if df is None:
            return render_template('experiments.html')
        context = {
            'experiments':df.to_html(classes='table table-stripped col-11',index=False)
        }        
        return render_template('experiments.html',context=context)
    except Exception as e:
        return str(e)
    
    

env_file_path=os.path.join(os.getcwd(),"env.yaml")

def set_env_variable(env_file_path):
    if os.getenv('MONGO_DB_URL',None) is None:
        env_config = read_yaml_file(file_path=env_file_path)
        os.environ['MONGO_DB_URL']=env_config['MONGO_DB_URL']
        

if __name__ == '__main__':
    try:
        set_env_variable(env_file_path)
        app.run(debug=True)
    except Exception as e:
        logging.info(f"{e}")
        print(str(e))
