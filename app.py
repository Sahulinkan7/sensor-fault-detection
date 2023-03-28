from flask import Flask,render_template
from sensor.pipeline.training_pipeline import TrainPipeline

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
    pipeline.run_pipeline()