# from flask import Flask,render_template,request,jsonify
# from src.pipeline.prediction_pipeline import PredictionPipelineConfig,Customclass


# app = Flask(__name__)

# @app.route('/',methods=['GET','POST'])
# def prediction_data():
#     if request.method == "GET":
#         return render_template("home.html")
#     else:
#         data = Customclass(
#             age=int(request.form.get('age')),
#             workclass=int(request.form.get('workclass')),
#             education_num=int(request.form.get('education_num')),
#             marital_status=int(request.form.get('marital_status')),
#             occupation=int(request.form.get('occupation')),
#             relationship=int(request.form.get('relationship')),
#             race=int(request.form.get('race')),
#             sex=int(request.form.get('sex')),
#             native_country=int(request.form.get('native_country')),
#             capital_gain=int(request.form.get('capital_gain')),
#             capital_loss=int(request.form.get('capital_loss')),
#             hours_per_week=int(request.form.get('hours_per_week'))
#         )
        
#         final_data = data.get_data_as_dataframe()
#         pipeline_prediction = PredictionPipelineConfig()
#         prediction = pipeline_prediction.predict(final_data)
        
#         result = prediction
        
#         if result == 0:
#             return render_template("result.html",final_result="Your yearly income is less than equal to 50k:{}".format(result))
#         elif result == 1:
#             return render_template("result.html",final_result="Your yearly income is greater than 50k:{}".format(result))
        
#         if __name__== "__main__":
#             app.run(host="0.0.0.0",debug=True)


# from flask import Flask, render_template, request, jsonify
# from src.pipeline.prediction_pipeline import PredictionPipelineConfig, Customclass

# app = Flask(__name__)

# @app.route('/', methods=['GET', 'POST'])
# def prediction_data():
#     if request.method == "GET":
#         return render_template("home.html")
#     else:
#         data = Customclass(
#             age=int(request.form.get('age')),
#             workclass=int(request.form.get('workclass')),
#             education_num=int(request.form.get('education_num')),
#             marital_status=int(request.form.get('marital_status')),
#             occupation=int(request.form.get('occupation')),
#             relationship=int(request.form.get('relationship')),
#             race=int(request.form.get('race')),
#             sex=int(request.form.get('sex')),
#             native_country=int(request.form.get('native_country')),
#             capital_gain=int(request.form.get('capital_gain')),
#             capital_loss=int(request.form.get('capital_loss')),
#             hours_per_week=int(request.form.get('hours_per_week'))
#         )
        
#         final_data = data.get_data_as_dataframe()
#         pipeline_prediction = PredictionPipelineConfig()
#         prediction = pipeline_prediction.predict(final_data)
        
#         result = prediction
        
#         if result == 0:
#             return render_template("result.html", final_result="Your yearly income is less than or equal to 50k: {}".format(result))
#         elif result == 1:
#             return render_template("result.html", final_result="Your yearly income is greater than 50k: {}".format(result))

# if __name__ == "__main__":
#     app.run(host="0.0.0.0",port=5001, debug=True)

from flask import Flask, render_template, request, jsonify
from src.pipeline.prediction_pipeline import PredictionPipelineConfig, Customclass

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def prediction_data():
    if request.method == "GET":
        return render_template("home.html")
    else:
        data = Customclass(
            age=int(request.form.get('age')),
            workclass=int(request.form.get('workclass')),
            education_num=int(request.form.get('education_num')),
            marital_status=int(request.form.get('marital_status')),
            occupation=int(request.form.get('occupation')),
            relationship=int(request.form.get('relationship')),
            race=int(request.form.get('race')),
            sex=int(request.form.get('sex')),
            native_country=int(request.form.get('native_country')),
            capital_gain=int(request.form.get('capital_gain')),
            capital_loss=int(request.form.get('capital_loss')),
            hours_per_week=int(request.form.get('hours_per_week'))
        )
        
        final_data = data.get_data_as_dataframe()
        pipeline_prediction = PredictionPipelineConfig()
        pred = pipeline_prediction.predict(final_data)
        
        result = pred                       
        
        if result == 0:
            return render_template("result.html", final_result="Your yearly income is less than or equal to 50k: {}".format(result))
        elif result == 1:
            return render_template("result.html", final_result="Your yearly income is greater than 50k: {}".format(result))

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)


    





