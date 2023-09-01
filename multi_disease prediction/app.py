import csv
from flask import Flask, render_template,request,redirect,url_for,jsonify
import diseaseprediction
from pyperclip import paste
from webbrowser import open as op
import pandas as pd


app = Flask(__name__)

selected_location=""


doc_list = []
with open('templates/Testing.csv', newline='') as f:
        reader = csv.reader(f)
        symptoms = next(reader)
        symptoms = symptoms[:len(symptoms)-1]

def disease_name():
    with open('dis_pred.txt', newline='') as f:
            predicted_dis = f.read()
    return (predicted_dis)

def get_doc_location():
    df_doc = pd.read_excel("templates/Doctor-DATA_new.xlsx")
    predicted_dis = disease_name()
    df_doc_data = df_doc.loc[(df_doc['Disease'] == predicted_dis)]
    location = df_doc_data["Location"].tolist()
    
    return (location)

def get_doc_list(loca):
    df_doc = pd.read_excel("templates/Doctor-DATA_new.xlsx")
    predicted_dis = disease_name()
    
    df_doc_data = df_doc.loc[(df_doc['Disease'] == predicted_dis) & (df_doc['Location'] == loca)]
    doc_list = df_doc_data["DocName"].tolist()
    
    return(doc_list)

def get_doc_address(doc_name):
    df_doc = pd.read_excel("templates/Doctor-DATA_new.xlsx")
    df_doc_data = df_doc.loc[(df_doc['DocName'] == str(doc_name))]
    #address = df_doc_data["Address"].tolist()
    lat = df_doc_data["Lat"].tolist() 
    lon = df_doc_data["Lon"].tolist() 
    return(lat[0],lon[0])

@app.route('/', methods=['GET'])
def dropdown():
        # return render_template('includes/default.html', symptoms=symptoms)
        return render_template('login.html', symptoms=symptoms)

@app.route('/disease_predict', methods=['POST'])
def disease_predict():
    selected_symptoms = []
    if(request.form['Symptom1']!="") and (request.form['Symptom1'] not in selected_symptoms):
        selected_symptoms.append(request.form['Symptom1'])
    if(request.form['Symptom2']!="") and (request.form['Symptom2'] not in selected_symptoms):
        selected_symptoms.append(request.form['Symptom2'])
    if(request.form['Symptom3']!="") and (request.form['Symptom3'] not in selected_symptoms):
        selected_symptoms.append(request.form['Symptom3'])
    if(request.form['Symptom4']!="") and (request.form['Symptom4'] not in selected_symptoms):
        selected_symptoms.append(request.form['Symptom4'])
    if(request.form['Symptom5']!="") and (request.form['Symptom5'] not in selected_symptoms):
        selected_symptoms.append(request.form['Symptom5'])

    if len(selected_symptoms) < 5:
        disease = ["Select All Symptoms"]
    else:
        disease = diseaseprediction.dosomething(selected_symptoms)
    file1 = open("dis_pred.txt","w")
    file1.write(str(disease[0]))
    return render_template('disease_predict.html',disease=disease,symptoms=symptoms)

@app.route('/',methods=['POST'])
def default():
        predicted_dis = disease_name()
        location = get_doc_location()
        return render_template('includes/default.html', symptoms=symptoms,disease=predicted_dis,location=location)

@app.route('/drug', methods=['POST'])
def drugs():
    medicine = request.form['medicine']
    return render_template('homepage.html',medicine=medicine,symptoms=symptoms)

@app.route('/login', methods=['POST'])
def login():
    if(request.form['username']!="") and (request.form['password']!=""):
        uname=request.form['username']
        passwd = request.form['password']
        if(uname == "admin" and passwd == "admin"):
            predicted_dis = disease_name()
            location = get_doc_location()
            return render_template('includes/default.html', symptoms=symptoms,disease=predicted_dis,location=location)
        else:
            return render_template('login.html')

@app.route('/doc_names', methods=['POST'])
def doctor_names():
    if(request.form['Location']!=""):
        global selected_location
        selected_location = str(request.form['Location'])
        doc_list = get_doc_list(selected_location)
        doc_list = doc_list[0]
        print(doc_list)
        
        return render_template('find_doctor.html', doc_list=doc_list)

@app.route('/location', methods=['POST'])
def loca():
    doc_name = request.form['Doctors']
    location = get_doc_location()
    lat,lon = get_doc_address(doc_name)
    predicted_dis = disease_name()

    import folium
    my_map = folium.Map(location = [lat, lon],
                                            zoom_start = 12)

    folium.Marker([lat,lon],
        popup = 'Location',icon=folium.Icon(color='red')).add_to(my_map)

    # save as html
    my_map.save("my_map.html ")
    import webbrowser
    import os
    webbrowser.open('file://' + os.path.realpath("my_map.html"),new=2)
    #return render_template('includes/my_map.html')
    return render_template('includes/default.html', symptoms=symptoms,disease=predicted_dis,location=location)

if __name__ == '__main__':
    app.run(debug=True)




