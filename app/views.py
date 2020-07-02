from flask import Flask, render_template, request, url_for, redirect, session, jsonify, Response, g
from flask_cors import CORS
from app import app
import makedata
import os

@app.route('/',methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/',methods=['POST'])
def run():
    inputPath = request.form['inputPath']
    outputFolder = request.form['outputFolder']
    print(inputPath)
    print(outputFolder)
    result = makedata.start(inputPath, outputFolder)
    return render_template('index.html', result=result, inputPath=inputPath, outputFolder=outputFolder)
