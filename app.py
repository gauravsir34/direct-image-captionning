from flask import Flask, render_template, redirect, request

import Caption_it

# create an object of flask
app = Flask(__name__)


# to handle different urls, using routes
@app.route('/') 
def hello():
	return render_template("index.html")


#by default method is GET
@app.route('/',methods=['POST'])
def submit_data(): 
	if request.method == 'POST':
		f = request.files['userfile']
		path = "./static/{}".format(f.filename) 
		f.save(path)

		caption = Caption_it.caption_this_image(path)
		
		result_dic = {
		'image' : path,
		'caption' : caption
		}

	return render_template("index.html", your_result = result_dic)



if __name__ == '__main__':
	# app.debug = True
	app.run(debug = True)