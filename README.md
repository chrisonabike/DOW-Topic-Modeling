# DOW-Topic-Modeling
Topic Modeling for DOW Chemical using LDA<BR><BR>

About: <BR>
DOW Chemical compiled a file of online reviews of their products, scraped from prominent ecommerce wwebsites such as Amazon and Walmart. The goal is to analyze these reviews and identify key prodct attributes and topics to promote how its existing products can be employed to improve performance in areas where unmet needs exist. 

This code was developed using Python 3.5. It also requires the following packages to be installed:<BR>
pandas, re, time, nltk, sklearn, numpy, textblob<BR>
<BR>
This folder contains 3 Python Jupyter notebooks:<BR>
Homecare.ipynb<BR>
Feature Building For LDA.ipynb<BR>
LDA and NMF.ipynb<BR>
<BR>
Three python scripts with the same code as the notebooks above:<BR>
Homecare.py<BR>
Feature Building For LDA.py<BR>
LDA and NMF.py<BR>
	
Three datasets:<BR>
wipes_reviews.csv<BR>
Ratings_Only_Reviews.csv<BR>
wipes_market_item_name_terms.csv<BR>


#/////////////////<BR>
#/ INSTRUCTIONS //<BR>
#/////////////////<BR>

1) Ensure all files are saved in the same directory and all packages and dependencies are installed.<BR>
2) To run the fiels, paste the code into your IDE or open the notebooks in Jupyter.<BR>
3) Run 'Homecare' (.py or ipynb) first. This will output 4 .csv files:<BR>
home_products.csv    <- The data in this file is used by the subsequent script in step 4 <BR>
count_of_products.csv <BR>
products_with_more_than_ten_reviews.csv <BR>
only_sentences.csv <- can be used for topic modeling on individual sentences or for sentiment analysis. <BR>
<BR>
4) Run 'Feature Building For LDA' (.py or ipynb). This uses the output from 'Homecare.py' (named 'home_products.csv') <BR>
This script is creating new columns in the dataframe that are variations on the original review text. <BR>
These new feature columns can be used for analysis with LDA and other algorithms to determine if better topics/attributes can be identified. <BR>
The features that are built are applied to both the file with the whole review and the sentence only file. <BR>
Additionally, these records are "tagged" with the topics the contain. <BR>
<BR>
5) The last file to run is 'LDA and NMF' (.py or ipynb). <BR>
This code contains two models and is set up to automatically read in the data and train an LDA and NMF (Non-Negative Matrix Factorization) model. <BR>
There is a section for each model at the end of the script with many parameters that can be tuned to affect the output of the topics. <BR>
Comments are included with each parameter (taken from the SKLearn website). This code is designed to be run multiple times, iteratively adjusting the parameters and manually inspecting the topics and attributes to assess for furthur use. <BR>
				
<center>*******PARAMETER TUNING**********</center><BR>
When running this script, each model section has 12 lines of code that set the data_sample (for lda or nmf) to a specific feature set created in the previous script. To change which feature set the model uses, simply uncomment (remove the #) from one of the lines. After the data sample section you will find a group of variables (such as lda_features) that are set to an integer or float value. By experimenting with these values, you will change the output of the model and the topics it creates. 
