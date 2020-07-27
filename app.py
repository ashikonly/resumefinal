from flask import *
from pdfminer.high_level import extract_text

import pandas as pd
import matplotlib.pyplot as plt

import re
import string
import pickle

import nltk  # Module for text processing
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

from wordcloud import WordCloud  # Plotting the word_cloud

filename = 'resume_prediction.pkl'
clf = pickle.load(open(filename, 'rb'))

filename2 = 'word_vec.pkl'
word_vectorizer = pickle.load(open(filename2, 'rb'))

filename3 = 'le.pkl'
le = pickle.load(open(filename3, 'rb'))

app = Flask(__name__)


@app.route('/')
def upload():
    return render_template("index.html")

@app.route('/result', methods=['POST'])
def process():
    try:
        if request.method == 'POST':
            resume = request.files['resume']
            input_jd = request.form['jd']
            # f.save(f.filename)
            # print(job_d)
            input_resume = extract_text(resume)

            # print(input_resume)
            def cleanResume(resumeText):
                resumeText = re.sub('http\S+\s*', ' ', resumeText)  # remove URLs
                resumeText = re.sub('RT|cc', ' ', resumeText)  # remove RT and cc
                resumeText = re.sub('#\S+', '', resumeText)  # remove hashtags
                resumeText = re.sub('@\S+', '  ', resumeText)  # remove mentions
                resumeText = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ',
                                    resumeText)  # remove punctuations
                resumeText = re.sub(r'[^\x00-\x7f]', r' ', resumeText)
                resumeText = re.sub('\s+', ' ', resumeText)  # remove extra whitespace
                return resumeText

            # importing the necessary package of the nltk for text processing
            nltk.download('stopwords')
            nltk.download('punkt')

            # Cleaning the text in the resume and also displaying most common words as a word cloud
            oneSetOfStopWords = set(stopwords.words('english') + ['``', "''"])
            totalWords_resume = []
            Sentences_resume = input_resume
            cleanedSentences_resume = ""
            cleanedText_resume = cleanResume(input_resume)
            cleanedText_resume = cleanedText_resume.lower()
            cleanedSentences_resume += cleanedText_resume
            requiredWords_resume = nltk.word_tokenize(cleanedText_resume)
            porter = PorterStemmer()
            for word in requiredWords_resume:
                if word not in oneSetOfStopWords and word not in string.punctuation:
                    totalWords_resume.append(porter.stem(word))

            input_resume_cleaned = ' '.join(totalWords_resume)
            wordfreqdist_resume = nltk.FreqDist(totalWords_resume)
            mostcommon_resume = wordfreqdist_resume.most_common(50)

            print(mostcommon_resume)
            print(cleanedText_resume)

            wc_resume = WordCloud().generate(cleanedText_resume)
            plt.figure(1, figsize=(15, 15))
            plt.imshow(wc_resume, interpolation='bilinear')
            plt.axis("off")

            counter = 0
            for i in input_resume_cleaned.split():
                if i == 'cid':
                    counter = counter + 1
            if counter > 20:
                return render_template("warning.html")

            # Repeating the procedure for Jd
            oneSetOfStopWords = set(stopwords.words('english') + ['``', "''"])
            totalWords_jd = []
            Sentences_jd = input_jd
            cleanedSentences_jd = ""
            cleanedText_jd = cleanResume(input_jd)
            cleanedText_jd = cleanedText_jd.lower()
            cleanedSentences_jd += cleanedText_jd
            requiredWords_jd = nltk.word_tokenize(cleanedText_jd)
            for word in requiredWords_jd:
                if word not in oneSetOfStopWords and word not in string.punctuation:
                    totalWords_jd.append(porter.stem(word))
            wordfreqdist_jd = nltk.FreqDist(totalWords_jd)
            mostcommon_jd = wordfreqdist_jd.most_common(50)
            print(mostcommon_jd)
            print(totalWords_jd)

            wc_jd = WordCloud().generate(cleanedText_jd)
            plt.figure(2, figsize=(15, 15))
            plt.imshow(wc_jd, interpolation='bilinear')
            plt.axis("off")

            # Converting the distinct values in resume to a set
            input_resume_w = set(totalWords_resume)
            len(input_resume_w)
            input_jd_w = totalWords_jd
            len(input_jd_w)

            # Calculating the score of how many distinct words in jd are present in resume
            score = 0
            for word_1 in input_jd_w:
                for word_2 in input_resume_w:
                    if word_1 == word_2:
                        score = score + 1
            print(score)
            Final_score_p = (score / len(input_jd_w)) * 100
            print(Final_score_p)

            # Creating a data frame in the format for prediction
            rez = pd.DataFrame(
                {'Category_input': '', 'Resume_input': input_resume, 'cleaned_resume_input': input_resume_cleaned},
                index=[0])
            requiredText_new = rez['cleaned_resume_input'].values

            # WordFeatures = word_vectorizer.transform(input_resume_cleaned)
            WordFeatures_new = word_vectorizer.transform(requiredText_new)
            prediction_final = clf.predict(WordFeatures_new)

            Resume_header = le.inverse_transform(prediction_final)[0]

            print(le.inverse_transform(prediction_final))

            return render_template("success.html", name=Resume_header, jd=Final_score_p)
    except:
        return render_template("filemissing.html")

if __name__ == '__main__':
    app.run(debug=True)
