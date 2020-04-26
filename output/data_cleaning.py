import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import seaborn as sns
import plotly.express as px
from plotly.offline import plot 

df = pd.read_csv('20.25.04_IN_videos.csv')

df = df.drop(['video_id', 'channelId', 'thumbnail_link', 'comments_disabled', 'ratings_disabled'], axis = 1)

published_date = df['publishedAt'].apply(lambda x: x.split('T')[0])
"""
#views vs channels
fig = px.bar(df[['channelTitle', 'view_count']].sort_values('view_count', ascending = 'False'),
            y = 'view_count', x = 'channelTitle', color = 'channelTitle', log_y = True,
             template = 'ggplot2', title = 'channels vs views')
plot(fig)
fig.write_html("C:/Users/Tejan/.spyder-py3/work/youtubeScraper/output/views_vs_channel.html")

#likes vs channels
fig = px.bar(df[['channelTitle', 'likes']].sort_values('likes', ascending = 'False'),
            y = 'likes', x = 'channelTitle', color = 'channelTitle', log_y = True,
             template = 'ggplot2', title = 'channels vs likes')
plot(fig)
fig.write_html("C:/Users/Tejan/.spyder-py3/work/youtubeScraper/output/likes_vs_channel.html")

#dislikes vs channels
fig = px.bar(df[['channelTitle', 'dislikes']].sort_values('dislikes', ascending = 'False'),
            y = 'dislikes', x = 'channelTitle', color = 'channelTitle', log_y = True,
             template = 'ggplot2', title = 'channels vs dislikes')
plot(fig)
fig.write_html("C:/Users/Tejan/.spyder-py3/work/youtubeScraper/output/dislikes_vs_channel.html")


#comments vs channels
fig = px.bar(df[['channelTitle', 'comment_count']].sort_values('comment_count', ascending = 'False'),
            y = 'comment_count', x = 'channelTitle', color = 'channelTitle', log_y = True,
             template = 'ggplot2', title = 'channels vs comments')
plot(fig)
fig.write_html("C:/Users/Tejan/.spyder-py3/work/youtubeScraper/output/comments_vs_channel.html")


plt.figure(figsize = (50, 15))
#plt.bar(df.channelTitle, df.view_count, label = 'Total Views')
plt.bar(df.channelTitle, df.likes, label = 'Total likes')
plt.bar(df.channelTitle, df.dislikes, label = 'Total Dislikes')
plt.bar(df.channelTitle, df.comment_count, label = 'Total comments')
plt.xlabel('Channels')
plt.xticks(rotation=90)
plt.ylabel('count')
plt.legend(frameon = True, fontsize = 12)
plt.title('Views vs Likes vs Dislikes vs Comments ')
fig = plt.gcf()
plt.savefig('allView.png', dpi = 100)
plt.show()

category_count = df['categoryId'].value_counts() # frequency for each category
category_count
#plot title length vs channel
df["title_length"] = df["title"].apply(lambda x: len(x))
fig = px.bar(df[['channelTitle', 'title_length']].sort_values('title_length', ascending = 'False'),
            y = 'title_length', x = 'channelTitle', color = 'channelTitle', log_y = True,
             template = 'ggplot2', title = 'channels vs length of title')
plot(fig)
fig.write_html("C:/Users/Tejan/.spyder-py3/work/youtubeScraper/output/titlelength_vs_channel.html")

#normal distribution
df['likes_log'] = np.log(df['likes'] + 1)
df['views_log'] = np.log(df['view_count'] + 1)
df['dislikes_log'] = np.log(df['dislikes'] + 1)
df['comment_log'] = np.log(df['comment_count'] + 1)

plt.figure(figsize = (12,6))

plt.subplot(221)
g1 = sns.distplot(df['views_log'])
g1.set_title("VIEWS LOG DISTRIBUITION", fontsize=16)

plt.subplot(224)
g2 = sns.distplot(df['likes_log'],color='green')
g2.set_title('LIKES LOG DISTRIBUITION', fontsize=16)

plt.subplot(223)
g3 = sns.distplot(df['dislikes_log'], color='r')
g3.set_title("DISLIKES LOG DISTRIBUITION", fontsize=16)

plt.subplot(222)
g4 = sns.distplot(df['comment_log'])
g4.set_title("COMMENTS LOG DISTRIBUITION", fontsize=16)

plt.subplots_adjust(wspace = 0.2, hspace = 0.4,top = 0.9)
plt.savefig('Norma_distribution.png', dpi = 100)
plt.show()

#category id vs likes
plt.figure(figsize = (14,6))
g = sns.boxplot(x='categoryId', y='likes_log', data=df, palette="Set1")
g.set_xticklabels(g.get_xticklabels(),rotation=45)
g.set_title("Likes Distribuition by Category Names ", fontsize=15)
g.set_xlabel("", fontsize=12)
g.set_ylabel("Likes(log)", fontsize=12)
plt.savefig('boxplotcatIdlikes.png', dpi = 100)
plt.show()

#correlation
h_labels = [x.replace('_', ' ').title() for x in 
            list(df.select_dtypes(include=['number', 'bool']).columns.values)]
corr = df.corr()
fig, ax = plt.subplots(figsize=(10,6))
sns_plot = sns.heatmap(corr, annot=True, xticklabels=h_labels, yticklabels=h_labels, cmap=sns.cubehelix_palette(as_cmap=True), ax=ax)
sns_plot.figure.savefig('heatmap.png')


#word map 
from wordcloud import WordCloud, ImageColorGenerator, STOPWORDS
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

words = " ".join(df['tags'])
def punctuation_stop(text):
    filtered = []
    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(text)
    for w in word_tokens:
        if w not in stop_words and w.isalpha():
            filtered.append(w.lower())
    return filtered
    
words_filtered = punctuation_stop(words)
text = " ".join([ele for ele in words_filtered])

wc = WordCloud(background_color = 'white', random_state = 1, stopwords = STOPWORDS, max_words = 2000,
               width = 800, height = 1500)
wc.generate(text)
plt.figure(figsize = [10,10])
plt.imshow(wc, interpolation = "bilinear")
plt.axis('off')
plt.savefig('tags_wordcloud.png', dpi = 100)
plt.show()

words = " ".join(df['title'])
def punctuation_stop(text):
    filtered = []
    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(text)
    for w in word_tokens:
        if w not in stop_words and w.isalpha():
            filtered.append(w.lower())
    return filtered
    
words_filtered = punctuation_stop(words)
text = " ".join([ele for ele in words_filtered])

wc = WordCloud(background_color = 'white', random_state = 1, stopwords = STOPWORDS, max_words = 2000,
               width = 800, height = 1500)
wc.generate(text)
plt.figure(figsize = [10,10])
plt.imshow(wc, interpolation = "bilinear")
plt.axis('off')
plt.savefig('title_wordcloud.png', dpi = 100)
plt.show()
