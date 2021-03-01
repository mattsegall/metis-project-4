# **Analyzing Trump's Tweets with NLP**

by Matt Segall

### Overview

America seems to be growing ever more divided. As we just got through one of the most polarizing presidencies in history, I thought it would be important to dissect some of Trump’s public-facing communication. At first, I was thinking of investigating his speeches, but I thought his famous tweets might be a more raw source from which to draw insight. Ultimately, I wanted to attempt to answer the questions: What was Trump always so busy tweeting about? How did these change throughout his presidency? And what does this say about his overall style, and him as a person and leader?

### Data

Because trump is banned from Twitter, his activity is no longer available through their API. SO, I began with a [dataset](https://www.kaggle.com/austinreese/trump-tweets) I found on Kaggle, of 40,000 tweets from when he first created his account (2009) though April 2020. Notably, this misses most of the COVID pandemic, his failed reelection and second impeachment.

### Process

I began by working some topic modeling, trying multiple tools for representing documents (count-vectorizer, TF-IDF) in combination with various dimension-reduction techniques (PCA, NMF). Ultimately, count-vectorizer + NMF yielded the most plausible collections of words.

One topic particularly interested me. It's top words by weight are:

"white, house, totally, dont, **sources**, **stories**, russia, **dishonest**, **reporting**, ratings, said, like, good, bad, **corrupt**, **story**, cnn, media, **fake**, **news**"

So, I named this topic vector "fake news", and I followed it through the rest of my analysis.

![Fake_news_tweets_vs_time](/Users/user/Desktop/Metis/Projects/metis-project-4/Images/Fake_news_tweets_vs_time.png)

Shown above is the strength of this fake news topic weight throug time, which surprisingly does not track with some of the more major events of his presidency. I would expect that he would tweet more about fake news, attempt to stoke distrust in the media, immediately following his impeachment, for example. I would love to further explore this by digging deeper into his specific tweets during these peaks and valleys.