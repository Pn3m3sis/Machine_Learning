import pandas as pd
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn import metrics
import numpy as np
import itertools
import re
import nltk
import pickle
import argparse

def main(file_name):
    Model = pickle.load(open(file_name, 'rb'))

    choice = input('Do you want to test a fake or real article? input fake or real: ').lower()
    if choice == str('real'):
        article = ['''PALO ALTO, Calif. After years of scorning the political process, Silicon Valley has leapt into the fray.
The prospect of a President Donald J. Trump is pushing the tech community to move beyond its traditional
role as donors and to embrace a new existence as agitators and activists. A distinguished venture capital
firm emblazoned on its corporate home page an earthy   epithet. One prominent tech chieftain says the 
consequences of Mr. Trumps election would range between disastrous and terrible. Another compares him to
a dictator. And nearly 150 tech leaders signed an open letter decrying Mr. Trump and his campaign of
anger and bigotry. Not quite all the action is  . Peter Thiel, a founder of PayPal and Palantir who was
the first outside investor in Facebook, spoke at the Republican convention in July. The New York Times
reported on Saturday that Mr. Thiel is giving $1. 25 million to support Mr. Trumps candidacy even as other
supporters flee. (He also recently gave $1 million to a super PAC that supports Senator Rob Portman, the
Republican freshman running for in Ohio.) Getting involved in politics used to be seen as clashing with
Silicon Valleys value system: You transform the world by making problems obsolete, not solving them
through Washington. Nor did entrepreneurs want to alienate whatever segment of customers did not agree
with them politically. Such reticence is no longer in style here. We're a bunch of nerds not used to
having a lot of limelight, said Dave McClure, an investor who runs a tech incubator called 500 Startups.
But to quote With great power comes great responsibility. Mr. McClure grew worried after the Republican
and Democratic conventions as Mr. Trump began to catch up to Hillary Clinton in the polls. He wanted
Silicon Valley to do more, and so late last month he announced Nerdz4Hillary, an informal  effort. An
initial group of donors pledged $50, 000 the goal was to ask the nerdz for small donations to match that
sum. They have not come through yet. We're kind of optimistic we'll get the other $50, 000 in a few
weeks, Mr. McClure said. That relatively slow pace reflects Silicon Valleys shifting position: Even as it
becomes increasingly free with its opinions, it has been less free with its checkbook. The most recent 
data, from late August, shows Mrs. Clinton taking in $7. 7 million from the tech community, according to
Crowdpac, a   that tracks donations. By that point in 2012, Crowdpac says, President Obama had raised $21 
million from entrepreneurs and venture capitalists. Reid Hoffman, the billionaire of the business
networking site LinkedIn, offers a snapshot of Silicon Valley's evolving approach to politics. 
Mr. Hoffman was a top Obama donor, giving $1 million to the Priorities USA political action committee, 
something several of his peers did as well. Last month, Mr. Hoffman garnered worldwide publicity for
saying he would donate up to $5 million to veterans groups if Mr. Trump released his taxes, a remote
possibility that never came to pass. He has castigated Mr. Trump in interviews, saying he was speaking
for those who were afraid. Mr. Hoffman's outright donations, however, have been smaller this election
cycle. In May, he gave $400, 000 to the Hillary Victory Fund. Asked if there was more recent giving that
had not shown up in federal election records, Mr. Hoffman cryptically responded in an email, Looking at
some PACs, etc. He declined several opportunities to elaborate. Even as Priorities USA has raised $133
million this election cycle, far exceeding its total in 2012, its tech contributions have dwindled. The
only familiar tech name this time around is John Doerr of the venture capital firm Kleiner Perkins 
Caufield  Byers, who gave $500, 000. The AOL   Steve Case said his September endorsement of Mrs. Clinton,
via an in The Washington Post, was the first time he ever publicly declared for a candidate. I always
focused on policy and avoided politics, he said. But if Trump were elected president, I would be 
disappointed in myself for not acting. When he wrote the he was uncertain about donating money to Mrs.
Clinton, saying only that it was probable. A spokeswoman said Sunday that Mr. Case gave $25, 000 to the
Hillary Victory Fund. Mason Harrison, Crowdpac's head of communications, offered a possible reason for
Mrs. Clinton's support. Donors give to support candidates they love, not to defeat candidates they fear,
he said. A few billionaires are acting instead of talking. Dustin Moskovitz, a founder of Facebook, said
he was giving $20 million to various Democratic election efforts the first time he and his wife, Cari 
Tuna, have endorsed a candidate. He declined to be interviewed. Part of the problem for Mrs. Clinton is
that, however preferable she may be to Mr. Trump in the tech community, she pales in comparison to 
President Obama. After some initial misgivings, Silicon Valley found its champion in him. There has been
a revolving door between tech and the Obama administration, just as previous Democratic administrations 
had a revolving door with Wall Street. In June, President Obama seemed to suggest that he might become a
venture capitalist after his term ends. Mrs. Clinton is not as enthusiastic toward Silicon Valley and 
its disruptive ways. In a speech in the summer of 2015, she noted that in the or gig economy Uber, 
Airbnb and their ilk were unleashing innovation but also raising hard questions about workplace 
protection and what a good job will look like in the future. The Clinton campaign declined to comment. 
The Trump campaign did not respond to a query. Even as Silicon Valley works against Mr. Trump, there is 
quiet acknowledgment that his campaign has bared some important issues. In an endorsement this month of 
Mrs. Clinton, the venture capital firm Union Square Ventures pointed out that the benefits of technology
and globalization have not been evenly distributed, and that this needed to change. If Silicon Valley's 
political involvement outlasts this unusual election, the tech community may start contributing more to 
the process than commentary and cash. Not only are tech people going to be wielding influence, but 
they're going to be the candidate, Mr. McClure said. Reid Hoffman, Sheryl Sandberg the chief operating 
officer of Facebook and a bunch of other folks here have political aspirations. Others may be inspired
to enter politics through other doors. Palmer Luckey is the founder of the Oculus virtual reality 
company, which he sold to Facebook for $2 billion. Mr. Luckey donated $10, 000 to a group dedicated to 
spreading messages about Mrs. Clinton both online and off. The groups first billboard, said to be 
outside Pittsburgh, labeled her Too Big to Jail. Mr. Luckey told The Daily Beast that his thinking went 
along the lines of, Hey, I have a bunch of money. I would love to see more of this stuff. He added, I 
thought it sounded like a real jolly good time. Many virtual reality developers were less happy, and 
Mr. Luckey quickly posted his regrets on Facebook. He declined to comment further. If we're going to be
more vocal, we'll have to live more transparently, said Hunter Walk, a venture capitalist whose 
campaign to persuade tech companies to give workers Election Day off signed up nearly 300 firms, 
including Spotify, SurveyMonkey and TaskRabbit. There will be a period of adjustment. But perhaps being
vocal is a temporary condition after all. The venture firm CRV was in the spotlight at the end of August
with its blunt message, which included the earthy epithet. A few weeks later, it cleaned up its website. 
The partners went from employing a publicist to seek out attention to declining interviews. We reached 
everyone we wanted to reach, and hopefully influenced opinions, said Saar Gur, a CRV venture capitalist.
Then the buzz died down and we went back to our day jobs, which are super busy.''']

        print('\n')
        print('Title: Specter of Trump Loosens Tongues, if Not Purse Strings, in Silicon Valley - The New York Times')
        print('\n')
        print(*article)
        print('\n')

    if choice == str('fake'):
        article = ['''Homeless woman protecting Trumps Hollywood Star gets attacked by big fat guy Fat man assaults homeless woman over
politics By Lexi Morgan - October 29, 2016 HOLLYWOOD, Calif. ( INTELLIHUB ) A local homeless woman was captured on
video defending Donald Trumps Hollywood Star, before being attacked by a big fat guy. The overweight man man could
be seen on video pushing the woman while yelling, Get out of here bitch. A split second later the woman was
pushed to the ground while clinging to a cart containing her belongings. Another bystander could be heard asking the
victim, Hey, where's Donald Trump at? He still ain't here, the man said. This is disgraceful behavior for any
American to engage in. Why would a big fat guy roughhouse a woman over politics? If this is what we are looking at 
now, what will we be looking at on Election Night or the days following?''']

        print('\n')
        print('Title: Homeless woman protecting Trumps Hollywood Star gets attacked by big fat guy')
        print('\n')
        print(*article)
        print('\n')

    ps = PorterStemmer()
    testdata = []
    for i in range(0, len(article)):
        temp = re.sub('[^a-zA-Z]', ' ', article[i])
        temp = temp.lower()
        temp = temp.split()
    
        temp = [ps.stem(word) for word in temp if not word in stopwords.words('english')]
        temp = ' '.join(temp)
        testdata.append(temp)

    vect = pickle.load(open('tfidf_vector.pickle', 'rb'))
    new = vect.transform(testdata)
    pred = Model.predict(new)
    if pred == [0]:
        print('Model prediction: This article is likely reliable')
    elif pred == [1]:
        print('Model prediction: This article is likely unreliable')
    else:
        print('something went wrong')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('file_name', type=str, help='Name of the model used for prediction.')
    args = parser.parse_args()
    main(args.file_name)