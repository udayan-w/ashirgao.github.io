---
layout: post
title:  "Visual Search using Deep Learning (pt. 0) - Introduction"
date:   2017-09-09 12:12:12 +0530
---
<br>
![visual_meme2](/resources/vs/2.jpg)
<br><br>
<br>

Visual Search is a very interesting task where (not to everyones) surprisingly deep learning out performs many traditional soultions.
<br><br><br>
![visual_meme1](/resources/vs/1.png) 
<br>

I think that this task, visual search, is the perfect example to showcase  how deep learning is more than just a "fancy" term used by the industry and how it has the potential to revolutionalize many industry practices. Let's talk more about this in another post. 
<br>

When one talks about visual search, many intersesting applications come to mind. In this post, I will first describe the application I had in mind while taking up this task. In the subsequent post I will share parts of my implementation to get you started.

### My application for visual search

<br>
If you have checked my [About page](https://ashirgao.github.io/about/) you know that I am currently *learning* as a Data Scientist for [OnlineSales.ai](https://onlinesales.ai/). 

PLUG 

> [OnlineSales.ai](https://onlinesales.ai/) is a startup that aims to revolutionalize the digital marketing industry by using data-driven and other intelligent technologies.


It was during my previous stint here as a data science intern that I was introduced to this problem statement by my mentor [Harshad](https://in.linkedin.com/in/harshadss). 

The e-commerce industry has had exploding success of the past 2-3 years. Apparels make up a significant portion of these online transactions. To explain my application I have considered the case of some Indian apparel e-commerce giants.

Whenever you open a product page for any apparel that you like (and want to know more about) you are also presented with some recommendations under the title "Customers also viewed : "/"You might also like :". 
<br>

For example, the following red dress :
![img](/resources/vs/a1)
<br><br>
Has the following suggestions :
<br><br>
![img](/resources/vs/a2)


Recommendations can be generated using a variety of techniques. Techniques typically fall in one of the falling categories:

1. The most commonly used technique is [Collaborative Filtering](https://en.wikipedia.org/wiki/Collaborative_filtering) which leverages data from previous customers to make recommendations.

2. Other way of generating recommendations is [Content based recommendations](https://en.wikipedia.org/wiki/Recommender_system#Content-based_filtering). These methods are based on a description of the item and a profile of the user’s preferences.

Methods currently used on many retail sites fall in the first category. We often get good results using such methods.

**But we can do better.**

Apparels choices are majorly dictated by their visual appearance. It is safe to assume that the customer who has visited a product page has either liked the color / pattern / design or some other visual attribute of the product and might be interested in seeing other products with features similar to the existing product. We hope to exploit this desire by having an usupervised model looking through the entire catalogue and suggest visually similar products.

Have a look at the suggestions generated my model trained on a catalogue of an Indian e-commerce giant.
<br><br>
![img](/resources/vs/3.png)
Notice that all the suggestions have floral patterns similar to the given image.

<br><br>
![img](/resources/vs/4.png)
In this case, the model unknowingly suggested a different image for the same product as a suggestion. Notice all suggested product images have numeric prints on them.
<br><br>
Such suggestions can definitly influence the time shoppers spend on the retailer's websites, their browsing patterns and possibly the amount of revenue they bring. Thus, this application makes sense and is worth exploring while implementing visual search. 
### Another interesting application / future work 
A live solution where one could use a phone camera to point at any apparel and get shopping links to purchase similar looking products online.
<br><br>
### In conclusion...
In subsequent posts, I will share the paper that I referred and implementation details. I have used the library ```Keras``` using a TensorFlow backend for this task.

Let us continue further in the next post. Meanwhile, comment down below other applications where you think visual search can help solve interesting problems.
