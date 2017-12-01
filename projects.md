---
layout: page
title: Projects
permalink: /projects/
---

<center><img src="https://i.imgur.com/OgFoqwt.jpg" alt="Photo by Ilya Pavlov on Unsplash" width="300" height="300" class="img-circle"> </center><br><br>

#### HELLO!

Find a list of some of my notable projects that I undertook after 2014. During my inital years, as a typical student, I hardly documented or used version control for my projects.
<br><br>
 ![need no git](http://adameivy.com/slides-versioncontrol/img/meme-noneedforscm.jpg)
<br><br>
#### I now regret my naivety and have started using version control locally and GitHub as well. For now, I have managed to scrap links and details of my previous projects and have provided them in the list alongwith a small story to provide my motivation behind these projects.
<br><br>

***
<br><br>
 - My [first laptop](https://support.hp.com/us-en/product/hp-mini-110-3000-pc-series/4166077/model/4232406/product-info) suddenly started having heating issues. (Configuration : 256GB storage, 2GB RAM and Intel Atom Processor. It was a triple booted: capable of running [Xubuntu](https://xubuntu.org/), [Arch Linux](https://www.archlinux.org/) and Windows 7). Thus, I wrote a [**Linux Standard Base Init script**](https://ashirgao.github.io/CPU-and-HDD-Temp-Logging/) to log individual CPU core and HDD temperatures every 2 seconds over every session to find the cause of the heating. I ended up buying a [new laptop](https://www3.lenovo.com/in/en/laptops/thinkpad/thinkpad-p/ThinkPad-P50/p/22TP2WPWP50). The old laptop now functions as a Network Attached Storage. 	
<br><br>
	![old&new](https://lh3.googleusercontent.com/OS9kwZZdphlOmAUnYMAxOoWlyquFfn50cEyWbqpJLB54HjeW1xMsnILrBSNOoOOyEJooPZPd3cswzhQuZJZR-EkaL7a6gHHm2mUrkdIhnnjXDduF6R4IuwE5xtfrlxd1bQQkBnI9Tp8=w542-h406-no)
<br><br>

***
<br><br>
 - **IoT solution deployed in real world** : Given my interest in operating systems and embedded system architecture, I indulged myself a [Raspberry Pi 2](https://www.raspberrypi.org/products/raspberry-pi-2-model-b/). Within a month, it found multiple applications as a media server (KODI + Wi-Fi Audio broadcasting), FTP server, print server etc. only to find its permanent home, deployed as part of an air gapped system responsible for controlling my buildings lighting. Picture coming soon. (ESP 8266 integration on its way) 
 <br><br>
	![esp 8266](https://lh3.googleusercontent.com/QX48v2YUiD0Nqnw0OGA4pnKGBab0zujk9dVNpu43d4nVnLXfmFNM6UPV-T6jWPb-XP-HFnoKouVN5pLAH3sTABaPgnTxr-96ypUly8ThW0oBL7PS8FhxB8Ir5GBxyLHe0ncmyT3paKc=w644-h859-no)
<br><br>

***
<br><br>
- A system similar in working to the aforementioned system, in combination with MQTT broker service provided by [IBM Bluemix](https://console.bluemix.net/catalog//) (now part of  [IBM Cloud](https://console.bluemix.net/)) and a pretty UI system was coded in a span of 24hrs for a [**hackathon**](http://i4c.co.in/img/dph/DPH_Result.htm) organised my Persistent Pvt. Ltd. in line with the ‘Digital India’ vision outlined by Honourable Prime Minister Narendra Modi. Hardware used for this solution was an Arduino Uno interfaced with a network shield and an ultrasonic sensor.
 <br><br>
 	![ultrasonic](https://lh3.googleusercontent.com/6H2u7N3bl0_u4I3PwFLEfBQGgPuVAG1dIlr1SxHNYsu6ZNLLGMdXtfkCOpiiJr4Jxqq-WhhISIWwoZG9c5zHCR73JPm_RNBOEvMpu3UcuY5IW5-5RLP1ZdBv195BTcrAOhxixMI4QUM=w644-h859-no)
<br><br>

***
<br><br>
- **Online 2 player code vs code framework** : I was an active participant in activities conducted at my [college IEEE branch](https://pict.edu/ieee/). [Credenz](https://pict.edu/event/credenz/), our annual technical event, sees particiaption from all over the country. My branch is the honoured recipient of the "Outstanding Student Branch IC Award, 2015". My notable contribution was to a biannual sub-event called [XOdia](http://xodia.pythonanywhere.com/xodia/). It involved building an online 2 player game where participants would submit a code script to play their parts. I and my fellow colleagues built the backend, validation and scoring framework in Python using the Django framework.
>I. **EnSquare**<br> 
>	The game consists of two player making either horizontal or vertical edges between any 2 consecutive dots in a 2-dimensional grid of dots. When more than two edges of a 1x1 box are occupied by same player he gets a point for that square and he captures the remaining edge of that box. The player with maximum points wins.
<br>II. **Grow**<br>
>	This game consists of a grid. Players start from a leaf node and go on occupying further branches. One branch leads  to two branches in next move. The player with larger tree or the player with maximum number of edges wins the game.
<br><br>

***
<br><br>
- **Face recognition using PCA** : An application was developed for human face recognition  using Principal Component Analysis. The application was tested on 100 people and a data-set containing 1000+ images was generated and labelled. The accuracy obtained was in the range of 72%-75% depending on the lighting conditions at the time of testing.  Programming environment used - MatLab.
<br><br>

***
<br><br>
- **Deep Learning, the BLACKBOX** : The buzz of deep learning as a solution to artificial intelligence was on and I too was a part of it. I started with multiple blogs online, courses on Coursera, YouTube, etc. and books. Graduating to a convolutional neural network, I found these models hard to visualize and interpret. Thus, I wrote a generic module to help me visualize all my models.

 ![1](/resources/proj/conv3_1.gif) 
 ![2](/resources/proj/1.png) ![3](/resources/proj/2.png)
 MORE BELOW
<br><br>

***
<br><br>
- **MNIST for Marathi** : [Marathi](https://en.wikipedia.org/wiki/Marathi_language) is the world's 19th most spoken language. Marathi also happens to be my mother tongue, and is spoken in most of western India (the state of Maharashtra). Data collection of a MNIST like marathi character recognition task was undertaken by Prof. D.T. Mane, my project guide at [Pune Institute of Computer Technology](https://pict.edu), for his PhD project. I assisted him in taking this further, by providing a deep learning solution capable of giving acceptable (94%) digit classification results. Unlike MNIST, this dataset had scarce and low quality labelled data and hence an intelligent architectural change was included in the solution. More below. Some intermediate results.
 PS. Marathi digit for 2 is very similar

![4](/resources/proj/3.png)
![5](/resources/proj/4.png)
![6](/resources/proj/5.png)
![7](/resources/proj/6.png)
<br><br>

***
<br><br>
- **Traffic Density Detection** : Graduating from toy Deep Learning problems onto real world problems.  Traffic is a problem most of us city dwelling folks experience on a daily basis and thus seemed like a good idea for a deep learning application. With the help of Prof. D.T. Mane, I was able to come up with a sufficiently large (tens of minutes) dataset of manually labelled footage collected at road junctions and over-passes in Pune a.k.a. [Oxford of the East](https://en.wikipedia.org/wiki/Pune). Detailed results and anlysis of this project are available at [https://github.com/ashirgao/Traffic-Intelligence](https://github.com/ashirgao/Traffic-Intelligence). Future plan for this project is to deploy this model alongside a number plate recognition model on Raspberry Pi *using TensorFlow for ARM or [GEMM](https://petewarden.com/2015/04/20/why-gemm-is-at-the-heart-of-deep-learning/)* at various juntions in the city to get highly accurate traffic updates for intelligent routing. Some examples of the data samples collected : <br><br> ![traffic](/resources/proj/traffic.png)![traffic](/resources/proj/traffic2.png)![traffic](/resources/proj/traffic3.jpg)
<br><br>

***
<br><br>
- Taking a break from computer vision model, I started reading up on deep learning architectures commonly used in NLP (temporal data). Reinforcement learning was making big waves and OpenAI had just launched Gym. With such daily advancements in not just AI or deep learning but related fields especially in distributed computing (BLOCKCHAINS, democrataizing H/W, AI and a lot more; Ethereum FTW), it is very easy to fall behind the current updates in related fields. Thus, I implemented fun projects in a variety of these domains of computer applications to keep myself updated of these developments. Training a ```wod2vec``` model on Game of Throne book was among the project that provided interesting results and was a cool geeky conversation topic.   
<br><br>

***
<br><br>
- **Convolutional AutoEncoders** : This unsupervised architecture features in multiple future projects of mine. In my opinion, these models are an excellent alternative to transfer learning and fit a use-case when one has little labelled data but plenty of un-labelled data (from the same distribution). 
<br><br>

***
<br><br>
- **CTR prediction** : This was one of my major projects. I undertook this projects alongside [Pranav](https://www.linkedin.com/in/pranav-havanurkar-96b641107), [Shivani](https://in.linkedin.com/in/shivani-firodiya-0646659a) and [Rudra](https://in.linkedin.com/in/rudra-lande-41747321). An application for the model in the project and guidance was provided by [Harshad Saykhedkar](https://in.linkedin.com/in/harshadss). <br>Digital marketing is a growing industry with annual spends running in billions of dollars.<br><br>
![shirt](/resources/proj/p1.png)
A advertisement, as shown in the figure above, is a very visual medium and the chances of interaction by the viewer depends as much on the image as much on the actual price of the product. This project aimed to identify what made images clickabe and score multiple images of a single product on their clickability. A convolutional autoencoder in conjunction with a convolutional neural network were used to identify features that make (or do not make) images clickable. Some examples :
<br><br>
GOOD
<br><br>
![good](/resources/proj/good.jpg)
<br><br>
BAD
<br><br>
![bad](/resources/proj/bad.jpg)
<br><br>

***
<br><br>
- **Visual Search** - This project was done during my time as a Data Science intern at Merkle-Sokrati. This involved implementing a three way siamese network where the replicated network itself is a combination of a deep and multiple shallow networks. This network is defined here :<br> [https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/42945.pdf](https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/42945.pdf) 
<br>
This task was implemented with a apparel recommendation application in mind. Details of this project can be found in my 4 part writeup starting [here](http://127.0.0.1:4000/2017/09/09/visual-search-1.html).
<br><br>

***
<br><br>
More coming soon....


