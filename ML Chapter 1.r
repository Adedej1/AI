##############################################################################################################################
################################################## CHAPTER 1 ################################################################
############################################################################################################################

# The concept of artificial intelligence has been portrayed in science fiction as inevitably leading to catastrophic wars between 
# machines and humans. However, the reality of machine learning is far from this dramatic narrative. Initially, computers were able
# to play simple games like chess and later were given control of traffic lights and communication systems. As technology advanced, 
# machines were also used in military drones and missiles. However, once computers became capable of learning and teaching themselves, 
# the dynamics changed, and humans were no longer needed to program them. This significant turning point in the evolution of machines has 
# sparked concerns about the potential risks and consequences of creating autonomous machines. Fortunately, current machines still require 
# user input, and the goal of machine learning today is to assist humans in making sense of large data stores, not to create an artificial
# brain. 
# This chapter aims to provide a nuanced understanding of machine learning, its fundamental concepts, and its practical applications. 
# By the end of this chapter, readers will gain a deeper understanding of machine learning, its origins, and its uses, as well as learn how
# to apply machine learning algorithms to real-world problems.


# The Origin of Machine Learning
# Machine learning started with the way humans learn and understand the world. From the moment we're born, our bodies are collecting data 
# through our senses - what we see, hear, smell, taste, and touch. Our brain takes all this data and helps us understand the world.
# As humans, we've always been curious and wanted to learn more. We've been collecting data and writing it down for thousands of years -
# from tracking the stars in the sky to recording the weather.

# The Next Step: Electronic Sensors and Computers
# With the invention of electronic sensors and computers, we can now collect and store huge amounts of data. This data can help us make 
# better decisions, predict what might happen in the future, and even control machines.

# The Birth of Machine Learning
# Machine learning was born out of the idea that computers can learn from data, just like humans do. It's a way to teach computers to 
# make sense of data, learn from it, and make decisions on their own.
# In simple terms, machine learning is a way to help computers learn from data, so they can do things on their own, like recognize pictures, 
# understand speech, and make predictions

# Machine learning is a field of study that focuses on developing computer algorithms to transform data into intelligent action. It originated
# from the rapid evolution of data, computing power, and statistical methods.

# Machine learning is closely related to data mining, which involves searching for valuable insights from large databases. While there's some 
# overlap between the two fields, a key difference is that:

# - Machine learning teaches computers to use data to solve problems.
# - Data mining teaches computers to identify patterns that humans can use to solve problems.

# To illustrate the difference:

# - Using machine learning to analyze traffic data to identify patterns related to accident rates is an example of data mining.
# - Using machine learning to teach a computer to drive a car is an example of machine learning without data mining.

# Machines have achieved impressive feats, such as playing chess at expert levels, yet their ability to comprehend problems remains limited. They excel
# at processing large datasets and identifying patterns but rely on humans for direction, motivation, and the ability to translate their findings into 
# meaningful actions. Machines are better at answering questions than formulating them, making human-machine collaboration essential. This partnership 
# can be likened to a bloodhound and its trainer: the dog has a superior sense of smell, but without guidance, it may not achieve its purpose.

# The effective application of machine learning requires an understanding of its uses and abuses, as outlined below:

# Uses of Machine Learning
# Pattern Recognition: Analyzing extensive datasets to identify trends and make predictions.
# Automation: Streamlining repetitive tasks, enabling humans to focus on creativity and strategy.
# Decision Support: Providing insights and recommendations to enhance human decision-making.
# Improved Accuracy: Enhancing performance in fields like image recognition, speech processing, and natural language understanding.

# Abuses of Machine Learning
# Bias and Discrimination: Reinforcing societal biases if trained on skewed or incomplete data.
# Job Displacement: Automating tasks, potentially leading to workforce redundancies.
# Privacy Concerns: Collecting and analyzing personal data can raise surveillance and privacy issues.
# Lack of Transparency: The complexity of machine learning models can obscure errors and biases, making them hard to address.
# Over-reliance on Technology: Excessive dependence may erode human critical thinking and problem-solving skills.

# Machine Learning Successes
# Machine learning is most effective when it complements human expertise. It is used widely across fields to analyze and make sense of large datasets. 
# Prominent applications include:

# Filtering unwanted spam in emails.
# Customer behavior segmentation for targeted advertising.
# Weather and long-term climate forecasts.
# Detecting fraudulent credit card transactions.
# Calculating financial damage from storms and disasters.
# Predicting election outcomes.
# Algorithms for autonomous drones and self-driving cars.
# Optimizing energy use in buildings.
# Identifying areas prone to criminal activity.
# Discovering genetic sequences linked to diseases.

# Note:The essence of machine learning lies in recognizing patterns from data to enable further decisions.

# Limits of Machine Learning
# Lack of Contextual Understanding: Machines lack common sense and cannot generalize beyond their training data.
# Example: Erroneous autocorrects and misinterpretation in speech recognition (e.g., Apple Newton's famous failure to understand handwriting).
# Machine learning struggles with complex language and context, often leading to humorous or problematic outcomes.

# Machine Learning Ethics

# Machine Learning Ethics and Risks
# Machine learning is a transformative tool capable of simplifying complex data, but it can be misused if applied without care. While it can
# benefit society through innovation, automation, and personalization, its broad or thoughtless use may treat individuals as mere data points,
# leading to unintended consequences.

# Ethical Considerations
# As machine learning evolves, ethical concerns must take precedence. These include:

# Privacy and Trust: Organizations must be cautious about how they obtain, analyze, and use personal data to avoid violating laws, terms of service,
# or consumer trust. Transparency in data usage is vital.

# Sensitive Information: Even when protected attributes (e.g., race, ethnicity, religion) are excluded, machine learning models can sometimes infer 
# this information through indirect patterns, potentially leading to biased or discriminatory outcomes.

# Illustrative Example
# This story highlights how predictive analytics can misfire:
# A retailer used machine learning to identify expectant mothers and send them promotional materials for maternity products. In one case, a father 
# was upset to receive such offers for his teenage daughter — only to later discover she was pregnant.
# This example underscores the need for caution when handling sensitive data and reminds us to consider the social and emotional implications of predictions.

# Business Risks
# Using personal data without adequate oversight or ethical considerations can harm customers and businesses alike. Negative consequences may include:

# Loss of customer trust.
# Legal repercussions from privacy violations.
# Public backlash against perceived corporate overreach.

# Key Takeaways
# Machine learning is only as reliable as the data it learns from and must be applied thoughtfully to avoid reinforcing biases or breaching ethical boundaries.
# Transparency, fairness, and responsible data usage are critical in designing algorithms that respect user privacy and dignity.
# Ethical frameworks, such as "Do no harm," should guide all machine learning initiatives to balance innovation with accountability.

#How machines learn
# Machine Learning Overview and Key Components
# Machine learning is the process by which machines use experience to improve their performance on similar tasks in the future. It plays a fundamental role in 
# artificial intelligence (AI), and its functionality relies on four essential components:

# Data Storage
# Machine learning starts with collecting and storing data, which forms the basis for learning and improvement.
# High-quality and relevant data is essential for effective learning.

# Abstraction
# This involves translating raw data into broader, more meaningful concepts, identifying patterns, and relationships that the machine can use.

# Generalization
# The abstracted data is then used to create knowledge and inform decisions or predictions.
# This is where models apply learned patterns to new, unseen data.

# Evaluation
# Feedback is provided to measure the success of the model and improve its performance.
# Continuous evaluation ensures the system refines itself over time.

# While the machine learning process is divided into four components—data storage, abstraction, generalization, and evaluation—these processes occur 
# subconsciously in humans. For machines, however, these steps are explicit, enabling their learning processes to be analyzed, shared, and applied to 
# various tasks.

# Key Differences Between Human and Machine Learning
# 
# Subconscious vs. Explicit Learning
# Human Learning: Happens subconsciously and is often intuitive.
# Machine Learning: Operates explicitly, breaking down each step into a structured process.

# Subjectivity vs. Transparency
# Human Learning: Influenced by emotions, personal experiences, and subjective interpretations.
# Machine Learning: Transparent and data-driven, with processes open to examination.

# Individuality vs. Replicability
# Human Learning: Unique to each individual, shaped by personal context and environment.
# Machine Learning: Standardized, replicable, and transferable across different systems and tasks.

# Data Storage 
# Data storage is the foundation of learning for both humans and computers. Humans use their brain's electrochemical signals to store and process observations,
# while computers use hardware like hard drives, flash drives, and RAM, combined with a CPU.
# Limitations of Data Storage
# However, data storage alone is insufficient for learning. Without a higher level of understanding, knowledge is limited to recall, and data is merely a 
# collection of ones and zeros without inherent meaning.
# The Importance of Understanding
# To truly learn, one must go beyond mere memorization and develop strategies to relate ideas and store information. This approach enables the understanding of
# large concepts without needing to memorize them by rote.
# Effective Learning Strategies
# Effective learning involves selectively memorizing key ideas, developing relationships between them, and creating strategies for storing and retrieving 
# information. This approach allows for a deeper understanding of complex concepts and enables learners to apply their knowledge in new and unexpected situations.


# Abstraction
# Abstraction assigns meaning to raw data, turning it into comprehensible concepts, as illustrated by René Magritte’s artwork "The Treachery of Images," emphasizing
# the distinction between objects and their representations.
# In machine learning, abstraction occurs through knowledge representation, where data is summarized using models that uncover patterns. This process, called training, 
# fits a model to the data based on the task and available information.
# Key Insights from Machine Learning Models
# Trained models don’t create new data but reveal hidden structures and relationships. For example, the discovery of gravity involved fitting equations to observed data, 
# uncovering an entirely new concept.
# Applications of Machine Learning
# Genomics: Identifying genes linked to diseases like diabetes.
# Fraud Detection: Recognizing transaction patterns before fraud occurs.
# Psychology: Highlighting traits linked to newly identified disorders.

# Generalization
# Definition: Generalization is the ability to apply abstracted knowledge to new, unseen situations. It involves transforming raw data into actionable insights.

# Importance:
# Without generalization, learners (human or machine) can only recall information but cannot apply it effectively.
# It reduces complex patterns into a manageable set of findings relevant for future tasks.

# Challenges:
# It's impossible to manually examine and rank every possible pattern by utility in large datasets.
# Machine learning uses algorithms to automate this process, focusing only on the most useful abstractions.
# 
# Key Concept:
# Heuristics: Generalization relies heavily on heuristics—educated guesses that simplify problem-solving. While efficient, they may lead to errors or oversights.
# Example: A person’s gut instinct often uses mental heuristics, like judging risks based on vivid memories.

# Heuristics and Bias in Machine Learning
# Heuristics in Algorithms:
# Machine learning algorithms employ heuristics to identify patterns quickly.
# These shortcuts reduce complexity but risk introducing bias.
# Bias:
# Definition: Systematic errors in machine learning models, often arising from flawed heuristics.
# Example: A facial recognition algorithm trained only on specific demographics might fail to recognize diverse faces.
# Bias isn't inherently bad; small biases can guide decision-making but must be managed to avoid misrepresentation or discrimination.

# Illustrative Example:
# A machine learning algorithm may detect faces by identifying eyes, nose, and mouth arranged in a specific way. It might struggle with:
# Faces with glasses or turned at odd angles.
# Variations in skin tone, shape, or lighting conditions.
# This highlights the need for diverse training data to minimize algorithmic bias.

# Real-World Insights:
# Human decision-making also uses heuristics, such as the availability heuristic, where vivid memories of rare events (e.g., plane crashes) skew perception of risk.

# Evaluation
# Purpose:
# Evaluation ensures that generalized models work effectively on unseen data and not just on the training dataset.
# It is crucial to measure how well a model generalizes and identify overfitting or underfitting.

# Key Concepts:
# Noise:
# Refers to random variations in data that can mislead learning algorithms.
# Causes include:
# Measurement errors from sensors.
# Human input errors, such as inaccurate survey responses.
# Data quality issues like missing or corrupted values.

# Overfitting:
# Occurs when a model learns the "noise" in training data rather than its actual patterns.
# Overfitted models perform poorly on new data since they fail to generalize.

# The “No Free Lunch” Theorem:
# No single machine learning technique works best for all tasks.
# Selecting the right model requires understanding the specific problem and dataset characteristics.

# Practical Steps in Evaluation:
# Train the model on a subset of data.
# Test the model on unseen data to evaluate its predictive accuracy.
# Avoid focusing too much on training performance—strong training results may not translate to real-world success.

# Machine learning in practice
# five-step process for applying machine learning in practice:

#Data Collection: Gathering data from various sources and consolidating it into a single, actionable format like a text file, spreadsheet, or database.

# Data Exploration and Preparation: Cleaning and preparing the data by addressing inconsistencies, removing unnecessary data, and ensuring it meets the
# requirements of the machine learning model.

# Model Training: Using the prepared data to train a machine learning algorithm, which represents the data in a model that fits the task at hand.

# Model Evaluation: Assessing the model's performance by evaluating its accuracy or using specific measures suited to the problem and data type.

# Model Improvement: Enhancing the model's performance by applying advanced strategies, using additional data, or switching to a different model type if needed.

#After these steps are completed, if the model appears to be performing well, it can be deployed for its intended task. As the case may be, you might utilize your
#model to provide score data for predictions, for projections of financial data, to generate useful insight for marketing or research, or to automate a task......


# Types of Input Data
# The practice of machine learning involves matching the characteristics of input data to the biases of the available approaches. Thus, before applying machine
# learning to real-world problems, it is important to understand the terminology that distinguishes among datasets.

# Unit of Observation: Refers to the smallest entity with measured properties of interest for a study. Examples include transactions, time points, or geographic 
# regions. Although the observed and analyzed units may not always align, understanding the unit of observation is crucial for analyzing trends and making inferences.

# Datasets: Composed of two main elements:
# Examples: Instances of the unit of observation for which properties have been recorded.
# Features: Recorded properties or attributes of examples that may be useful for learning.

# It is easiest to understand features and examples through real-world cases. 
# For instance:
# In spam email detection, the unit of observation is a email, with examples being specific emails and features being the words used in those messages.
# In cancer detection, the unit of observation might be a patient, with features such as genomic markers, weight, height, or blood pressure.

# Organizing Data for Machine Learning

# In machine learning, data is often structured in a matrix format, with:
# Rows as examples (e.g., car models).
# Columns as features (e.g., price, mileage, color).

# Types of Features
# Numeric: Measurable values (e.g., price, mileage).
# Categorical (Nominal): Categories with no order (e.g., color, transmission type).
# Ordinal: Ordered categories (e.g., sizes: small, medium, large).
# Understanding feature types ensures proper algorithm selection and data preparation.


## Types of machine learning algorithms
#Machine learning algorithms are divided into categories based on their purpose. Understanding these categories is key to achieving the desired outcomes.

#1 Predictive Model
# A predictive model is used for tasks that involve predicting one value based on other values in a dataset.
# The learning algorithm identifies and models the relationship between the target feature (the feature being predicted) and the input features.
# Predictive models are not limited to forecasting future events; they can also be used for understanding past events or managing real-time systems like 
# traffic lights.

#because predictive models are given clear instructions on what they need to learn and how they intend to learn it, the process of training a predictive 
#model is known as supervised learning

# Examples of Classification Tasks:

# Detecting if an email is spam.
# Diagnosing whether a person has cancer.
# Predicting if a football team will win or lose.
# Determining if a loan applicant will default.

# In classification, the target feature is a categorical feature (class) with distinct levels. These levels can be ordinal (e.g., small, medium, large) or
# non-ordinal (e.g., spam or not spam). Classification is widely used, and there are many algorithms designed to handle different input data.

# Supervised Learning for Numeric Prediction
# Supervised learners can also predict numeric values such as income, test scores, or counts of items.
# A common algorithm for numeric prediction is linear regression, which fits input data to a model for accurate predictions.
# Regression models quantify both the magnitude and uncertainty of relationships between inputs and targets.

#2 Descriptive Model
# Descriptive models summarize data to uncover meaningful insights.
# Unlike predictive models, descriptive models do not target a specific feature. Instead, they focus on identifying patterns and relationships within the data.

# Key Applications:

# Pattern Discovery:
# Identifies useful associations in data.
# Example: In market basket analysis, retailers analyze transactional data to find frequently purchased item combinations (e.g., swimming trunks and sunglasses).
# This insight can inform marketing tactics like upselling or repositioning items in stores.

# Fraud Detection and Screening:
# Originally used in retail, pattern discovery is now applied in fraud detection, identifying genetic defects, and mapping crime hotspots.

# Clustering
# A descriptive modeling task that groups data into homogeneous clusters.
# Useful for segmentation analysis, clustering is applied to group similar customers for personalized marketing campaigns.

#3 Meta-Learners
# Meta-learning focuses on teaching algorithms how to learn more effectively.
# It often combines results from multiple algorithms to improve performance.
# Useful for challenging tasks requiring high accuracy.

# To match a learning task to a machine learning approach, start by identifying the task type: classification, numeric prediction, pattern detection, or clustering. 
# Certain tasks align naturally with specific algorithms:

# Pattern detection: Association rules.
# Clustering: K-means algorithm.
# Numeric prediction: Regression analysis or regression trees.
# Classification: Requires careful consideration of algorithm distinctions.

# For classification, interpretability matters. For example, decision trees provide easily understandable models, while neural networks, though potentially more accurate,
# are less interpretable and may not be suitable for tasks requiring explanations, such as credit scoring.

# Key strengths and weaknesses of algorithms guide the choice. Often, multiple algorithms may work, allowing for flexibility based on comfort or testing. For tasks prioritizing 
# predictive accuracy, testing several models and selecting the best is essential. Advanced techniques for combining models to leverage their strengths will be explored later.

# Machine learning with R
# Using R for machine learning often requires additional packages not included in the base installation. These packages, developed by R's open-source community, are free and add 
# functionality for various machine learning algorithms. A package is a collection of R functions shared among users, and many exist for machine learning.

# At the time of writing, there were over 6,779 packages available, which can be browsed on the Comprehensive R Archive Network (CRAN). CRAN provides the latest versions of R software
# and packages for download. If you downloaded R, it was likely from CRAN. Visit https://cran.r-project.org/index.html for installation instructions and troubleshooting help.

# To explore machine learning-related packages:
# Use the Packages link on CRAN to browse packages alphabetically or by publication date.
# Check the CRAN Task Views, which organizes packages by subject. The machine learning task view is available https://cran.r-project.org/web/views/MachineLearning.html.


#Installing and Loading R Packages
#R packages simplify the process of adding functionality to R. As an example, the RWeka package provides access to machine learning algorithms from the Java-based Weka software. 
#For more information about Weka, visit http://www.cs.waikato.ac.nz/~ml/weka/. To use RWeka, ensure Java is installed. If not, download it for free at https://www.java.com/en/.

# Installing an R Package:
#   Use the install.packages() function:
install.packages("RWeka")

# R connects to CRAN, downloads the package, and installs dependencies automatically.
# If prompted, choose a CRAN mirror close to your location for faster downloads.

# To specify a custom installation path (e.g., when lacking admin privileges):
install.packages("RWeka", lib="/path/to/library")

# Other options include installing from a local file, source, or experimental versions. Details are available using:
?install.packages

#R conserves memory by not loading all installed packages by default. Instead, users load packages as needed with the library() function.
#Key Distinction:
# A library refers to the location where packages are installed.
# A package is the collection of functions and data.

#To load the previously installed RWeka package, use:
library(RWeka)

#To unload an R package, use the detach() function....................
detach("package:RWeka", unload = TRUE)


##### Chapter 3: Classification using Nearest Neighbors --------------------
# Understanding Classification Using Nearest Neighbors
# Imagine dining in a pitch-dark restaurant, relying solely on your senses of taste and smell to identify the food. At first, diners might rapidly
# assess prominent spices, aromas, and textures, comparing these sensations to prior experiences to identify the dish. For example, briny flavors 
# might evoke seafood, while earthy tones may suggest mushrooms.

# This intuitive process is similar to a principle in machine learning: if it tastes like a duck and smells like a duck, it’s probably a duck. 
# Similarly, birds of a feather flock together captures another key concept: similar items tend to share properties. These ideas underpin the 
# k-Nearest Neighbors (kNN) algorithm.

# What You’ll Learn:
# Key concepts of nearest neighbor classifiers, including why they are “lazy” learners.
# Measuring similarity between examples using distance.
# Using kNN in R to diagnose breast cancer.

# Nearest Neighbor Classifiers in a Nutshell
# Nearest neighbor classifiers assign an unlabeled example to the class of the most similar labeled examples. Despite its simplicity, this approach is 
# highly effective for tasks like:

# Computer vision: Optical character and facial recognition.
# Recommender systems: Predicting movie preferences (e.g., Netflix).
# Genetics: Detecting proteins or diseases in genetic data.

# Nearest neighbor classifiers are ideal for tasks where feature-target relationships are complex and hard to define, but similar class items are homogeneous.
# They work well when the concept is intuitively recognizable despite being difficult to formalize. However, they are less effective when group distinctions 
# are unclear, as the algorithm struggles to identify boundaries.


# kNN Algorithm 

# Strengths:
# - Simple and effective
# - Makes no assumptions about the data distribution
# - Training phase is fast

# Weaknesses:
# - No model produced, limiting insight into feature relationships
# - Classification phase is slow and memory-intensive
# - Additional processing needed for nominal features or missing data

# How it works:
# - A training dataset contains examples classified by a nominal variable.
# - A test dataset has unlabeled examples with the same features as the training data.
# - For each test record:
#   1. Identify the k nearest records from the training dataset based on similarity.
#   2. Assign the test record the majority class of the k nearest neighbors.

# Example:
# - Dataset contains two features: sweetness (1-10) and crunchiness (1-10).
# - Ingredients are labeled as 'fruit', 'vegetable', or 'protein'.
# - Example dataset:
#   | Ingredient | Sweetness | Crunchiness | Food Type |
#   |------------|-----------|-------------|-----------|
#   | Apple      | 10        | 9           | Fruit     |
#   | Bacon      | 1         | 4           | Protein   |
#   | Banana     | 10        | 1           | Fruit     |
#   | Carrot     | 7         | 10          | Vegetable |
#   | Celery     | 3         | 10          | Vegetable |
#   | Cheese     | 1         | 1           | Protein   |

# kNN Algorithm: Feature Space and Scatterplot

# The kNN algorithm represents features as coordinates in a multidimensional feature space.
# In this case, the dataset has two features: sweetness (x-axis) and crunchiness (y-axis).

# The scatterplot of the dataset shows:
# - Vegetables (e.g., celery, carrot) tend to be crunchy but not sweet.
# - Fruits (e.g., apple, grape, pear) are sweet, and they can be either crunchy or not.
# - Proteins (e.g., bacon, cheese, shrimp) tend to be neither sweet nor crunchy.

# The pattern in the scatterplot suggests that similar types of food tend to cluster together.
# kNN uses this clustering of features to classify test data based on proximity to labeled data.

# The diagram uses a nearest neighbor approach to classify a tomato as either a fruit or a vegetable
# based on its sweetness and crunchiness. Foods are grouped into three categories: Vegetables, Fruits,
# and Proteins, plotted on a 2D graph. The tomato’s position determines its closest group, helping decide 
# its classification (pg 69)

# CALCULATING DISTANCE

# The k-Nearest Neighbors (kNN) algorithm determines a data point's classification 
# by comparing its features to those of nearby data points using a distance metric.

# Distance Metrics:
# - Euclidean distance: Measures the shortest straight-line distance between two points.
#   Formula: dist(p, q) = sqrt((p1 - q1)^2 + (p2 - q2)^2 + ... + (pn - qn)^2)
# - Other distance metrics: Manhattan distance, etc., can also be used depending on the context.

# Example: Classifying a tomato (sweetness = 6, crunchiness = 4)
# - Nearby foods:
#   - Grape (sweetness = 8, crunchiness = 5): 
#     Distance = sqrt((6 - 8)^2 + (4 - 5)^2) = 2.2
#   - Green bean (sweetness = 3, crunchiness = 7): 
#     Distance = sqrt((6 - 3)^2 + (4 - 7)^2) = 4.2
#   - Nuts (sweetness = 3, crunchiness = 6): 
#     Distance = sqrt((6 - 3)^2 + (4 - 6)^2) = 3.6
#   - Orange (sweetness = 7, crunchiness = 3): 
#     Distance = sqrt((6 - 7)^2 + (4 - 3)^2) = 1.4

# Classification Process:
# - In 1NN (k = 1), the classification is based on the single nearest neighbor.
# - The orange, with a distance of 1.4, is the closest neighbor to the tomato.
# - As the orange belongs to the "fruit" class, the 1NN algorithm classifies the tomato as a fruit.

# General Notes:
# - The kNN algorithm can be extended to consider more neighbors (k > 1), 
#   with the majority class among the neighbors determining the classification.
# - The choice of distance metric and value of k significantly impacts the model's performance.
# - This method works best with normalized features to prevent features with larger ranges 
#   from dominating the distance calculation.

# Choosing k in kNN:
# - The choice of k depends on the dataset's complexity and the number of training examples.
# - A common rule of thumb: k = sqrt(n), where n is the number of training examples.
#   - For example, with 15 training records, k = 4 (sqrt(15) = 3.87, rounded).
# - However, this rule may not always yield the best k:
#   - Testing multiple k values on test datasets can help identify the optimal k.
#   - Larger, representative datasets make the choice of k less critical because subtle patterns are better captured.

# Alternative k strategies:
# - Weighted voting:
#   - Larger k can be combined with weighted voting.
#   - Closer neighbors have more influence than farther neighbors in classification.

# Preparing data for kNN:
# - Feature scaling is essential for kNN because distance measurements depend on feature magnitudes.
# - If features vary greatly in scale, the larger-scale features dominate the distance calculation.
#   - Example: Adding a "spiciness" feature (measured in Scoville units) to the dataset.
#     - Spiciness values range from 0 to over 1,000,000, overshadowing sweetness (1–10) and crunchiness (1–10).
# - Solution: Rescale all features to a standard range (e.g., 1–10) to ensure equal contribution to distances.

# Methods for feature scaling:
# - Min-max scaling: Rescales features to a specified range (e.g., 0–1 or 1–10).
# - Standardization: Converts features to have a mean of 0 and standard deviation of 1.
# - Example application:
#   - Original features: sweetness (1–10), crunchiness (1–10), spiciness (0–1,000,000).
#   - After scaling, all features would contribute equally to the distance formula.

# Rescaling Features for kNN:
# - Rescaling ensures all features contribute equally to distance calculations, especially if they have different ranges.

# Min-Max Normalization:
# - Transforms feature values to a range between 0 and 1.
# - Formula: X_new = (X - min(X)) / (max(X) - min(X))
# - Normalized values represent the percentage position of the original value within its range.
#   - Example: A value halfway between the minimum and maximum will have a normalized value of 0.5.

# Z-Score Standardization:
# - Rescales feature values based on their deviation from the mean, measured in standard deviations.
# - Formula: X_new = (X - Mean(X)) / StdDev(X)
# - Produces z-scores, which indicate how many standard deviations a value is above or below the mean.
#   - Z-scores have no predefined range and can take any positive or negative value.
# - Useful when data follows a normal distribution.

# Handling Nominal Data:
# - Euclidean distance is undefined for nominal (categorical) features.
# - Solution: Convert nominal data into numeric form using dummy coding.
# - Dummy Coding:
#   - Assigns binary values (0 or 1) to categories.
#   - Example: For a gender variable:
#     - male = 1 if gender is "male"; male = 0 otherwise.
#   - Only one feature is needed for binary categories since they are mutually exclusive.
# - Dummy coding allows categorical features to be incorporated into distance calculations.

# Summary:
# - Min-max normalization and z-score standardization are common methods for rescaling numeric features.
# - Dummy coding transforms nominal data for compatibility with distance metrics in kNN.

# Dummy Coding for Nominal Features:
# - For an n-category nominal feature, create (n - 1) binary variables as dummy features.
#   - Example: For a three-category "temperature" variable (hot, medium, cold):
#     - hot = 1 if temperature is hot, 0 otherwise.
#     - medium = 1 if temperature is medium, 0 otherwise.
#     - No feature is needed for "cold" since it's implied when both hot and medium are 0.
# - Benefits of dummy coding:
#   - The distance between dummy-coded features is always 0 or 1.
#   - Compatible with normalized numeric data without additional transformations.

# Handling Ordinal Features:
# - Ordinal features (e.g., cold < warm < hot) can be assigned numeric values (e.g., 1, 2, 3).
#   - Normalize these numeric values to fall between 0 and 1.
#     - Example: cold = 0, warm = 0.5, hot = 1.
# - Caveat:
#   - This approach assumes that the difference between categories is uniform (e.g., cold to warm is equal to warm to hot).
#   - If the differences are not equivalent, dummy coding is a safer choice.

# Why is kNN a "Lazy" Algorithm?
# - kNN is classified as a lazy learning algorithm because it skips abstraction and generalization.
#   - It doesn't build a model or learn patterns during training.
# - Instead, all computation happens at the prediction stage:
#   - It compares new data points directly to stored training examples.
# - This approach contrasts with "eager" algorithms that generalize from training data (e.g., decision trees, neural networks).
# - Lazy learners (e.g., kNN) do not "learn" during training; they store training data verbatim.
#   - Pros: Training is fast since no model is built.
#   - Cons: Prediction is slower because all computation occurs during inference.
# - Also known as "instance-based learning" or "rote learning."
#   - Classified as a non-parametric method because no parameters or models are generated.
#   - Non-parametric methods identify natural patterns rather than forcing data into predefined forms.
#   - However, they provide limited insight into how the data is being used for classification.

# Strength of kNN:
# - Despite being "lazy," kNN is powerful and flexible.
# - Its simplicity allows application to various complex problems, such as medical diagnostics.

# Diagnosing Breast Cancer with kNN:
# - Early detection of breast cancer involves routine screening and biopsies for abnormal breast masses.
#   - Biopsies use fine-needle aspiration to extract cells for microscopic examination.
#   - Clinicians determine whether a mass is malignant or benign based on cell characteristics.
# - Machine learning can automate cancer detection, benefiting the healthcare system:
#   - Improves efficiency, allowing physicians to focus on treatment.
#   - Reduces subjective errors in diagnosis, potentially improving accuracy.
# - Application of kNN:
#   - kNN can be used to analyze measurements of biopsied cells from women with abnormal breast masses.
#   - It automates the classification of masses as malignant or benign, supporting early cancer detection.

# Step 1 – Collecting Data

# - Dataset Source: 
#   - The "Breast Cancer Wisconsin Diagnostic" dataset is available from the UCI Machine Learning Repository: http://archive.ics.uci.edu/ml.
#   - The dataset was donated by University of Wisconsin researchers.
#   - It includes measurements of cell nuclei from digitized fine-needle aspirate breast mass images.

# - Dataset Reference: 
#   - For more information, refer to the paper: "Nuclear feature extraction for breast tumor diagnosis" (1993) by W.N. Street, W.H. Wolberg, and O.L. Mangasarian.

# - Data Overview:
#   - The dataset contains 569 examples of breast cancer biopsies.
#   - Each example has 32 features:
#     - 1 Identification Number.
#     - 1 Diagnosis Code: 
#       - "M" = Malignant.
#       - "B" = Benign.
#     - 30 Numeric Features: These are laboratory measurements of the characteristics of the cell nuclei.

# - Numeric Features:
#   - Features include:
#     - Mean, standard error, and worst (largest) values for 10 different characteristics of the digitized cell nuclei:
#       - Radius.
#       - Texture.
#       - Perimeter.
#       - Area.
#       - Smoothness.
#       - Compactness.
#       - Concavity.
#       - Concave points.
#       - Symmetry.
#       - Fractal dimension.
#   - These features are primarily related to the shape and size of the cell nuclei.

# - Key Insights:
#   - Many features are specific to oncology and understanding of how they relate to benign or malignant masses will emerge as 
#   - we progress with machine learning.


## Example: Classifying Cancer Samples ----
# Step 2: Exploring and preparing the data ----

# Let's explore the data and see if we can shine some light on the relationships. At the same time, we will prepare the data for 
# use with the kNN learning method.

# If you plan on following along, download the wisc_bc_data.csv file from the Packt website and save it to your R working directory.
# The dataset was modified very slightly for this book. In particular, a header line was added and the rows of data were randomly ordered.

# We'll begin by importing the CSV data file as we have done previously, saving the Wisconsin breast cancer data to the wbcd data frame:

# import the CSV file
wbcd <- read.csv("wisc_bc_data.csv", stringsAsFactors = FALSE)
wbcd

# examine the structure of the wbcd data frame
# Using the command str(wbcd), we can confirm that the data is structured with 569 examples and 32 features as we expected. The first several
# lines of output are as follows:
str(wbcd)
View(wbcd)

# The first variable is an integer variable named id. As this is simply a unique identifier (ID) for each patient in the data, it does not provide
# useful information and we will need to exclude it from the model.

# Regardless of the machine learning method, ID variables should always be excluded. Neglecting to do so can lead to erroneous findings because the 
# ID can be used to uniquely "predict" each example. Therefore, a model that includes an identifier will most likely suffer from overfitting, and is 
# not likely to generalize well to other data.

# Let's drop the id feature altogether. As it is located in the first column, we can exclude it by making a copy of the wbcd data frame without column 1:
# drop the id feature
wbcd <- wbcd[-1]

# Check the distribution of the diagnosis variable (Benign vs. Malignant)
# The 'diagnosis' variable indicates whether a tumor is benign (B) or malignant (M).
# Benign tumors are non-cancerous, meaning they are not life-threatening and usually do not spread to other parts of the body.
# Malignant tumors are cancerous, meaning they can grow uncontrollably and spread to other parts of the body, potentially causing harm.
# The table() function will show how many tumors in the dataset are benign (B) and how many are malignant (M).
table(wbcd$diagnosis)

# Recode the diagnosis variable to be a factor with more informative labels
# The 'factor' function converts the diagnosis column into a factor type.
# The 'levels' argument defines the original values (B and M), and the 'labels' argument assigns more meaningful labels.
# Now, the diagnosis variable will show "Benign" for "B" and "Malignant" for "M".
wbcd$diagnosis <- factor(wbcd$diagnosis, levels = c("B", "M"),
                         labels = c("Benign", "Malignant"))

# Display the proportions of each class in the diagnosis variable
# This function shows the proportion of benign and malignant cases as percentages.
# We multiply by 100 to get the percentage values.
round(prop.table(table(wbcd$diagnosis)) * 100, digits = 1)

# View the summary statistics of three specific features: radius_mean, area_mean, and smoothness_mean
# The 'summary()' function provides summary statistics for the given features (columns) in the dataset.
# This includes the minimum, 1st quartile (25%), median (50%), mean, 3rd quartile (75%), and maximum values.
summary(wbcd[c("radius_mean", "area_mean", "smoothness_mean")])

# Example summary output:
# This will display the following statistics for each feature:
# radius_mean   area_mean  smoothness_mean
# Min. : 6.981 Min. : 143.5 Min. :0.05263
# 1st Qu.:11.700 1st Qu.: 420.3 1st Qu.:0.08637
# Median :13.370 Median : 551.1 Median :0.09587
# Mean :14.127 Mean : 654.9 Mean :0.09636
# 3rd Qu.:15.780 3rd Qu.: 782.7 3rd Qu.:0.10530
# Max. :28.110 Max. :2501.0 Max. :0.16340

# Notice that the values of 'area_mean' are much larger than 'smoothness_mean'.
# For example, area_mean ranges from 143.5 to 2501, while smoothness_mean ranges from 0.05 to 0.16.
# Because of this difference in scale, 'area_mean' will dominate the distance calculation for kNN.
# To address this, we need to scale the features to a similar range so they can contribute equally to the model.

# Normalize the features to a standard range (0 to 1) using min-max normalization
# The normalization formula is: (x - min(x)) / (max(x) - min(x)), which rescales each value in the feature
# to be between 0 and 1, where the minimum value becomes 0 and the maximum becomes 1.


# Transformation – Normalizing Numeric Data
# To normalize these features, we need to create a normalize() function in R. This function takes a vector x of numeric values, and for each 
# value in x, subtracts the minimum value in x and divides by the range of values in x. Finally, the resulting vector is returned.

normalize <- function(x) {
  return ((x - min(x)) / (max(x) - min(x)))  
}

# Test the normalize function on a vector from 1 to 5
normalize(c(1, 2, 3, 4, 5))  # Expected Output: [1] 0.00 0.25 0.50 0.75 1.00

# Test the normalize function on a vector from 10 to 50 (larger values)
normalize(c(10, 20, 30, 40, 50))  # Expected Output: [1] 0.00 0.25 0.50 0.75 1.00


# Now apply the normalize function to the entire dataset's numeric features
# Use lapply() to apply the normalize function to each feature in the data frame
# The data frame contains 32 columns, with the first column being the ID, so we start with column 2
# Apply normalize() to columns 2 through 31, which are the numeric features
wbcd_n <- as.data.frame(lapply(wbcd[2:31], normalize))
View(wbcd_n)
# lapply() applies the normalize function to each column individually
# It returns a list, so we convert it back to a data frame using as.data.frame()


# The resulting 'wbcd_n' data frame now contains the normalized values 
# for all numeric columns in the original dataset. We append '_n' to the name
# to indicate that these values are normalized.

# Verify the transformation by checking summary statistics of one normalized feature
summary(wbcd_n$area_mean)

# Expected Output:
# Min. 1st Qu. Median Mean 3rd Qu. Max.
# 0.0000 0.1174 0.1729 0.2169 0.2711 1.0000
# Explanation: The summary shows that the values in 'area_mean' are now between 0 and 1,   

# Data Preparation – Creating Training and Test Datasets
# In machine learning, it is essential to test a model's ability to generalize to unseen data rather than simply predicting outcomes for the data
# it was trained on. Predicting already-known outcomes (like the benign or malignant labels in our dataset) is not useful for evaluating the model's
# real-world performance.

# The true challenge lies in assessing the model's ability to accurately classify new, unlabeled data. Ideally, we would use data from real-life scenarios
# (e.g., future masses with unknown cancer status) and compare the model’s predictions to the actual diagnoses. However, in the absence of such data, we 
# simulate this scenario by splitting the dataset into two portions:

# Training Dataset: Used to build the k-Nearest Neighbors (kNN) model.
# Test Dataset: Used to evaluate how well the model predicts unseen data.
# Splitting the Dataset:
# For this example:

# First 469 records are used as the training dataset.
# Last 100 records are used as the test dataset, simulating new patients. 

# Create the training dataset using the first 469 rows
wbcd_train <- wbcd_n[1:469, ]
wbcd_train

# Create the test dataset using rows 470 to 569
wbcd_test <- wbcd_n[470:569, ]
wbcd_test
# Data Preparation – Handling Target Labels for Training and Testing
# When constructing training and test datasets, it is crucial to ensure they are representative subsets of the entire dataset. In this example, the data was 
# pre-shuffled into a random order, allowing us to directly extract consecutive rows for training and testing. However, if the data were ordered non-randomly 
# (e.g., by time or grouped values), random sampling methods would be necessary to avoid bias.

# In our earlier steps, the target variable diagnosis was excluded when normalizing the features. To train and evaluate the k-Nearest Neighbors (kNN) model, we 
# need to separate the target labels into two corresponding vectors for the training and test datasets.

# Extract the diagnosis labels for the training dataset (first 469 rows)
wbcd_train_labels <- wbcd[1:469, 1]
wbcd_test_labels
# Extract the diagnosis labels for the test dataset (rows 470 to 569)
wbcd_test_labels <- wbcd[470:569, 1]

# By separating the target labels into wbcd_train_labels and wbcd_test_labels, we prepare the data for use in the kNN model. This step ensures the model can learn 
# from the correct labels during training and be evaluated against the actual labels in the test data.

# Step 3 – Training a Model on the Data

# The kNN algorithm is a lazy learner, meaning it doesn't build a model during training.
# Instead, it simply stores the training data in a structured format for later use.
# This makes kNN straightforward to implement but computationally intensive during prediction.

# Installing and loading the 'class' package
# The 'class' package provides basic functions for classification, including kNN.
# If the package is not installed, use the following command to install it:
# > install.packages("class")

install.packages("class")

# To load the package, use:
library(class)

# Using the knn() function from the 'class' package
# The knn() function implements the k-Nearest Neighbors algorithm.
# For each instance in the test dataset, it:
# 1. Identifies the k-nearest neighbors based on Euclidean distance.
# 2. Assigns the test instance to the majority class among the k neighbors.
# 3. Resolves ties randomly if there is a tie vote among the neighbors.

# Note: The value of k (number of neighbors) is user-defined and affects the model's performance.
# A smaller k is sensitive to noise, while a larger k generalizes better.

# Alternative kNN implementations
# While the knn() function in the 'class' package works well for basic classification tasks,
# other R packages offer advanced or more efficient implementations of kNN.
# Explore the Comprehensive R Archive Network (CRAN) for additional options
# if you encounter limitations with the basic knn() function.

# The next step involves using the knn() function to classify the test data
# based on the training dataset and its corresponding labels.

# Training and classification using the knn() function is performed in a single function
# call, using four parameters as shown in the following table:

# Define the training dataset ('wbcd_train') and test dataset ('wbcd_test').
# Both datasets contain numeric features, where 'wbcd_train' is used for learning
# and 'wbcd_test' contains instances to classify.

# The labels for the training data are stored in 'wbcd_train_labels', 
# which is a factor vector indicating the class for each training instance.

# Set the value of k (number of neighbors). We use k = 21:
# - It is chosen as an odd number to avoid ties in the majority vote.
# - It is roughly equal to the square root of the training dataset size (469).

k <- 21

# Use the knn() function to perform classification:
# - 'train' specifies the training dataset.
# - 'test' specifies the dataset for which we want predictions.
# - 'cl' provides the class labels for the training data.
# - 'k' specifies the number of neighbors to consider for voting.
wbcd_test_pred <- knn(
  train = wbcd_train,
  test = wbcd_test,
  cl = wbcd_train_labels,
  k = k
)
wbcd_test_pred
# The result is stored in 'wbcd_test_pred', which is a factor vector containing
# the predicted class for each instance in the test dataset. These predictions are
# based on the majority class among the k nearest neighbors from the training data.


# Step 4: Evaluating Model Performance

# Before evaluating, ensure the 'gmodels' package is installed.
# If not already installed, uncomment the line below to install it:
# install.packages("gmodels")
install.packages("gmodels")

# Load the gmodels package, which provides the CrossTable() function
library(gmodels)

# The wbcd_test_labels vector contains the actual class labels for the test data.
# The wbcd_test_pred vector contains the predicted class labels from the kNN algorithm.

# Use the CrossTable() function to compare the actual and predicted values:
# - 'x' specifies the actual class labels (ground truth).
# - 'y' specifies the predicted class labels from the kNN model.
# - 'prop.chisq = FALSE' excludes chi-square statistics from the output since it's not needed here.
CrossTable(
  x = wbcd_test_labels,  # Actual labels for the test dataset
  y = wbcd_test_pred,    # Predicted labels from the kNN model
  prop.chisq = FALSE     # Disable chi-square statistics in the output
)
# Evaluating Model Performance with CrossTable

# The CrossTable() function provides a summary of the model's predictions.
# It creates a confusion matrix that highlights the following categories:
# - True Negatives (TN): Correct predictions where the mass is benign.
# - True Positives (TP): Correct predictions where the mass is malignant.
# - False Negatives (FN): Incorrect predictions where the mass is predicted benign but is malignant.
# - False Positives (FP): Incorrect predictions where the mass is predicted malignant but is benign.

# Example interpretation of CrossTable results:
# - TN: Top-left cell. Example: 61 out of 100 predictions were true negatives 
#       (benign masses correctly identified).
# - TP: Bottom-right cell. Example: 37 out of 100 predictions were true positives 
#       (malignant masses correctly identified).
# - FN: Lower-left cell. Example: 2 out of 100 predictions were false negatives 
#       (malignant masses incorrectly identified as benign).
# - FP: Top-right cell. If there were any, these would represent false positives 
#       (benign masses incorrectly identified as malignant).

# False Negatives (FN) can be very costly in medical diagnoses:
# - Example: A tumor classified as benign when it is actually malignant.
# - Impact: This could result in the patient believing they are cancer-free, 
#   allowing the disease to spread untreated.

# False Positives (FP) are less severe but still problematic:
# - Example: A benign mass classified as malignant.
# - Impact: Leads to unnecessary tests or treatments, financial burden, and patient stress.

# Accuracy is calculated as:
# (True Positives + True Negatives) / Total Predictions
# Example: If 98 out of 100 predictions are correct, the accuracy is 98%.

# -----------------------------------------------
# Improving Model Performance in kNN Classification
# -----------------------------------------------

# Goal: Improve the classifier by:
# 1. Trying z-score standardization instead of normalization.
# 2. Testing multiple values of k to find the optimal k.

# ----------------------
# Z-Score Standardization
# ----------------------
# Why z-score standardization?
# - Unlike normalization (scaling to 0-1 range), z-score standardization centers 
#   data around a mean of 0 and scales it based on standard deviation (SD = 1).
# - Outliers are not compressed toward the center, which can help when 
#   extreme values (e.g., malignant tumors) are meaningful.

# Using the `scale()` function to standardize numeric features:
wbcd_z <- as.data.frame(scale(wbcd[-1]))  # Exclude the first column (diagnosis)
wbcd_z
# Add a "_z" suffix to remind us that features are z-score standardized.

# Verify the transformation by checking the summary statistics of a feature:
summary(wbcd_z$area_mean)
# Expected output:
# Min.   : -1.4530 
# 1st Qu.: -0.6666 
# Median : -0.2949 
# Mean   :  0.0000 
# 3rd Qu.:  0.3632 
# Max.   :  5.2460 
# - Mean is 0, and extreme values (outside -3 to 3) are outliers.

# ----------------------
# Splitting Data
# ----------------------
# Split the z-score standardized dataset into training and test sets:
wbcd_train <- wbcd_z[1:469, ]  # Training data (first 469 rows)
wbcd_test <- wbcd_z[470:569, ] # Test data (last 100 rows)

# Define labels for training and test sets:
wbcd_train_labels <- wbcd[1:469, 1]  # Labels for training (column 1: diagnosis)
wbcd_test_labels <- wbcd[470:569, 1] # Labels for testing (column 1: diagnosis)

# ----------------------
# Running kNN
# ----------------------
# Classify the test data using kNN with k = 21:
library(class)  # Load the kNN function
wbcd_test_pred <- knn(train = wbcd_train, 
                      test = wbcd_test, 
                      cl = wbcd_train_labels, 
                      k = 21)

# ----------------------
# Evaluating Performance
# ----------------------
# Compare the predicted vs. actual test labels using CrossTable():
library(gmodels)  # Load the gmodels package
CrossTable(x = wbcd_test_labels, 
           y = wbcd_test_pred, 
           prop.chisq = FALSE)  # Remove chi-square values

# ----------------------
# Observing Results
# ----------------------
# The z-score standardization led to a slight decline in accuracy:
# - Before: 98% accuracy with normalized data.
# - After: 95% accuracy with z-score standardized data.
# - The number of false negatives (misclassified malignant tumors) remained unchanged.

# ---------------------------------------
# Testing Alternative Values of k
# ---------------------------------------
# Testing the kNN classifier with different k values:
# Below are the outcomes for different k values:
# -------------------------------------------------
# k value  | False Negatives | False Positives | % Incorrect
# -------------------------------------------------
# 1        | 1               | 3               | 4%
# 5        | 2               | 0               | 2%
# 11       | 3               | 0               | 3%
# 15       | 3               | 0               | 3%
# 21       | 2               | 0               | 2%
# 27       | 4               | 0               | 4%

# Observations:
# - A smaller k (e.g., k = 1) reduces false negatives but increases false positives.
# - A moderate k (e.g., k = 5 or k = 21) balances false positives and false negatives.

# Trade-off:
# - False negatives (failing to detect malignant tumors) are more dangerous, so a model 
#   minimizing them is preferable, even if it slightly increases false positives.

# ----------------------
# Key Insights:
# ----------------------
# 1. Z-score standardization didn't outperform normalization in this case.
# 2. Varying k values helped identify that k = 5 or k = 21 provides the best balance.
# 3. Always evaluate models across multiple k values to find the optimal one.

# ----------------------
# Future Consideration:
# ----------------------
# - Use cross-validation (repeated testing on random subsets of data) to ensure the model
#   generalizes well to unseen datasets. This avoids overfitting to the specific test set.
