BWS Argument Similarity Corpus
-------------------------------------

The BWS Argument Similarity Corpus includes 
3,400 sentence pairs for 8 controversial topics
with 425 argument pairs each for every topic.

These 8 controversial topics are -  

- Cloning
- Abortion
- Minimum wage
- Marijuana legalization
- Nuclear energy
- Death penalty
- Gun control
- School uniforms

The original arguments are taken from the dataset released 
by Stab et al. [1]. For details of the sampling process
please refer to the main paper. The argument similarity 
scores are continuous and have been normalized in the range 
of [0,1], where values close to 0 show that the arguments 
stem from the same topic but do not cover the same aspect 
or they have different topics and values close to 1 show 
that they mean exactly the same thing in different words or
they cover the same aspect and only differ in minor details.


Each line includes the following information:
* id: identifier of the sentence pair
* argument1: field for the first argument
* argument2: field for the second argument
* topic: the topic used to retrieve the documents
* score: the continous argument similarity score 
normalised in the range of [0,1]