# The Man and the Machines

## Abstract
Did you ever get lost in your Wikipedia session, when a chain reaction in your thoughts lead you from the discography of your favourite artist to the colonial history of Tokelau?
Over the centuries, our personal experiences and the evolution of linguistics made us able to create connections, and we are now reflecting this tendency in knowledge graphs like Wikipedia.
However, among the countless paths connecting two topics, our fallible mind often relies on our biased knowledge to create those connections: this leads us to trace paths that can appear obvious to us, but unnecessarily long or even nonsensical to a machine, which is traditionally seeking the fastest and optimal approach.
The Man and the Machine starts from human navigation paths, defined by the users playing the Wikispeedia game, to then dwell into the navigation ability of the machine, through semantic-distance-based paths: how differently will the two entities approach this same problem?

## Research questions
* How does semantic distance affect the shortest paths found by A*, when compared to other heuristics for the shortest paths?
* Can we identify any patterns or recurring structures in the human paths (ie. going for a central hub)?
* What insights can we draw from the cases where human paths outperform the machines ones? And viceversa?
* Are there any specific Wikipedia categories that are correlated with the differences in the performances?
 
## Methods
First of all, we will use NetworkX to create the Wikispeedia graph to work on. After some exploratory analysis, we will clean the dataset from outliers we do not need _(ie. unfinished paths with just one page visited and less than a one minute stay, indicating players who started a game without trying)_. _To identify trends and differences in the patterns, we will group the human paths by the category of the target node, see their how correlated the categories are with the length of the shortest paths (ANOVA test) and test the differences between the distributions._ 

To compute the semantic distances between each page, we will embed each title into a vector, through the Bert transformers, and we will use the cosine similarity as our main similarity metric. After computing our similarity matrix (whose values will only be between -1 and +1, by definition), we will define the semantic distance as $dist(x, y) = 1 - cos \textunderscore similarity (x, y) + 1$: the first two terms are essential to represent the semantic distance as the opposite of the semantic similarity (the more similar two titles are, the closer they should be in the graph), the second $+1$ is a design choice to work only with values greater than 1 and effectively implement the A* search algorithm.

As mentioned above, we will implement the A* search algorithm with the semantic distances being the weights of the edges constructed between the linked articles. We will then find the shortest paths taking into account the semantic distance between the nodes, and compare them to the shortest paths in the dataset (being the research of the shortest path based on heuristics, we can expect significant differences based on the different design choices).

Furthermore, given the multitude of navigation paths that are provided along with the Wikispeedia network, we will be able to compare the human navigation paths with the "semantic-paths": to do so, we will group the paths with the same starting node and target node by "human path is shorter", "semantic path is shorter", "both approaches have the same length". This will be useful to draw our conclusions and ultimately look for patterns within the different performances measured, with targets belonging to the same Wikipedia category sharing the same performance.

## Virtual environment
For a smooth collaboration and code execution, we created a virtual environment (included in this GitHub) called as our team name, amonavis.
To activate it, run source ``` ./amonavis/bin/activate ```

## Proposed timeline and organisation within the team
* By the end of week 1, we will finish the data cleaning divide us in two groups: one for the main descriptive statistics and testing part (Carlos, Carolina, Daniele) and one for the graph algorithms design and implementation (Nicolas, Sophea);
* by the end of week 2, we will answer the first 2 research questions, meaning we will have mastered our way through the dataset provided;
* by the end of week 3, we will answer the last 2 research questions, meaning we will have mastered the graphs algorithms and the new data generated and we will be done with our comparisons;
* by the end of week 4, we will choose the final plots and elements of the notebook to include in the blog page, for a comprehensive and exhaustive storytelling;
* by the end of week 5, we will do our final reviews of the blog and polish the design for an effective publication and presentation.

## Questions for TAs (optional): 
* We incurred in extremely slow computations when evaluating the similarities: any tips to optimize the process?

