# Top Quark Reconstruction as a Combinatorial Classification Problem

The reconstruction of hadronically decaying top quarks can be formulated as a combinatorial classification problem. Rather than relying solely on kinematic fitting or heuristic selection rules, the task can be framed in a supervised learning context: among all possible jet combinations in an event, identify the triplet that corresponds to the true top quark decay.

## Physics Motivation

In hadronic top decay  
\[
t \to Wb \to q q' b,
\]  
the final state consists of three jets. In an idealized situation, one would simply group these three jets and reconstruct the top quark. In realistic LHC events, however, more than three jets are typically reconstructed due to additional QCD radiation, underlying event activity, and pileup. Crucially, reconstructed jets are not labeled by origin. Therefore, we do not know a priori which three jets originate from the top decay.

This leads to a combinatorial problem: given \(N\) reconstructed jets in an event, we can form \(\binom{N}{3}\) possible jet triplets. Only one of these (in signal events) corresponds to the true top quark decay. All others are combinatorial background.

## Turning Reconstruction into Classification

The strategy is to treat each jet triplet as a candidate object and to classify triplets as either:

- **Signal:** the true triplet corresponding to the top decay  
- **Background:** incorrect (combinatorial) triplets  

Monte Carlo truth information allows us to label triplets during training. For each event:

- The correctly matched triplet is added to the signal sample.
- All other triplets in the same event are added to the background sample.

Aggregating over many events produces a labeled dataset of true and fake triplets. We then train a classifier—such as a boosted decision tree (BDT) or a neural network—to distinguish between them.

## Feature Construction

Effective separation can already be achieved using simple, physics-motivated observables. For each jet triplet, we construct:

1. The three pairwise angular separations  
   \[
   \Delta R_{ij}
   \]
2. The three pairwise invariant masses normalized to the total triplet mass  
   \[
   \frac{m_{ij}}{m_{\mathrm{triplet}}}
   \]

These six variables encode both the angular geometry and internal mass structure of the candidate system. Together, they capture the expected kinematic correlations of a true top decay while remaining robust and interpretable.

In practice, training a BDT with these features already provides strong discrimination between correct and incorrect triplets.

## Inference Procedure

Once the classifier is trained, top reconstruction proceeds as follows for any new event:

1. Construct all possible jet triplets.
2. Compute the feature set for each triplet.
3. Evaluate the classifier score for every candidate.
4. Select the highest-scoring triplet as the reconstructed top quark.

This approach converts a combinatorial ambiguity into a ranking problem, where the classifier learns to identify the physically consistent triplet.

## Conceptual Perspective

This framework emphasizes a general principle: object reconstruction in complex hadronic environments can be viewed as structured classification over candidate subsets. By combining physics-motivated observables with supervised learning, one can systematically resolve combinatorial ambiguities while retaining interpretability and control.

Top quark reconstruction thus serves as a prototype for applying machine learning to combinatorial problems in collider physics.