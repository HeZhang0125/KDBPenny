package weka.classifiers.mmall.Online.Bayes;

import weka.classifiers.mmall.Utils.SUtils;
import weka.core.Instance;

public class GradientsUpdater_d extends GradientsUpdater {

    public GradientsUpdater_d(wdBayesOnline algorithm) {
	super(algorithm);
    }

    public void update(Instance instance, int t) {
	BayesTree forest = algorithm.dParameters_;

	BayesNode[] nodes = forest.findLeavesForInstance(instance);

	double[] probs = computeProbabilities(nodes, instance);

	// computing gradients
	forest.computeGradientForClass_d(instance, probs, algorithm.getRegularizationType(),
		algorithm.getLambda(), algorithm.getM_CenterWeights());
	for (BayesNode node : nodes) {
	    node.computeGradient_d(instance, probs, algorithm.getRegularizationType(),
		    algorithm.getLambda(), algorithm.getM_CenterWeights());
	}

	updateParameters(instance, nodes, t);

    }

    public double[] computeProbabilities(BayesNode[] nodes, Instance instance) {
	int nc = algorithm.getNc();
	double[] myProbs = new double[nc];

	BayesTree forest = algorithm.getdParameters_();

	// unboxed logDistributionForInstance_d
	for (int c = 0; c < nc; c++) {
	    myProbs[c] = forest.getClassParameter(c);
	}

	for (int u = 0; u < nodes.length; u++) {
	    BayesNode node = nodes[u];
	    int attValue = (int) instance.value(node.root.attNumber);
	    for (int c = 0; c < nc; c++) {
		myProbs[c] += node.getParameter(attValue, c);
	    }
	}

	SUtils.normalizeInLogDomain(myProbs);
	SUtils.exp(myProbs);
	return myProbs;
    }

}