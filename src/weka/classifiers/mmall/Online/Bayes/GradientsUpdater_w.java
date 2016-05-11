package weka.classifiers.mmall.Online.Bayes;

import weka.classifiers.mmall.Utils.SUtils;
import weka.core.Instance;

public class GradientsUpdater_w extends GradientsUpdater {

	public GradientsUpdater_w(wdBayesOnline algorithm) {
		super(algorithm);
	}

	public void update(Instance instance, int t) {
		BayesTree forest = algorithm.dParameters_;
		
		//getting the leaves
		BayesNode[] nodes = forest.findLeavesForInstance(instance);
		//computing proba for leaves
		double[] probs = computeProbabilities(nodes, instance);
		// computing gradients for leaves (with no smoothing)
		forest.computeGradientForClass_w(instance,probs,algorithm.getRegularizationType(),algorithm.getLambda(),algorithm.getM_CenterWeights());
		for (BayesNode node : nodes) {
			node.computeGradient_w(instance,probs,algorithm.getRegularizationType(),algorithm.getLambda(),algorithm.getM_CenterWeights());
		}
		//updating parameters at leaves level
		updateParameters(instance,nodes,t);
		
	}

	public double[] computeProbabilities(BayesNode []nodes, Instance instance) {
		int nc = algorithm.getNc();
		double[] myProbs = new double[nc];

		BayesTree forest = algorithm.getdParameters_();

		// unboxed logDistributionForInstance_w
		for (int c = 0; c < nc; c++) {
			myProbs[c] = forest.getClassParameter(c)*forest.getClassProbability(c);
		}

		for (int u = 0; u < nodes.length; u++) {
			BayesNode node = nodes[u];
			int attValue = (int) instance.value(node.root.attNumber);
			for (int c = 0; c < nc; c++) {
				myProbs[c] += node.getParameter(attValue,c)*node.getProbability(attValue, c);
			}
		}

		SUtils.normalizeInLogDomain(myProbs);
		SUtils.exp(myProbs);
		return myProbs;
	}



}