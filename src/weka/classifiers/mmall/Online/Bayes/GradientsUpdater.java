package weka.classifiers.mmall.Online.Bayes;

import weka.core.Instance;
import weka.core.Instances;

public abstract class GradientsUpdater {
    protected final wdBayesOnline algorithm;

    protected GradientsUpdater(wdBayesOnline algorithm) {
	this.algorithm = algorithm;
    }

    abstract public void update(Instance instance, int t);

    abstract public double[] computeProbabilities(BayesNode[] nodes, Instance instance);

    public void update(Instances m_Instances) {
	for (int i = 0; i < m_Instances.size(); i++) {
	    Instance inst = m_Instances.get(i);
	    update(inst, i);
	}
    }

    protected void updateParameters(Instance instance, BayesNode[] nodes, int t) {
	BayesTree forest = algorithm.dParameters_;
	// updating parameters with gradients
	forest.updateParametersForClass(instance, t, algorithm.getEta(), algorithm.getRegularizationType(), algorithm.getLambda());
	for (BayesNode node : nodes) {
	    node.updateParameters(instance, t, algorithm.getEta(), algorithm.getRegularizationType(), algorithm.getLambda());
	}

	// forest.resetGradientsForClass();
	for (BayesNode node : nodes) {
	    node.resetGradients(instance);
	}

    }
    
    

}
