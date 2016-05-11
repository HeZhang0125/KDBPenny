package weka.classifiers.mmall.Ensemble.logDistributionComputation.W;

import weka.classifiers.mmall.DataStructure.DBL.DBLParameters;
import weka.classifiers.mmall.Ensemble.logDistributionComputation.LogDistributionComputerDBL;
import weka.core.Instance;

public class DBL1LogDistributionComputerW extends LogDistributionComputerDBL {

	public static LogDistributionComputerDBL singleton = null;
	
	protected DBL1LogDistributionComputerW() {}
	public static LogDistributionComputerDBL getComputer() {
		if(singleton==null) {
			singleton = new DBL1LogDistributionComputerW();
		}
		return singleton;
	}

	@Override
	public void compute(double[] probs, DBLParameters params,Instance inst) {
			
		for (int c = 0; c < probs.length; c++) {
			probs[c] = params.getProbAtFullIndex(c) * params.getParameterAtFullIndex(c);
			for (int att1 = 0; att1 < params.getNAttributes(); att1++) {
				int att1val = (int) inst.value(att1);

				long index = params.getAttributeIndex(att1, att1val, c);
				probs[c] += params.getProbAtFullIndex(index) * params.getParameterAtFullIndex(index);
			}
		}
	}

}
