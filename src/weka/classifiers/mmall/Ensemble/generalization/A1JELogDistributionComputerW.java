package weka.classifiers.mmall.Ensemble.generalization;

import weka.core.Instance;

public class A1JELogDistributionComputerW extends LogDistributionComputerAnJE {

	public static LogDistributionComputerAnJE singleton = null;
	
	protected A1JELogDistributionComputerW(){}
	public static LogDistributionComputerAnJE getComputer() {
		if(singleton==null){
			singleton = new A1JELogDistributionComputerW();
		}
		return singleton;
	}

	@Override
	public void compute(double[] probs, wdAnJEParameters params,Instance inst) {
			
		for (int c = 0; c < probs.length; c++) {
			probs[c] = params.getProbAtFullIndex(c) * params.getParameterAtFullIndex(c);
			for (int att1 = 0; att1 < params.getNAttributes(); att1++) {
				int att1val = (int) inst.value(att1);

				long index1 = params.getAttributeIndex(att1, att1val, c);
				int index2 = params.getAttributeIndexIndex(att1);
				probs[c] += params.getProbAtFullIndex(index1) * params.getParameterAtFullIndex(index2);
			}
		}
	}

}
