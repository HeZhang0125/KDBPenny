package weka.classifiers.mmall.Ensemble.logDistributionComputation.W;

import weka.classifiers.mmall.DataStructure.DBL.DBLParameters;
import weka.classifiers.mmall.Ensemble.logDistributionComputation.LogDistributionComputerDBL;
import weka.core.Instance;

public class DBL3LogDistributionComputerW extends LogDistributionComputerDBL {

	public static LogDistributionComputerDBL singleton = null;
	
	protected DBL3LogDistributionComputerW() {}
	
	public static LogDistributionComputerDBL getComputer() {
		if(singleton==null){
			singleton = new DBL3LogDistributionComputerW();
		}
		return singleton;
	}

	@Override
	public void compute(double[] probs, DBLParameters params,Instance inst) {
		
		//double w = (double) params.getNAttributes()/3.0 * 1.0/SUtils.NC3(params.getNAttributes());
		double w = 1;
		
		for (int c = 0; c < probs.length; c++) {
			probs[c] = params.getProbAtFullIndex(c) * params.getParameterAtFullIndex(c);
			double probsClass = 0;
			for (int att1 = 0; att1 < params.getNAttributes(); att1++) {
				int att1val = (int) inst.value(att1);
				
				long index = params.getAttributeIndex(att1, att1val, c);
				probs[c] += params.getProbAtFullIndex(index) * params.getParameterAtFullIndex(index);

				for (int att2 = 0; att2 < att1; att2++) {
					int att2val = (int) inst.value(att2);
					
					index = params.getAttributeIndex(att1, att1val, att2, att2val, c);
					probsClass += (params.getProbAtFullIndex(index) * params.getParameterAtFullIndex(index));

					for (int att3 = 0; att3 < att2; att3++) {
						int att3val = (int) inst.value(att3);

						index = params.getAttributeIndex(att1, att1val, att2, att2val, att3, att3val, c);
						probsClass += (params.getProbAtFullIndex(index) * params.getParameterAtFullIndex(index));							
					}
				}
			}
			probs[c] += (w * probsClass);
		}
	}

}
