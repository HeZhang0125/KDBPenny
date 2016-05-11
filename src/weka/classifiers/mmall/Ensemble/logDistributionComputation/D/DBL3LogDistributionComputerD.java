package weka.classifiers.mmall.Ensemble.logDistributionComputation.D;

import weka.classifiers.mmall.DataStructure.DBL.DBLParameters;
import weka.classifiers.mmall.Ensemble.logDistributionComputation.LogDistributionComputerDBL;
import weka.core.Instance;

public class DBL3LogDistributionComputerD extends LogDistributionComputerDBL {

	public static LogDistributionComputerDBL singleton = null;
	
	protected DBL3LogDistributionComputerD(){}
	public static LogDistributionComputerDBL getComputer() {
		if(singleton==null){
			singleton = new DBL3LogDistributionComputerD();
		}
		return singleton;
	}

	@Override
	public void compute(double[] probs, DBLParameters params,Instance inst) {
		for (int c = 0; c < probs.length; c++) {
			probs[c] = params.getParameterAtFullIndex(c);
			double probsClass = 0;
			for (int att1 = 2; att1 < params.getNAttributes(); att1++) {
				int att1val = (int) inst.value(att1);

				for (int att2 = 1; att2 < att1; att2++) {
					int att2val = (int) inst.value(att2);

					for (int att3 = 0; att3 < att2; att3++) {
						int att3val = (int) inst.value(att3);

						long index = params.getAttributeIndex(att1, att1val, att2, att2val, att3, att3val, c);
						probsClass += params.getParameterAtFullIndex(index);									
					}
				}
			}
			probs[c] += probsClass;
		}
	}

}
