package weka.classifiers.mmall.Ensemble.logDistributionComputation.D;

import weka.classifiers.mmall.DataStructure.DBL.DBLParameters;
import weka.classifiers.mmall.Ensemble.logDistributionComputation.LogDistributionComputerDBL;
import weka.core.Instance;

public class DBL5LogDistributionComputerD extends LogDistributionComputerDBL {

	public static LogDistributionComputerDBL singleton = null;
	
	protected DBL5LogDistributionComputerD() {}
	public static LogDistributionComputerDBL getComputer() {
		if(singleton==null){
			singleton = new DBL5LogDistributionComputerD();
		}
		return singleton;
	}

	@Override
	public void compute(double[] probs, DBLParameters params,Instance inst) {
		for (int c = 0; c < probs.length; c++) {
			probs[c] = params.getParameterAtFullIndex(c);
			double probsClass = 0;
			for (int att1 = 4; att1 < params.getNAttributes(); att1++) {
				int att1val = (int) inst.value(att1);

				for (int att2 = 3; att2 < att1; att2++) {
					int att2val = (int) inst.value(att2);

					for (int att3 = 2; att3 < att2; att3++) {
						int att3val = (int) inst.value(att3);

						for (int att4 = 1; att4 < att3; att4++) {
							int att4val = (int) inst.value(att4);

							for (int att5 = 0; att5 < att4; att5++) {
								int att5val = (int) inst.value(att5);

								long index = params.getAttributeIndex(att1, att1val, att2, att2val, att3, att3val, att4, att4val, att5, att5val, c);
								probsClass += params.getParameterAtFullIndex(index);
							}
						}
					}
				}
			}
			probs[c] += probsClass;
		}

	}

}
