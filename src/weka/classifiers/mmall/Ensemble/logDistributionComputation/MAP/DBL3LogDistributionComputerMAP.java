package weka.classifiers.mmall.Ensemble.logDistributionComputation.MAP;

import weka.classifiers.mmall.DataStructure.DBL.DBLParameters;
import weka.classifiers.mmall.Ensemble.logDistributionComputation.LogDistributionComputerDBL;
import weka.classifiers.mmall.Utils.SUtils;
import weka.core.Instance;

public class DBL3LogDistributionComputerMAP extends LogDistributionComputerDBL {

	public static LogDistributionComputerDBL singleton = null;
	
	protected DBL3LogDistributionComputerMAP(){}
	public static LogDistributionComputerDBL getComputer() {
		if(singleton==null){
			singleton = new DBL3LogDistributionComputerMAP();
		}
		return singleton;
	}

	@Override
	public void compute(double[] probs, DBLParameters params,Instance inst) {
		
		
		double alpha_w1 = 0;
		double alpha_w2 = 0.01;
		double alpha_w3 = 1 - alpha_w1 - alpha_w2;
		
		double w1 = 1;
		double w2 = params.getNAttributes()/2.0 * 1.0/SUtils.NC2(params.getNAttributes());
		double w3 = params.getNAttributes()/3.0 * 1.0/SUtils.NC3(params.getNAttributes());		
		
		for (int c = 0; c < probs.length; c++) {
			
			probs[c] = params.getProbAtFullIndex(c);
			
			double probsClass = 0;
			for (int att1 = 0; att1 < params.getNAttributes(); att1++) {
				int att1val = (int) inst.value(att1);
				
				long index = params.getAttributeIndex(att1, att1val, c);
				probsClass += (alpha_w1 * w1 * params.getProbAtFullIndex(index));

				for (int att2 = 0; att2 < att1; att2++) {
					int att2val = (int) inst.value(att2);
					
					index = params.getAttributeIndex(att1, att1val, att2, att2val, c);
					probsClass += (alpha_w2 * w2 * params.getProbAtFullIndex(index));

					for (int att3 = 0; att3 < att2; att3++) {
						int att3val = (int) inst.value(att3);

						index = params.getAttributeIndex(att1, att1val, att2, att2val, att3, att3val, c);
						probsClass += (alpha_w3 * w3 * params.getProbAtFullIndex(index));							
					}
				}
			}
			
			probs[c] += (probsClass);
		}
	}

}
