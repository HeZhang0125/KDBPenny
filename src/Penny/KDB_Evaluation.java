package Penny;

import java.util.Random;

import weka.classifiers.AbstractClassifier;
import weka.classifiers.Classifier;
import weka.classifiers.evaluation.Evaluation;
import weka.classifiers.evaluation.output.prediction.AbstractOutput;
import weka.core.Instances;

public class KDB_Evaluation extends Evaluation {

	/** For serialization */
	private static final long serialVersionUID = -7010314486866816271L;

	public KDB_Evaluation(Instances data) throws Exception {
		super(data);
	}
	
	public double crossValidateModel(KDBPenny classifier, Instances data, int numFolds, Random random,
			Object... forPredictionsPrinting) throws Exception {

		// Make a copy of the data we can reorder
		data = new Instances(data);
		data.randomize(random);
		if (data.classAttribute().isNominal()) {
			data.stratify(numFolds);
		}

		// We assume that the first element is a
		// weka.classifiers.evaluation.output.prediction.AbstractOutput object
		AbstractOutput classificationOutput = null;
		if (forPredictionsPrinting.length > 0) {
			// print the header first
			classificationOutput = (AbstractOutput) forPredictionsPrinting[0];
			classificationOutput.setHeader(data);
			classificationOutput.printHeader();
		}
		
		double[] valueN0 = { 0, 0.5, 1, 1.5, 2, 5, 10, 15, 20, 50, 100, 200 ,1000};
		double bestParameter = 0;
		
		double minRMSE = Double.MAX_VALUE;
		
		for (int i = 0; i < numFolds; i++) {
			Instances train = data.trainCV(numFolds, i, random);
			setPriors(train);
			KDBPenny copiedClassifier = (KDBPenny) AbstractClassifier.makeCopy(classifier);	
			Instances test = data.testCV(numFolds, i);
			System.out.println(copiedClassifier.m_KDB);
			System.out.println(copiedClassifier.getLaplacePseudocount());
			
			for(int j =0; j < valueN0.length; j++){
				
				copiedClassifier.setN0(valueN0[j]);
				copiedClassifier.buildClassifier(train);	
				evaluateModel(copiedClassifier, test, forPredictionsPrinting);
				double rmse = rootMeanSquaredError();
				
				if (rmse < minRMSE) {
					minRMSE = rmse;
					bestParameter = valueN0[j];
				}
			}
		}
		m_NumFolds = numFolds;

		if (classificationOutput != null) {
			classificationOutput.printFooter();
		}
		return bestParameter;
	}
}
