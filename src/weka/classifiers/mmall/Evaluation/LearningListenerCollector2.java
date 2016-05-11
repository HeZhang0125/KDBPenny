package weka.classifiers.mmall.Evaluation;

import java.io.File;
import java.util.ArrayList;

import weka.classifiers.mmall.Online.Bayes.LearningListener;

public class LearningListenerCollector2 implements LearningListener {
	int nInstancesStep;
	int nErrors;
	int nTested;

	double se;

	double[] errorBuffer;
	double[] seBuffer;

	ArrayList<Double> errorRates;
	ArrayList<Double> rmses;

	public LearningListenerCollector2(int howOften) {
		this.nInstancesStep = howOften;

		errorBuffer = new double[nInstancesStep];
		seBuffer = new double[nInstancesStep];

		errorRates = new ArrayList<>();
		rmses = new ArrayList<>();
	}

	@Override
	public void updated(long nInstancesSoFar, double CLL, double sePartial, double error) {

		if (nTested >= (2 * nInstancesStep)) {
			//if (nTested % nInstancesStep == 0) { 
			errorRates.add(getAverage(errorBuffer));
			rmses.add(getAverage(seBuffer));
			//}
		}

		int index = nTested % nInstancesStep;

		errorBuffer[index] = error;
		seBuffer[index] = sePartial;

		nTested++;
	}

	public double getAverage(double[] array) {
		double avg  = 0.0;
		int lengthArray = array.length;
		for (int i = 0; i < lengthArray; i++) {
			avg += array[i];
		}
		return Math.sqrt(avg/lengthArray);
	}

	public ArrayList<Double> getErrorRates(){
		return errorRates;
	}

	public ArrayList<Double> getRMSEs(){
		return rmses;
	}

}
