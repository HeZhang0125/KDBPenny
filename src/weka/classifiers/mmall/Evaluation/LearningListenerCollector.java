package weka.classifiers.mmall.Evaluation;

import java.io.File;
import java.util.ArrayList;

import weka.classifiers.mmall.Online.Bayes.LearningListener;

public class LearningListenerCollector implements LearningListener {
    int nInstancesStep;
    int nErrors;
    int nTested;
    double se;
    ArrayList<Double> errorRates;
    ArrayList<Double> rmses;

    public LearningListenerCollector(int howOften) {
	this.nInstancesStep = howOften;
	errorRates = new ArrayList<>();
	rmses = new ArrayList<>();
    }

    @Override
    public void updated(long nInstancesSoFar, double CLL, double sePartial, double error) {
	nTested++;
	this.se += sePartial;
	this.nErrors += (int) error;

	if (nTested % nInstancesStep == 0) {
	    double rmse = Math.sqrt(this.se / nTested);
	    double errorRate = 1.0 * nErrors / nTested;
	    errorRates.add(errorRate);
	    rmses.add(rmse);
	    this.se = 0.0;
	    this.nTested = 0;
	    this.nErrors = 0;
	}
    }
    
    public ArrayList<Double> getErrorRates(){
	return errorRates;
    }
    
    public ArrayList<Double> getRMSEs(){
	return rmses;
    }

}
