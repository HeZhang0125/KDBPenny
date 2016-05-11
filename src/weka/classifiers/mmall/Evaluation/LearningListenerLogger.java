package weka.classifiers.mmall.Evaluation;

import java.io.BufferedOutputStream;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.PrintWriter;
import java.util.ArrayList;

import weka.classifiers.mmall.Online.Bayes.LearningListener;

public class LearningListenerLogger implements LearningListener {
	PrintWriter out;
	int nInstancesStep;
	int nErrors;
	int nTested;
	double se; 
	boolean verbose;


	public LearningListenerLogger(File logFile,int howOften, boolean verbose) {
		try {
			nInstancesStep = howOften;
			this.verbose = verbose;
			out = new PrintWriter(new BufferedOutputStream(new FileOutputStream(logFile)));
			out.println("t,RMSE,error rate");
			if(verbose){
				System.out.println("t,RMSE,error rate");
			}
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		}
	}

	@Override
	public void updated(long nInstancesSoFar, double CLL, double sePartial, double error) {
		nTested++;
		this.se+=sePartial;
		this.nErrors+=(int)error;
//		System.out.println(nTested+"-"+nErrors+"-"+(1.0*nErrors/nTested));

		if(nInstancesSoFar%nInstancesStep==0){
			double rmse = Math.sqrt(this.se/nTested);
			double errorRate = 1.0*nErrors/nTested;
			out.println(nInstancesSoFar+","+rmse+","+errorRate);
			if(verbose){
				System.out.println(nInstancesSoFar+","+rmse+","+errorRate);
			}
			this.se = 0.0;
			nTested=0;
			this.nErrors=0;
		}
	}

	public void finish(){
		out.flush();
		out.close();
	}

	@Override
	public ArrayList<Double> getErrorRates() {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public ArrayList<Double> getRMSEs() {
		// TODO Auto-generated method stub
		return null;
	}

}
