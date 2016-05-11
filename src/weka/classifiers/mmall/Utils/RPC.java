package weka.classifiers.mmall.Utils;

import java.util.Arrays;
import java.util.Comparator;

public class RPC {

	private double[][] data;
	private int numCutPoints;
	private int N;
	
	public RPC(double[][] data, int numCutPoints) {
		this.data = data;
		this.numCutPoints = numCutPoints;
		
		N = data.length;
		
		Arrays.sort(data, new Comparator<double[]>() {
		    @Override
		    public int compare(double[] o1, double[] o2) {
		        return Double.compare(o2[0], o1[0]);
		    }
		});
	}

	public double[][] generateCurve() {
		double[][] rpcCurve = null;
		
		numCutPoints = 1000;
		int step = (int) Math.floor((double) N/numCutPoints);
		
		rpcCurve = new double[numCutPoints][2];
		
		for (int i = step, j = 0; i < N && j < numCutPoints; i += step, j++) {
			System.out.println("i = " + i + ", j = " + j);
			// set first i probabilities in data to 1, set remaining to zero
			resetProbabilites(i);
			
			double[] RP = new double[2]; 
			
			computePrecisionRecall(RP);
			rpcCurve[j][0] = RP[0];
			rpcCurve[j][1] = RP[1];			
		}		
		
		return rpcCurve;
	}

	private void computePrecisionRecall(double[] RP) {
		// Compute TP and FP and FN
		int TP = 0;
		int FP = 0;
		int FN = 0;
		for (int i = 0; i < N; i++) {
			if (data[i][0] == 1 && data[i][1] == 1) {
				TP += 1;
			}
			if (data[i][0] == 1 && data[i][1] == 0) {
				FP += 1;
			}
			if (data[i][0] == 0 && data[i][1] == 1) {
				FN += 1;
			}
		}
		
		RP[0] = (double) TP / (TP + FN); // recall
		
		RP[1] = TP / (TP + FP); // precision
	}

	private void resetProbabilites(int num) {
		for (int i = 0; i < N; i++) {
			if (i < num) {
				data[i][0] = 1;	
			} else {
				data[i][0] = 0;
			}			
		}	
	}

	public double getAuRPC(double[][] rpcCurve) {
		// TODO Auto-generated method stub
		return 0;
	}

}
