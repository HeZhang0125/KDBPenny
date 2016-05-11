package weka.classifiers.mmall.DataStructure;

import java.util.concurrent.Callable;

import weka.classifiers.mmall.Utils.SUtils;
import weka.core.Instance;
import weka.core.Instances;

public class CallableXXXYDist implements Callable<Double>{

	private Instances instances;
	private int start;
	private int stop;

	private int threadID;
	
	private int XYCount[][][][];

	public CallableXXXYDist(int start, int stop, Instances data, int[][][][] XYCount, int th) {
		this.start = start;
		this.stop = stop;
		this.XYCount = XYCount;
		this.threadID = th;
	}

	@Override
	public Double call() throws Exception {
		
		int numProcessed = 0;
		for (int i = start; i <= stop; i++) {
			Instance instance = instances.instance(i);
			int x_C = (int) instance.classValue();

			
			numProcessed++;
			if ((numProcessed % 1000) == 0) {
				//System.out.print(perfOutput.charAt(threadID));
				System.out.print(SUtils.perfOutput.charAt(threadID));
			}
		}
		
		return 0.0;
	}

}
