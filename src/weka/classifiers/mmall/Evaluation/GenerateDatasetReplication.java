package weka.classifiers.mmall.Evaluation;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;

import org.apache.commons.math3.random.MersenneTwister;
import org.apache.commons.math3.random.RandomGenerator;

public class GenerateDatasetReplication {

	private static void printHeader(PrintWriter arff,int nReplicationsX1){
		arff.println("@relation noisy-xor\n");
		arff.println("@attribute x1 {0,1}");
		arff.println("@attribute x2 {0,1}");
		for (int i = 0; i < nReplicationsX1; i++) {
			arff.println("@attribute x1rep"+(i+1)+" {0,1}");
		}
		arff.println("@attribute class {0,1}");
		arff.println();
		arff.println("@data");
	}

	public static void main(String[] args) throws IOException {
		//parameters
		int nReplicationsX1 = 500;
		int nInstances = 10000;

		double priorClass = 0.5;
		double amountNoiseX1 = .05;
		double amountNoiseX2 = .25;

		long seed = 3071980;
		//===========

		//File arffFile = File.createTempFile("DataReplication-", ".arff");
		File arffFile = new File("/Users/nayyar/WExperiments/11_synExpWANBIAC/DatasetB500.arff");

		RandomGenerator rg = new MersenneTwister();
		rg.setSeed(seed);

		PrintWriter arff = new PrintWriter(new BufferedWriter(new FileWriter(arffFile),1000000));
		printHeader(arff, nReplicationsX1);

		for (int i = 0; i < nInstances; i++) {
		    
		    boolean y = rg.nextDouble()<priorClass;
		    boolean x1  = (rg.nextDouble()<amountNoiseX1)?!y:y;
		    boolean x2  = (rg.nextDouble()<amountNoiseX2)?!y:y;
		    arff.print((x1?1:0)+",");
		    arff.print((x2?1:0)+",");
		    for (int r = 0; r < nReplicationsX1; r++) {
			 arff.print((x1?1:0)+",");
		    }
		    arff.println((y?1:0));

		}

		arff.flush();
		arff.close();
		System.out.println("File written to "+arffFile.getAbsolutePath());
	}

}
