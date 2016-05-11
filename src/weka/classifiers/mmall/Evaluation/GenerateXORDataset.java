package weka.classifiers.mmall.Evaluation;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;

import org.apache.commons.math3.random.MersenneTwister;
import org.apache.commons.math3.random.RandomGenerator;

public class GenerateXORDataset {

	private static void printHeader(PrintWriter arff,int nCovariates){
		arff.println("@relation noisy-xor\n");
		for (int i = 0; i < nCovariates; i++) {
			arff.println("@attribute x"+(i+1)+" {0,1}");
		}
		arff.println("@attribute class {0,1}");
		arff.println();
		arff.println("@data");
	}

	public static void main(String[] args) throws IOException {
		//parameters
		int nCovariates = 100;
		int nInstances = 10000;

		//double probORvsXOR = 0.1;
		//double priorProbCovariatesOR = .01;
		//double priorProbCovariatesXOR = .4;

		double probORvsXOR = 0;
		double priorProbCovariatesOR = .007;
		double priorProbCovariatesXOR = .5;

		long seed = 3071980;
		//===========

		//File arffFile = File.createTempFile("xor", ".arff");
		File arffFile = new File("/Users/nayyar/WExperiments/11_synExpWANBIAC/Dataset_XOR.arff");

		RandomGenerator rg = new MersenneTwister();
		rg.setSeed(seed);

		PrintWriter arff = new PrintWriter(new BufferedWriter(new FileWriter(arffFile),1000000));
		printHeader(arff, nCovariates);

		boolean[]values = new boolean[nCovariates];

		int[] numORClasses = new int[2];
		int[] numXORClasses = new int[2];

		for (int i = 0; i < nInstances; i++) {

			boolean isORInstance = rg.nextDouble()<probORvsXOR;
			double priorProbCovariates = (isORInstance)?priorProbCovariatesOR:priorProbCovariatesXOR;

			boolean isORtrue = false;
			int nCovTrue = 0;
			//sample xs
			for (int x = 0; x < nCovariates; x++) {
				values[x] = (rg.nextDouble()<priorProbCovariates);
				arff.print((values[x]?1:0)+",");

				//aggregating statistics to know if y will be true
				isORtrue = isORtrue||values[x];
				if (values[x]) {
					nCovTrue++;
				}
			}

			if (isORInstance) {
				arff.println((isORtrue?1:0));
				if (isORtrue) 
					numORClasses[0]++;
				else 
					numORClasses[1]++;
			} else {//xor
				arff.println(((nCovTrue%2==1)?1:0));
				if (nCovTrue%2==1) 
					numXORClasses[0]++;
				else
					numXORClasses[1]++;
			}
		}

		arff.flush();
		arff.close();
		System.out.println("Number of OR (0): " + numORClasses[0] + " and OR (1): " + numORClasses[1]);
		System.out.println("Number of XOR (0): " + numXORClasses[0] + " and XOR (1): " + numXORClasses[1]);
		System.out.println("File written to "+arffFile.getAbsolutePath());
	}

}
