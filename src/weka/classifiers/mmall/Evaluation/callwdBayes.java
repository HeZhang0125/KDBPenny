package weka.classifiers.mmall.Evaluation;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.Reader;
import java.util.ArrayList;

import org.apache.commons.math3.random.MersenneTwister;

import weka.classifiers.mmall.Online.Bayes.RegularizationType;
import weka.classifiers.mmall.Online.Bayes.wdBayesOnline;
import weka.classifiers.mmall.Utils.RPC;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Utils;
import weka.core.converters.ArffLoader.ArffReader;

public class callwdBayes {

	private static String m_TrainingData = "";
	private static String m_TestingData = "";
	private static String m_OutputResults = "";

	private static String m_S = "NB"; 					// -S (NB, TAN, KDB, Chordalysis)
	private static String m_P = "MAP"; 					// -P (MAP, dCCBN, wCCBN, eCCBN)
	
	private static int m_K = 1; 						// -K
	
	private static boolean m_MVerb = false; 			// -V	

	private static RegularizationType m_Regularization = RegularizationType.None; 		// -R
	private static int m_Epochs = 1; 						// -A
	private static double m_Eta = 0.001; 					// -B
	private static double m_Lambda = 0.01; 					// -L 
	
	private static long m_Chordalysis_Mem = Long.MAX_VALUE; 	// -F (in thousands of free parameters)
	
	private static Instances instances = null;

	public static void main(String[] args) throws Exception {

		setOptions(args);

		if (m_TrainingData.isEmpty()) {
			System.err.println("No Training File given");
			System.exit(-1);			
		}

		File sourceFile;
		sourceFile = new File(m_TrainingData);
		if (!sourceFile.exists()) {
			System.err.println("File " + m_TrainingData + " not found!");
			System.exit(-1);
		}

		/*
		 * Read file sequentially, 10000 instances at a time
		 */
		ArffReader reader = new ArffReader(new BufferedReader(new FileReader(sourceFile)), 10000);

		Instances structure = reader.getStructure();
		structure.setClassIndex(structure.numAttributes() - 1);
		int nc = structure.numClasses();

		/*
		 *  Build Learner (wdBayes)
		 */
		wdBayesOnline learner = new wdBayesOnline();
		learner.set_m_S(m_S);
		learner.set_m_P(m_P);
		learner.setRegularization(m_Regularization);
		learner.setLambda(m_Lambda);
		learner.setK(m_K);
		learner.setNEpochs(m_Epochs);
		learner.setMaxNParameters(m_Chordalysis_Mem);
		learner.setRandomGenerator(new MersenneTwister(3071980));

		
		learner.buildClassifier(sourceFile);
		
		/*
		 * Done with learning (including everything)
		 * Now test
		 */			
		System.out.println();
		System.out.println(" ------------------------------- ");
		System.out.println("Results on Testing Data");
		System.out.println(" ------------------------------- ");			
		if (m_TestingData.isEmpty()) {
			m_TestingData = m_TrainingData;			
		}

		File sourceFileTest;
		sourceFileTest = new File(m_TestingData);
		if (!sourceFileTest.exists()) {
			System.err.println("File " + m_TestingData + " not found!");
			System.exit(-1);
		}

		reader = new ArffReader(new BufferedReader(new FileReader(sourceFileTest)), 10000);

		double RMSE = 0;
		double error = 0;
		int NTest = 0;
		ArrayList<double[]> rawResults = new ArrayList<double[]>();
		
		Instance current;
		while ((current = reader.readInstance(structure)) != null) {
			double[] probs = new double[nc];
			probs = learner.distributionForInstance(current);
			int x_C = (int) current.classValue();
			
			// ------------------------------------
			// Update Error and RMSE
			// ------------------------------------
			int pred = -1;
			double bestProb = Double.MIN_VALUE;
			for (int y = 0; y < nc; y++) {
				if (!Double.isNaN(probs[y])) {
					if (probs[y] > bestProb) {
						pred = y;
						bestProb = probs[y];
					}
					RMSE += (1/(double)nc * Math.pow((probs[y]-((y == x_C) ? 1 : 0)), 2));
				} else {
					System.err.println("probs[ " + y + "] is NaN! oh no!");
				}
			}				

			if (pred != x_C) {
				error += 1;
			}
			rawResults.add(new double[]{probs[x_C],x_C});
			NTest++;
		}
		
		// Error and RMSE
		System.out.println("Error: " + error/NTest);
		System.out.println("RMSE: " + Math.sqrt(RMSE/NTest));

//		// RPC
//		int numCutPoints = 10000;
//		double[][] tabRawResults = rawResults.toArray(new double[0][0]);
//		RPC rpc = new RPC(tabRawResults, numCutPoints);
//		
//		double[][] rpcCurve = rpc.generateCurve();
//		double auRPC = rpc.getAuRPC(rpcCurve);
//
//		System.out.println("AuRPC: " + auRPC);									
	}


	public static int ind(int i, int j) {
		return (i == j) ? 1 : 0;
	}

	public static void setOptions(String[] options) throws Exception {

		String Strain = Utils.getOption('t', options);
		if (Strain.length() != 0) {
			m_TrainingData = Strain;
		}

		String Stest = Utils.getOption('T', options);
		if (Stest.length() != 0) {
			m_TestingData = Stest;
		}

		String Soutput = Utils.getOption('O', options);
		if (Soutput.length() != 0) {
			m_OutputResults = Soutput;
		}

		m_MVerb = Utils.getFlag('V', options);

		String Mem = Utils.getOption('F', options);
		if (Mem.length() != 0) {
			m_Chordalysis_Mem = Long.parseLong(Mem);
		}
		
		String SK = Utils.getOption('S', options);
		if (SK.length() != 0) {
			m_S = SK;
		}

		String MP = Utils.getOption('P', options);
		if (MP.length() != 0) {
			m_P = MP;
		}
		
		String MK = Utils.getOption('K', options);
		if (MK.length() != 0) {
			m_K = Integer.parseInt(MK);
		}
		
		String MR = Utils.getOption('R', options);
		if (MR.length() != 0) {
		    switch (MR) {
		    case "L1":
			m_Regularization = RegularizationType.L1;
			break;
		    case "L2":
			m_Regularization = RegularizationType.L2;
		    default:
			m_Regularization = RegularizationType.None;
		    }
		} else {
		    m_Regularization = RegularizationType.None;
		}
		
		String strA = Utils.getOption('A', options);
		if (strA.length() != 0) {
			m_Epochs = (Integer.valueOf(strA));
		}
		
		String strB = Utils.getOption('B', options);
		if (strB.length() != 0) {
			m_Eta = (Double.valueOf(strB));
		}

		String strL = Utils.getOption('L', options);
		if (strL.length() != 0) {
			m_Lambda = (Double.valueOf(strL));
		}

		Utils.checkForRemainingOptions(options);

	}

	public static String[] getOptions() {
		String[] options = new String[3];
		int current = 0;
		while (current < options.length) {
			options[current++] = "";
		}
		return options;
	}

}


