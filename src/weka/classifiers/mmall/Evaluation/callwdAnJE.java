package weka.classifiers.mmall.Evaluation;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;

import weka.classifiers.mmall.Online.AnJE.ObjectiveFunctionOnlineCLL;
import weka.classifiers.mmall.Online.AnJE.wdAnJEOnline;
import weka.classifiers.mmall.Utils.RPC;

import weka.core.Instance;
import weka.core.Instances;
import weka.core.Utils;

import weka.core.converters.ArffLoader.ArffReader;

public class callwdAnJE {

	private static String m_TrainingData = "";
	private static String m_TestingData = "";
	private static String m_OutputResults = "";

	private static String m_S = "A1JE"; 					// -S (A1JE, A2JE, A3JE, A4JE, A5JE)
	private static String m_P = "MAP"; 						// -P (MAP, dCCBN, wCCBN, eCCBN)
	private static String m_E = "CLL"; 						// -E (CLL, MSE)
	private static String m_I = "Flat"; 					// -I (Flat, Indexed, IndexedBig, BitMap)

	private static boolean m_MVerb = false; 					// -V
	private static int m_WeightingInitialization = 0; 			// -W	

	private ObjectiveFunctionOnlineCLL function_to_optimize;

	private static boolean m_Regularization = false; 		// -R
	private static int m_Epochs = 1; 						// -A
	private static double m_Eta = 0.001; 					// -B
	private static double m_Lambda = 0.01; 					// -L 

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

		File errorFile;
		String errorFileName = m_OutputResults;
		errorFile = new File(errorFileName);
		BufferedWriter writer = new BufferedWriter(new FileWriter(errorFile));

		/*
		 * Read file sequentially, 10000 instances at a time
		 */
		FileReader fReader = new FileReader(sourceFile);
		BufferedReader reader = new BufferedReader(fReader);
		ArffReader loader = new ArffReader(reader, 10000);

		Instances structure = loader.getStructure();
		structure.setClassIndex(structure.numAttributes() - 1);
		int N = structure.numInstances();
		int nc = structure.numClasses();

		/*
		 *  Build Learner (wdAnJE)
		 */
		wdAnJEOnline learner = new wdAnJEOnline();
		learner.set_m_S(m_S);
		learner.set_m_I(m_I);
		learner.set_m_P(m_P);
		learner.setRegularization(m_Regularization);
		learner.setLambda(m_Lambda);
		learner.setStepsize(m_Eta);

		learner.buildClassifier(structure);

		/*
		 * Update Classifier
		 */
		Instance current;

		while ((current = loader.readInstance(structure)) != null) {			
			learner.updateClassifier(current);
			N = N + 1;
		}

		if (learner.needSecondPass()) {

			fReader.close(); reader.close();
			fReader = new FileReader(sourceFile);
			reader = new BufferedReader(fReader);
			loader = new ArffReader(reader, 10000);

			while ((current = loader.readInstance(structure)) != null) {			
				learner.updateAfterFirstPass(current);				
			}			
		}

		if (learner.needThirdPass()) {
			int t = 0;		
			double[][] results = new double[m_Epochs][3];

			for (int iter = 0; iter < m_Epochs; iter++) {
				fReader.close(); reader.close();
				fReader = new FileReader(sourceFile);
				reader = new BufferedReader(fReader);
				loader = new ArffReader(reader, 10000);

				while ((current = loader.readInstance(structure)) != null) {			
					learner.update_gradient(current, t, results[iter]);	
					t = t + 1;
				}	
				System.out.println("Iteration: " + iter);				
			}	

			/*
			 * Done with all the Epochs. Lets print results
			 */
			System.out.println(" ------------------------------- ");
			System.out.println("Results on Training Data");
			System.out.println(" ------------------------------- ");
			System.out.print("fxNLL = [");
			for (int i = 0; i < m_Epochs; i++) {
				System.out.print(results[i][0] + ", ");
			}
			System.out.println("];");

			System.out.print("fxError = [");
			for (int i = 0; i < m_Epochs; i++) {
				System.out.print(results[i][1]/N + ", ");
			}
			System.out.println("];");

			System.out.print("fxRMSE = [");
			for (int i = 0; i < m_Epochs; i++) {
				System.out.print(Math.sqrt(results[i][2]/N) + ", ");
			}
			System.out.println("];");

			System.out.println();
			System.out.println(" ------------------------------- ");
			System.out.println("Results on Testing Data");
			System.out.println(" ------------------------------- ");			
			if (m_TestingData.isEmpty()) {
				m_TestingData = m_TrainingData;			
			}

			File sourceFile2;
			sourceFile2 = new File(m_TestingData);
			if (!sourceFile2.exists()) {
				System.err.println("File " + m_TestingData + " not found!");
				System.exit(-1);
			}

			fReader.close(); reader.close();
			fReader = new FileReader(sourceFile2);
			reader = new BufferedReader(fReader);
			loader = new ArffReader(reader, 10000);

			double RMSE = 0;
			double error = 0;
			int NTest = 0;

			while ((current = loader.readInstance(structure)) != null) {
				NTest++;
			}
			
			double[][] rawResults = new double[NTest][2];
			
			fReader.close(); reader.close();
			fReader = new FileReader(sourceFile2);
			reader = new BufferedReader(fReader);
			loader = new ArffReader(reader, 10000);

			int index = 0;
			while ((current = loader.readInstance(structure)) != null) {
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

				rawResults[index][0] = probs[x_C];
				rawResults[index][1] = x_C;
				index++;
			}
			
			// Error and RMSE
			System.out.println("Error: " + error/NTest);
			System.out.println("RMSE: " + Math.sqrt(RMSE/NTest));

			// RPC
			int numCutPoints = 10000;
			RPC rpc = new RPC(rawResults, numCutPoints);
			
			double[][] rpcCurve = rpc.generateCurve();
			double auRPC = rpc.getAuRPC(rpcCurve);

			System.out.println("AuRPC: " + auRPC);									
		}

		fReader.close(); 
		reader.close();
		writer.close();
		System.out.println("All Done");
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

		String SW = Utils.getOption('W', options);
		if (SW.length() != 0) {
			m_WeightingInitialization = Integer.parseInt(SW);
		}

		String SK = Utils.getOption('S', options);
		if (SK.length() != 0) {
			m_S = SK;
		}

		String MP = Utils.getOption('P', options);
		if (MP.length() != 0) {
			m_P = MP;
		}	

		String ME = Utils.getOption('E', options);
		if (ME.length() != 0) {
			m_E = ME;
		}

		String MI = Utils.getOption('I', options);
		if (MI.length() != 0) {
			m_I = MI;
		}

		m_Regularization = Utils.getFlag('R', options);

		String strB = Utils.getOption('B', options);
		if (strB.length() != 0) {
			m_Eta = (Double.valueOf(strB));
		}

		String strA = Utils.getOption('A', options);
		if (strA.length() != 0) {
			m_Epochs = (Integer.valueOf(strA));
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


