package weka.classifiers.mmall.Online.Bayes;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.Arrays;
import java.util.Set;

import org.apache.commons.math3.random.RandomGenerator;
import org.jgrapht.experimental.dag.DirectedAcyclicGraph;
import org.jgrapht.graph.DefaultEdge;

import weka.classifiers.mmall.Bayes.chordalysis.explorer.ChordalysisEstimatedNParams;
import weka.classifiers.mmall.Bayes.chordalysis.explorer.ChordalysisKL;
import weka.classifiers.mmall.Bayes.chordalysis.explorer.ChordalysisKLMaxNParam;
import weka.classifiers.mmall.Bayes.chordalysis.explorer.ChordalysisKLMaxNParamReg;
import weka.classifiers.mmall.Bayes.chordalysis.explorer.ChordalysisKLNoLimit;
import weka.classifiers.mmall.Bayes.chordalysis.model.DecomposableModel;
import weka.classifiers.mmall.DataStructure.xxyDist;
import weka.classifiers.mmall.Utils.CorrelationMeasures;
import weka.classifiers.mmall.Utils.SUtils;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Utils;
import weka.core.converters.ArffLoader.ArffReader;

public class BNStructure {

	private int[][] m_Parents;
	private int[] m_Order;

	int nInstances;
	int nAttributes;
	int nc;
	public int[] paramsPerAtt;
	xxyDist xxyDist_;

	private String m_S = "";

	private double[] mi = null;
	private double[][] cmi = null;

	private int k = 1;
	private long maxNFreeParams = Long.MAX_VALUE;

	protected static int MAX_INCORE_N_INSTANCES = 100000;

	public BNStructure(String m_S, int k, int nAttributes) {
		this.m_S = m_S;
		this.k = k;
		this.nAttributes = nAttributes;

		m_Parents = new int[nAttributes][];
		m_Order = new int[nAttributes];

		for (int i = 0; i < nAttributes; i++) {
			m_Order[i] = i;
		}
	}

	BNStructure(Instances m_Instances, String m_S, int k) {
		this.m_S = m_S;
		this.k = k;

		nInstances = m_Instances.numInstances();
		nAttributes = m_Instances.numAttributes() - 1;

		m_Parents = new int[nAttributes][];
		m_Order = new int[nAttributes];

		for (int i = 0; i < nAttributes; i++) {
			m_Order[i] = i;
		}

		xxyDist_ = new xxyDist(m_Instances);
		if (nInstances > 0) {
			xxyDist_.addToCount(m_Instances);
		}
	}

	private void updateXXYDist(Instance instance) {
		xxyDist_.update(instance);
	}

	public void buildStructure() throws Exception {

		if (m_S.equalsIgnoreCase("NB")) {

		} else if (m_S.equalsIgnoreCase("TAN")) {

		} else if (m_S.equalsIgnoreCase("KDB")) {

		} else if (m_S.equalsIgnoreCase("Chordalysis")) {

			for (int i = 0; i < nAttributes; i++) {
				System.out.print(i + " : ");
				if (m_Parents[i] != null) {
					for (int j = 0; j < m_Parents[i].length; j++) {
						System.out.print(m_Parents[i][j] + ",");
					}
				}
				System.out.println();
			}

		} else {
			System.out.println("m_S value should be from set {NB, TAN, KDB, Chordalysis}");
		}

	}

	public int[] get_Order() {
		return m_Order;
	}

	public int[][] get_Parents() {
		return m_Parents;
	}

	public xxyDist get_XXYDist() {
		return xxyDist_;
	}

	public void setMaxNFreeParams(long maxNFreeParams) {
		this.maxNFreeParams = maxNFreeParams;
	}

	public void learnStructure(Instances structure, File sourceFile,RandomGenerator rg) throws IOException {

		// First fill xxYDist; everybody needs it
		ArffReader reader = new ArffReader(new BufferedReader(new FileReader(sourceFile)), 10000);
		Instance row;
		while ((row = reader.readInstance(structure)) != null) {
			updateXXYDist(row);
			xxyDist_.setNoData();
			xxyDist_.xyDist_.setNoData();
		}		

		switch (m_S) {
		case "NB":
			break;
		case "TAN":
			learnStructureTAN();
			break;
		case "KDB":
			learnStructureKDB();
			break;
		case "Chordalysis":
			learnStructureChordalysis(structure, sourceFile,rg);
			break;
		default: 
				System.out.println("value of m_S has to be in set {NB, TAN, KDB, Chordalysis}");
		}

		printStructure();
	}

	private void learnStructureChordalysis(Instances structure, File sourceFile, RandomGenerator rg) throws IOException {
		// Chordalysis
		
		int nInstances = xxyDist_.getNoData();
		
		double samplingRate = (nInstances <= MAX_INCORE_N_INSTANCES) ? 1.0 : 1.0 * MAX_INCORE_N_INSTANCES / nInstances;
		ArffReader reader = new ArffReader(new BufferedReader(new FileReader(sourceFile)), 10000);
		
		// Chordalysis Core
//		ChordalysisKL structureLearning = new ChordalysisKLMaxNParam(k + 1, maxNFreeParams);
		ChordalysisKLMaxNParamReg structureLearning = new ChordalysisKLMaxNParamReg(k + 1, maxNFreeParams);
//		ChordalysisEstimatedNParams structureLearning = new ChordalysisEstimatedNParams(.05);
//		ChordalysisKL structureLearning = new ChordalysisKLNoLimit(maxNFreeParams);
		structureLearning.buildModel(structure, reader, samplingRate, true);
		structureLearning.setRandomGenerator(rg);
		DecomposableModel selectedModel = structureLearning.getModel();
		structureLearning = null;

		mi = new double[nAttributes];
		cmi = new double[nAttributes][nAttributes];

		CorrelationMeasures.getMutualInformation(xxyDist_.xyDist_, mi);
		CorrelationMeasures.getCondMutualInf(xxyDist_, cmi);

		// Not SUtils; elimination ordering is the opposite of the ordering
		// we want
		int[] preferedEO = Utils.sort(mi);
		DirectedAcyclicGraph<Integer, DefaultEdge> bn = selectedModel.getBayesianNetwork(preferedEO);

		// converting the BN in Nayyar's data structure
		int[][] m_TempParents = new int[nAttributes][];
		m_Parents = new int[nAttributes][];

		for (int i = 0; i < m_Parents.length; i++) {

			Set<DefaultEdge> parents = bn.incomingEdgesOf(i);
			if (parents.size() > 0) {

				m_TempParents[i] = new int[parents.size()];
				m_Parents[i] = new int[parents.size()];

				int j = 0;
				for (DefaultEdge pEdge : parents) {
					Integer parent = bn.getEdgeSource(pEdge);
					m_TempParents[i][j] = parent;
					j++;
				}

				// Nayyar: changing order of parents based on cmi
				// ----------------------------------------------
				double[] cmi_values = new double[parents.size()];
				for (j = 0; j < parents.size(); j++) {
					cmi_values[j] = cmi[i][m_TempParents[i][j]];
				}
				int[] cmiOrder = SUtils.sort(cmi_values);

				for (j = 0; j < parents.size(); j++) {
					m_Parents[i][j] = m_TempParents[i][cmiOrder[j]];
				}
				// ----------------------------------------------
			}
		}

		// Cleaning
		bn = null;
	}

	private void learnStructureKDB() {
		// KDB

		mi = new double[nAttributes];
		cmi = new double[nAttributes][nAttributes];
		CorrelationMeasures.getMutualInformation(xxyDist_.xyDist_, mi);
		CorrelationMeasures.getCondMutualInf(xxyDist_, cmi);

		// Sort attributes on MI with the class
		m_Order = SUtils.sort(mi);

		// Calculate parents based on MI and CMI
		for (int u = 0; u < nAttributes; u++) {
			int nK = Math.min(u, k);
			if (nK > 0) {
				m_Parents[u] = new int[nK];

				double[] cmi_values = new double[u];
				for (int j = 0; j < u; j++) {
					cmi_values[j] = cmi[m_Order[u]][m_Order[j]];
				}
				int[] cmiOrder = SUtils.sort(cmi_values);

				for (int j = 0; j < nK; j++) {
					m_Parents[u][j] = m_Order[cmiOrder[j]];
				}
			}
		}

		// Update m_Parents based on m_Order
		int[][] m_ParentsTemp = new int[nAttributes][];
		for (int u = 0; u < nAttributes; u++) {
			if (m_Parents[u] != null) {
				m_ParentsTemp[m_Order[u]] = new int[m_Parents[u].length];

				for (int j = 0; j < m_Parents[u].length; j++) {
					m_ParentsTemp[m_Order[u]][j] = m_Parents[u][j];
				}
			}
		}
		for (int i = 0; i < nAttributes; i++) {
			m_Order[i] = i;
		}
		m_Parents = null;
		m_Parents = m_ParentsTemp;
		m_ParentsTemp = null;

	}

	private void learnStructureTAN() {
		// TAN 
		
		cmi = new double[nAttributes][nAttributes];
		CorrelationMeasures.getCondMutualInf(xxyDist_, cmi);

		int[] m_ParentsTemp = new int[nAttributes];
		for (int u = 0; u < nAttributes; u++) {
			m_ParentsTemp[u] = -1;
		}
		CorrelationMeasures.findMST(nAttributes, cmi, m_ParentsTemp);

		for (int u = 0; u < nAttributes; u++) {
			if (m_ParentsTemp[u] != -1) {
				m_Parents[u] = new int[1];
				m_Parents[u][0] = m_ParentsTemp[u];
			}
		}

	}

	private void printStructure() {
		// Print the structure
		System.out.println(Arrays.toString(m_Order));
		for (int i = 0; i < nAttributes; i++) {
			System.out.print(i + " : ");
			if (m_Parents[i] != null) {
				for (int j = 0; j < m_Parents[i].length; j++) {
					System.out.print(m_Parents[i][j] + ",");
				}
			}
			System.out.println();
		}
	}

}
