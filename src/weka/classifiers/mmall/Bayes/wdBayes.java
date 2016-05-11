/*
 * MMaLL: An open source system for learning from very large data
 * Copyright (C) 2014 Nayyar A Zaidi, Francois Petitjean and Geoffrey I Webb
 * 
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program. If not, see <http://www.gnu.org/licenses/>.
 *
 * Please report any bugs to Nayyar Zaidi <nayyar.zaidi@monash.edu>
 */

/*
 * wdBayes Classifier
 * 
 * wdBayes.java     
 * Code written by: Nayyar Zaidi, Francois Petitjean
 * 
 * Options:
 * -------
 * 
 * -t /Users/nayyar/WData/datasets_DM/shuttle.arff
 * -V -M 
 * -S "chordalysis"
 * -K 2
 * -P "wCCBN"
 * -W 1
 * 
 */
package weka.classifiers.mmall.Bayes;

import java.util.Arrays;
import java.util.List;
import java.util.Set;

import lbfgsb.Minimizer;
import lbfgsb.Result;
import lbfgsb.StopConditions;

import org.eclipse.recommenders.jayes.util.Graph;
import org.eclipse.recommenders.jayes.util.triangulation.GraphElimination;
import org.eclipse.recommenders.jayes.util.triangulation.MinFillIn;
import org.jgrapht.experimental.dag.DirectedAcyclicGraph;
import org.jgrapht.graph.DefaultEdge;

import weka.classifiers.AbstractClassifier;
import weka.classifiers.bayes.BayesNet;
import weka.classifiers.bayes.net.search.SearchAlgorithm;
import weka.classifiers.mmall.Bayes.chordalysis.explorer.ChordalysisKL;
import weka.classifiers.mmall.Bayes.chordalysis.graph.ChordalGraph;
import weka.classifiers.mmall.Bayes.objectiveFunction.ObjectiveFunctionCLL;
import weka.classifiers.mmall.Bayes.objectiveFunction.parallel.ParallelObjectiveFunctionCLL_d;
import weka.classifiers.mmall.Bayes.objectiveFunction.parallel.ParallelObjectiveFunctionCLL_w;
import weka.classifiers.mmall.Bayes.objectiveFunction.parallel.ParallelObjectiveFunctionCLL_e;
import weka.classifiers.mmall.Bayes.objectiveFunction.ObjectiveFunctionCLL_d;
import weka.classifiers.mmall.Bayes.objectiveFunction.ObjectiveFunctionCLL_w;
import weka.classifiers.mmall.Bayes.objectiveFunction.ObjectiveFunctionCLL_e;
import weka.classifiers.mmall.DataStructure.Bayes.wdBayesNode;
import weka.classifiers.mmall.DataStructure.Bayes.wdBayesParametersTree;
import weka.classifiers.mmall.DataStructure.xxyDist;
import weka.classifiers.mmall.Utils.CorrelationMeasures;
import weka.classifiers.mmall.Utils.SUtils;
import weka.core.Capabilities;
import weka.core.Capabilities.Capability;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.OptionHandler;
import weka.core.Utils;
import weka.filters.supervised.attribute.Discretize;

public class wdBayes extends AbstractClassifier implements OptionHandler {

	private static final long serialVersionUID = 4823531716976859217L;

	private Instances m_Instances;

	int nInstances;
	int nAttributes;
	int nc;
	public int[] paramsPerAtt;
	xxyDist xxyDist_;

	private Discretize m_Disc = null;

	public wdBayesParametersTree dParameters_;

	private int[][] m_Parents;
	private int[] m_Order;

	private String m_S = "NB"; // -S (1: NB, 2:TAN, 3:KDB, 4:BN, 5:Chordalysis, 6:Saturated)
	private String m_P = "MAP"; // -P (1: MAP, 2:dCCBN, 3:wCCBN, 4:eCCBN)

	private int m_KDB = 1; // -K
	private int m_BN = 10; // -K
	private int m_Chordalysis = 12; // -K

	private boolean m_MVerb = false; // -V
	private boolean m_Discretization = false; // -D 
	private boolean m_TriangulateGraph = false; // -X 
	private boolean m_MultiThreaded = false; // -M

	private boolean m_Regularization = false; 			// -R
	private double m_Lambda = 0.01; 					// -L 

	private ObjectiveFunctionCLL function_to_optimize;
	private double maxGradientNorm = 0.000000000000000000000000000000001;
	//private int m_MaxIterations = 10000;
	private int m_MaxIterations = 200;

	private int m_WeightingInitialization = 0; 	        // -W 0, -1 for initialization to MAP estimates

	private double[] probs;

	@Override
	public void buildClassifier(Instances instances) throws Exception {

		// can classifier handle the data?
		getCapabilities().testWithFail(instances);

		// Discretize instances if required
		if (m_Discretization) {
			m_Disc = new Discretize();
			m_Disc.setInputFormat(instances);
			instances = weka.filters.Filter.useFilter(instances, m_Disc);
		}

		// remove instances with missing class
		m_Instances = new Instances(instances);
		m_Instances.deleteWithMissingClass();
		nAttributes = m_Instances.numAttributes() - 1;
		nc = m_Instances.numClasses();

		probs = new double[nc];
		nInstances = m_Instances.numInstances();

		paramsPerAtt = new int[nAttributes];
		for (int u = 0; u < nAttributes; u++) {
			paramsPerAtt[u] = m_Instances.attribute(u).numValues();
		}

		m_Parents = new int[nAttributes][];
		m_Order = new int[nAttributes];
		for (int i = 0; i < nAttributes; i++) {
			getM_Order()[i] = i;
		}

		double[] mi = null;
		double[][] cmi = null;

		/*
		 * Initialize structure array based on m_S
		 */
		if (m_S.equalsIgnoreCase("NB")) {
			// NB

			xxyDist_ = new xxyDist(m_Instances);
			xxyDist_.addToCount(m_Instances);

		} else if (m_S.equalsIgnoreCase("TAN")) {
			// TAN

			xxyDist_ = new xxyDist(m_Instances);
			xxyDist_.addToCount(m_Instances);

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

			for (int i = 0; i < nAttributes; i++) {
				System.out.print(i + " : ");
				if (m_Parents[i] != null) {
					for (int j = 0; j < m_Parents[i].length; j++) {
						System.out.print(m_Parents[i][j] + ",");
					}
				}
				System.out.println();
			}

		} else if (m_S.equalsIgnoreCase("Saturated")) {
			// Saturated

			xxyDist_ = new xxyDist(m_Instances);
			xxyDist_.addToCount(m_Instances);

			for (int u = 0; u < nAttributes; u++) {
				m_Parents[u] = new int[nAttributes - u - 1];

				int index = (u + 1);
				for (int j = 0; j < m_Parents[u].length; j++) {
					m_Parents[u][j] = index;
					index++;
				}
			}

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

		} else if (m_S.equalsIgnoreCase("KDBSaturated")) {
			// KDBSaturated

			xxyDist_ = new xxyDist(m_Instances);
			xxyDist_.addToCount(m_Instances);

			mi = new double[nAttributes];
			cmi = new double[nAttributes][nAttributes];
			CorrelationMeasures.getMutualInformation(xxyDist_.xyDist_, mi);
			CorrelationMeasures.getCondMutualInf(xxyDist_, cmi);

			// Sort attributes on MI with the class
			m_Order = SUtils.sort(mi);

			// Calculate parents based on MI and CMI
			for (int u = 0; u < nAttributes; u++) {
				int nK = Math.min(u, m_KDB);
				if (nK > 0) {
					m_Parents[u] = new int[nK];
					int[] m_TempParents = new int[nK];

					double[] cmi_values = new double[u];
					for (int j = 0; j < u; j++) {
						cmi_values[j] = cmi[m_Order[u]][m_Order[j]];
					}
					int[] cmiOrder = SUtils.sort(cmi_values);

					if (u > 2) {
						// u > 2, make sure add a link if it does not 
						// violated the perfectness property of the graph
						for (int j = 0; j < nK;j++) {
							m_TempParents[j] = m_Order[cmiOrder[j]];
						}
						m_Parents[u] = SUtils.CheckForPerfectness(m_TempParents, m_Parents, m_Order);											

					} else {
						// u =< 2, just add the link
						for (int j = 0; j < nK; j++) {
							m_Parents[u][j] = m_Order[cmiOrder[j]];
						}	
					}	

				}
			}			
		}
		else if (m_S.equalsIgnoreCase("KDB")) {
			// KDB

			xxyDist_ = new xxyDist(m_Instances);
			xxyDist_.addToCount(m_Instances);

			mi = new double[nAttributes];
			cmi = new double[nAttributes][nAttributes];
			CorrelationMeasures.getMutualInformation(xxyDist_.xyDist_, mi);
			CorrelationMeasures.getCondMutualInf(xxyDist_, cmi);

			// Sort attributes on MI with the class
			m_Order = SUtils.sort(mi);

			// Calculate parents based on MI and CMI
			for (int u = 0; u < nAttributes; u++) {
				int nK = Math.min(u, m_KDB);
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
				getM_Order()[i] = i;
			}
			m_Parents = null;
			m_Parents = m_ParentsTemp;
			m_ParentsTemp = null;

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

			if (m_KDB > 1 && m_TriangulateGraph) {

				Graph jayesGraph = new Graph(nAttributes);
				ChordalGraph chordGraph = new ChordalGraph();

				for (int u = 0; u < nAttributes; u++) {
					chordGraph.addVertex(u);
				}

				for (int u = 0; u < nAttributes; u++) {
					if (m_Parents[u] != null) {
						int np = m_Parents[u].length; //BN.getNrOfParents(u);

						int[] tempParents = new int[np];
						tempParents = m_Parents[u]; //BN.getParentSet(u).getParents();

						// Get rid of class attribute as its parent
						m_Parents[u] = updateTempParents(m_Instances.classIndex(), tempParents, np);

						// Jayes - reproducing the edges of the BN
						for (int parentNo : m_Parents[u]) {
							jayesGraph.addEdge(u, parentNo);
							chordGraph.addEdgeEvenIfNotChordal(u, parentNo);
						}

						// Jayes - moralizing the graph
						for (int i = 0; i < m_Parents[m_Order[u]].length; i++) {
							for (int j = i + 1; j < m_Parents[u].length; j++) {
								jayesGraph.addEdge(m_Parents[u][i], m_Parents[u][j]);
								chordGraph.addEdgeEvenIfNotChordal(m_Parents[u][i], m_Parents[u][j]);
							}
						}
					}
				}

				System.out.println("Before triangulation");
				for (int i = 0; i < nAttributes; i++) {
					System.out.print(i + " : ");
					if (m_Parents[i] != null) {
						for (int j = 0; j < m_Parents[i].length; j++) {
							System.out.print(m_Parents[i][j] + ",");
						}
					}
					System.out.println();
				}

				//	if (!m_P.equalsIgnoreCase("MAP")) {
				System.out.println("moralized graph");
				System.out.println(chordGraph);

				// Jayes - triangulating the graph
				// weighting the nodes by number of outcomes
				double[] nodesWeights = new double[nAttributes];
				for (int i = 0; i < nAttributes; i++) {
					nodesWeights[i] = m_Instances.attribute(i).numValues();
				}

				GraphElimination triangulate = new GraphElimination(jayesGraph,	nodesWeights, new MinFillIn());
				for (List<Integer> clique : triangulate) {
					for (int i = 0; i < clique.size(); i++) {
						for (int j = i + 1; j < clique.size(); j++) {
							jayesGraph.addEdge(clique.get(i), clique.get(j));
							chordGraph.addEdgeEvenIfNotChordal(clique.get(i), clique.get(j));
						}
					}
				}

				System.out.println("tiangulated graph");
				System.out.println(chordGraph);

				DirectedAcyclicGraph<Integer, DefaultEdge> triangulatedBN = chordGraph.getBayesianNetwork();

				// recording the actual parents now
				for (int i = 0; i < m_Parents.length; i++) {

					Set<DefaultEdge> parents = triangulatedBN.incomingEdgesOf(i);
					if (parents.size() > 0) {

						m_Parents[i] = new int[parents.size()];

						int j = 0;
						for (DefaultEdge pEdge : parents) {
							Integer parent = triangulatedBN.getEdgeSource(pEdge);
							m_Parents[i][j] = parent;
							j++;
						}

					} else {
						m_Parents[i] = null;
					}
				}

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

		} else if (m_S.equalsIgnoreCase("BN")) {
			// BN

			xxyDist_ = new xxyDist(m_Instances);
			xxyDist_.addToCount(m_Instances);

			BayesNet BN = new BayesNet();
			BN.m_Instances = getM_Instances();
			SearchAlgorithm newSearchAlgorithm = new weka.classifiers.bayes.net.search.local.HillClimber();

			String[] options = new String[3];
			options[0] = "-mbc";
			options[1] = "-P";
			options[2] = m_BN + "";
			newSearchAlgorithm.setOptions(options);

			BN.setSearchAlgorithm(newSearchAlgorithm);

			BN.initStructure(m_Instances.numAttributes());
			BN.buildStructure();

			if (m_TriangulateGraph) {

				Graph jayesGraph = new Graph(nAttributes);
				ChordalGraph chordGraph = new ChordalGraph();

				for (int u = 0; u < nAttributes; u++) {
					chordGraph.addVertex(u);
				}

				for (int u = 0; u < nAttributes; u++) {
					int np = BN.getNrOfParents(u);

					int[] tempParents = new int[np];
					tempParents = BN.getParentSet(u).getParents();

					// Get rid of class attribute as its parent
					m_Parents[u] = updateTempParents(m_Instances.classIndex(), tempParents, np);

					// Jayes - reproducing the edges of the BN
					for (int parentNo : m_Parents[u]) {
						jayesGraph.addEdge(u, parentNo);
						chordGraph.addEdgeEvenIfNotChordal(u, parentNo);
					}

					// Jayes - moralizing the graph
					for (int i = 0; i < m_Parents[u].length; i++) {
						for (int j = i + 1; j < m_Parents[u].length; j++) {
							jayesGraph.addEdge(m_Parents[u][i], m_Parents[u][j]);
							chordGraph.addEdgeEvenIfNotChordal(m_Parents[u][i], m_Parents[u][j]);
						}
					}
				}

				System.out.println("Before triangulation");
				for (int i = 0; i < nAttributes; i++) {
					System.out.print(i + " : ");
					if (m_Parents[i] != null) {
						for (int j = 0; j < m_Parents[i].length; j++) {
							System.out.print(m_Parents[i][j] + ",");
						}
					}
					System.out.println();
				}

				//	if (!m_P.equalsIgnoreCase("MAP")) {
				System.out.println("moralized graph");
				System.out.println(chordGraph);

				// Jayes - triangulating the graph
				// weighting the nodes by number of outcomes
				double[] nodesWeights = new double[nAttributes];
				for (int i = 0; i < nAttributes; i++) {
					nodesWeights[i] = m_Instances.attribute(i).numValues();
				}

				GraphElimination triangulate = new GraphElimination(jayesGraph,	nodesWeights, new MinFillIn());
				for (List<Integer> clique : triangulate) {
					for (int i = 0; i < clique.size(); i++) {
						for (int j = i + 1; j < clique.size(); j++) {
							jayesGraph.addEdge(clique.get(i), clique.get(j));
							chordGraph.addEdgeEvenIfNotChordal(clique.get(i), clique.get(j));
						}
					}
				}

				System.out.println("tiangulated graph");
				System.out.println(chordGraph);

				DirectedAcyclicGraph<Integer, DefaultEdge> triangulatedBN = chordGraph.getBayesianNetwork();

				// recording the actual parents now
				for (int i = 0; i < m_Parents.length; i++) {

					Set<DefaultEdge> parents = triangulatedBN.incomingEdgesOf(i);
					if (parents.size() > 0) {

						m_Parents[i] = new int[parents.size()];

						int j = 0;
						for (DefaultEdge pEdge : parents) {
							Integer parent = triangulatedBN.getEdgeSource(pEdge);
							m_Parents[i][j] = parent;
							j++;
						}

					} else {
						m_Parents[i] = null;
					}
				}

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

				for (int u = 0; u < nAttributes; u++) {
					int np = BN.getNrOfParents(u);

					int[] tempParents = new int[np];
					tempParents = BN.getParentSet(u).getParents();

					// Get rid of class attribute as its parent
					tempParents = updateTempParents(m_Instances.classIndex(), tempParents, np);

					m_Parents[u] = new int[tempParents.length];
					for (int j = 0; j < tempParents.length; j++) {					
						m_Parents[u][j] = tempParents[j];
					}
				}			

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

		} else if (m_S.equalsIgnoreCase("Chordalysis")) {
			// Chordalysis

			xxyDist_ = new xxyDist(m_Instances);
			xxyDist_.addToCount(m_Instances);


			mi = new double[nAttributes];
			cmi = new double[nAttributes][nAttributes];
			CorrelationMeasures.getMutualInformation(xxyDist_.xyDist_, mi);
			CorrelationMeasures.getCondMutualInf(xxyDist_, cmi);
			//Not SUtils; elimination ordering is the opposite of the ordering we want
			int[] preferedEO = Utils.sort(mi);

			// Chordalysis code
			Instances instancesWithoutClass = new Instances(instances);
			int classIndex = instancesWithoutClass.classAttribute().index();
			instancesWithoutClass.setClassIndex(0);
			instancesWithoutClass.deleteAttributeAt(classIndex);

			ChordalysisKL structureLearning = new ChordalysisKL(m_Chordalysis + 1);
			structureLearning.buildModel(instancesWithoutClass);
			DirectedAcyclicGraph<Integer, DefaultEdge> bn = structureLearning.getModel().getBayesianNetwork(preferedEO);

			// Cleaning Chordalysis temp structures
			instancesWithoutClass.delete();
			instancesWithoutClass = null;
			structureLearning = null;


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
			System.out.println("m_S value should be from set {NB, TAN, KDB, BN, Chordalysis, Saturated}");
		}

		/* 
		 * ----------------------------------------------------------------------------------------
		 * Start Parameter Learning Process
		 * ----------------------------------------------------------------------------------------
		 */
		Minimizer alg = new Minimizer();
		StopConditions sc = alg.getStopConditions();
		sc.setMaxGradientNorm(maxGradientNorm);
		sc.setMaxIterations(m_MaxIterations);
		//alg.setCorrectionsNo(15);
		//alg.setDebugLevel(1);
		Result result;

		/*
		 * Parameter Learning Scheme
		 */
		if (m_P.equalsIgnoreCase("MAP")) {
			// MAP
			dParameters_ = new wdBayesParametersTree(m_Instances, paramsPerAtt, m_Order, m_Parents, 1);

			// Update dTree_ based on parents
			for (int i = 0; i < nInstances; i++) {
				Instance instance = getM_Instances().instance(i);
				dParameters_.update(instance);
			}

			// Convert counts to Probability
			xxyDist_.countsToProbs();
			dParameters_.countsToProbability(xxyDist_);

			System.out.print(dParameters_.getNLL_MAP(m_Instances, getXxyDist_().xyDist_) + ", ");
			
			dParameters_.printProbabilities();

//			System.out.println();
//			for (int c = 0; c < nc; c++) {
//				System.out.print(xxyDist_.xyDist_.getClassCount(c) + ", ");
//			}
//			for (int u = 0; u < nAttributes; u++) {
//				for (int uval = 0; uval < paramsPerAtt[u]; uval++) {
//					for (int c = 0; c < nc; c++) {
//						System.out.print("P(x_" + u + "=" + uval + " | y = " + c + ") = " + xxyDist_.xyDist_.getCount(u, uval, c) + ",   " );
//					}
//					System.out.println();
//				}
//			}

		} else if (m_P.equalsIgnoreCase("dCCBN")) {
			// dCCBN
			dParameters_ = new wdBayesParametersTree(m_Instances, paramsPerAtt, m_Order, m_Parents, 2);

			// Update dTree_ based on parents
			for (int i = 0; i < nInstances; i++) {
				Instance instance = m_Instances.instance(i);
				dParameters_.initialize(instance);
			}
			dParameters_.allocate();
			dParameters_.reset();

			// Convert counts to Probability
			xxyDist_.countsToProbs();
			dParameters_.countsToProbability(xxyDist_);			

			dParameters_.initializeParameters(m_WeightingInitialization, xxyDist_.xyDist_.getClassProbs());

			// Use for testing the initialization, with zero iterations, and seeing if it resorts to MAP
			//dParameters_.copyParameters(dParameters_.getParameters());

			/*
			 * Discriminative training - Call a gradient-based routine
			 */
			if (m_MultiThreaded) {
				function_to_optimize = new ParallelObjectiveFunctionCLL_d(this);
			} else {
				function_to_optimize = new ObjectiveFunctionCLL_d(this);
			}

			if (isM_MVerb()) {
				System.out.println();
				System.out.print("fx = [");
				System.out.print(dParameters_.getNLL_dCCBN(m_Instances) + ", ");
				alg.setIterationFinishedListener((p,nll,g)->{System.out.print(nll+", "); return true;});

				result = alg.run(function_to_optimize, dParameters_.getParameters());
				System.out.println("];");

				System.out.println("NoIter = " + result.iterationsInfo.iterations);
				System.out.println();
			} else {
				result = alg.run(function_to_optimize, dParameters_.getParameters());
				System.out.println("NoIter = " + result.iterationsInfo.iterations);
			}
			function_to_optimize.finish();

		} else if (m_P.equalsIgnoreCase("wCCBN")) {
			// wCCBN
			dParameters_ = new wdBayesParametersTree(m_Instances,paramsPerAtt, m_Order, m_Parents,  2);

			// Update dTree_ based on parents
			for (int i = 0; i < nInstances; i++) {
				Instance instance = getM_Instances().instance(i);
				dParameters_.initialize(instance);
			}
			dParameters_.allocate();
			dParameters_.reset();

			// Convert counts to Probability
			xxyDist_.countsToProbs();
			dParameters_.countsToProbability(xxyDist_);

			if (m_WeightingInitialization == -1)
				m_WeightingInitialization = 1;

			dParameters_.initializeParameters(m_WeightingInitialization, xxyDist_.xyDist_.getClassProbs());

			// Use for testing the initialization, with zero iterations, and seeing if it resorts to MAP
			//dParameters_.copyParameters(dParameters_.getParameters());

			/*
			 * Discriminative training - Call a gradient-based routine
			 */
			if (m_MultiThreaded) {
				function_to_optimize = new ParallelObjectiveFunctionCLL_w(this);
			} else {
				function_to_optimize = new ObjectiveFunctionCLL_w(this);
			}

			if (isM_MVerb()) {
				System.out.println();
				System.out.print("fx = [");

				System.out.print(dParameters_.getNLL_wCCBN(m_Instances, getXxyDist_().xyDist_) + ", ");

				alg.setIterationFinishedListener((p,nll,g)->{System.out.print(nll+", "); return true;});
				result = alg.run(function_to_optimize, dParameters_.getParameters());
				System.out.println("];");

				System.out.println("NoIter = " + result.iterationsInfo.iterations);
				System.out.println();
			} else {
				result = alg.run(function_to_optimize, dParameters_.getParameters());
				System.out.println("NoIter = " + result.iterationsInfo.iterations);
			}
			function_to_optimize.finish();

		} else if (m_P.equalsIgnoreCase("eCCBN")) {
			// eCCBN
			dParameters_ = new wdBayesParametersTree(m_Instances,paramsPerAtt, m_Order, m_Parents, 2);

			// Update dTree_ based on parents
			for (int i = 0; i < nInstances; i++) {
				Instance instance = getM_Instances().instance(i);
				dParameters_.initialize(instance);
			}
			dParameters_.allocate();
			dParameters_.reset();

			// Convert counts to Probability
			xxyDist_.countsToProbs();
			dParameters_.countsToProbability(xxyDist_);

			dParameters_.initializeParameters(m_WeightingInitialization, xxyDist_.xyDist_.getClassProbs());

			// Use for testing the initialization, with zero iterations, and seeing if it resorts to MAP
			//dParameters_.copyParameters(dParameters_.getParameters());

			/*
			 * Discriminative training - Call a gradient-based routine
			 */
			if (m_MultiThreaded) {
				function_to_optimize = new ParallelObjectiveFunctionCLL_e(this);
			} else {
				function_to_optimize = new ObjectiveFunctionCLL_e(this);
			}

			if (isM_MVerb()) {
				System.out.println();
				System.out.print("fx = [");
				System.out.print(dParameters_.getNLL_eCCBN(m_Instances) + ", ");
				alg.setIterationFinishedListener((p,nll,g)->{System.out.print(nll+", "); return true;});
				result = alg.run(function_to_optimize, dParameters_.getParameters());
				System.out.println("];");

				System.out.println("NoIter = " + result.iterationsInfo.iterations);
				System.out.println();
			} else {
				result = alg.run(function_to_optimize, dParameters_.getParameters());
				System.out.println("NoIter = " + result.iterationsInfo.iterations);
			}
			function_to_optimize.finish();


		} else {
			System.out.println("m_P value should be from set {MAP,dCCBN,wCCBN,eCCBN}");
		}

		// free up some space
		m_Instances = new Instances(m_Instances, 0);
	}

	private int[] updateTempParents(int classIndex, int[] tempParents, int np) {

		int len = 0;
		for (int i = 0; i < np; i++) {
			if (tempParents[i] != classIndex)
				len++;
		}

		int[] bufferArray = new int[len];
		int j = 0;
		for (int i = 0; i < np; i++) {
			if (tempParents[i] != classIndex) {
				bufferArray[j] = tempParents[i];
				j++;
			}
		}

		tempParents = new int[len];
		for (int i = 0; i < len; i++) {
			tempParents[i] = bufferArray[i];
		}

		return tempParents;
	}

	@Override
	public double[] distributionForInstance(Instance instance) {

		double[] probs = null;

		if (m_P.equalsIgnoreCase("MAP")) {
			// MAP
			probs = logDistributionForInstance_MAP(instance);
		} else if (m_P.equalsIgnoreCase("dCCBN")) {
			// dCCBN
			probs = logDistributionForInstance_d(instance);

		} else if (m_P.equalsIgnoreCase("wCCBN")) {
			// wCCBN
			probs = logDistributionForInstance_w(instance);

		} else if (m_P.equalsIgnoreCase("eCCBN")) {
			// eCCBN
			probs = logDistributionForInstance_e(instance);

		} else {
			System.out.println("m_P value should be from set {MAP, dCCBN, wCCBN, eCCBN}");
		}

		SUtils.exp(probs);
		return probs;
	}

	public double[] logDistributionForInstance_MAP(Instance instance) {

		double[] probs = new double[nc];

		for (int c = 0; c < nc; c++) {
			probs[c] = xxyDist_.xyDist_.pp(c);
		}

		for (int u = 0; u < nAttributes; u++) {
			wdBayesNode bNode = dParameters_.getBayesNode(instance, u);
			for (int c = 0; c < nc; c++) {
				probs[c] += bNode.getXYCount((int) instance.value(m_Order[u]),	c);
			}
		}

		SUtils.normalizeInLogDomain(probs);
		return probs;
	}

	public double[] logDistributionForInstance_d(Instance instance) {

		double[] probs = new double[nc];

		for (int c = 0; c < nc; c++) {
			probs[c] = dParameters_.getClassParameter(c);
		}

		for (int u = 0; u < nAttributes; u++) {
			wdBayesNode bNode = dParameters_.getBayesNode(instance, u);
			for (int c = 0; c < nc; c++) {
				probs[c] += bNode.getXYParameter((int) instance.value(m_Order[u]), c);
			}
		}

		SUtils.normalizeInLogDomain(probs);
		return probs;
	}

	public double[] logDistributionForInstance_e(Instance instance) {

		double[] probs = new double[nc];

		for (int c = 0; c < nc; c++) {
			probs[c] = dParameters_.getClassParameter(c);
		}

		for (int u = 0; u < nAttributes; u++) {
			wdBayesNode bNode = dParameters_.getBayesNode(instance, u);
			for (int c = 0; c < nc; c++) {
				probs[c] += bNode.getXYParameter((int) instance.value(m_Order[u]), c);
			}
		}

		SUtils.normalizeInLogDomain(probs);
		return probs;
	}

	public double[] logDistributionForInstance_w(Instance instance) {

		double[] probs = new double[nc];

		for (int c = 0; c < nc; c++) {
			probs[c] = xxyDist_.xyDist_.pp(c) * dParameters_.getClassParameter(c);
		}

		for (int u = 0; u < nAttributes; u++) {
			wdBayesNode bNode = dParameters_.getBayesNode(instance, u);
			for (int c = 0; c < nc; c++) {
				probs[c] += bNode.getXYParameter((int) instance.value(m_Order[u]), c) * bNode.getXYCount((int) instance.value(m_Order[u]),	c);
			}
		}

		SUtils.normalizeInLogDomain(probs);
		return probs;
	}

	protected void findNodesForInstance(wdBayesNode[] nodes, Instance instance) {
		for (int u = 0; u < nAttributes; u++) {
			nodes[u] = dParameters_.getBayesNode(instance, u);
		}
	}

	public static void findNodesForInstance(wdBayesNode[] nodes,
			final Instance instance, final wdBayesParametersTree dParameters) {
		for (int u = 0; u < nodes.length; u++) {
			nodes[u] = dParameters.getBayesNode(instance, u);
		}
	}

	// ----------------------------------------------------------------------------------
	// Weka Functions
	// ----------------------------------------------------------------------------------

	@Override
	public Capabilities getCapabilities() {
		Capabilities result = super.getCapabilities();
		// attributes
		result.enable(Capability.NOMINAL_ATTRIBUTES);
		// class
		result.enable(Capability.NOMINAL_CLASS);
		// instances
		result.setMinimumNumberInstances(0);
		return result;
	}

	@Override
	public void setOptions(String[] options) throws Exception {
		m_MVerb = Utils.getFlag('V', options);
		m_Discretization = Utils.getFlag('D', options);
		m_TriangulateGraph = Utils.getFlag('X', options);
		m_MultiThreaded = Utils.getFlag('M', options);

		//m_WeightingInitialization = Utils.getFlag('W', options);
		String SW = Utils.getOption('W', options);
		if (SW.length() != 0) {
			// m_S = Integer.parseInt(SK);
			m_WeightingInitialization = Integer.parseInt(SW);
		}

		String SK = Utils.getOption('S', options);
		if (SK.length() != 0) {
			// m_S = Integer.parseInt(SK);
			m_S = SK;
		}

		String MK = Utils.getOption('K', options);
		if (MK.length() != 0) {
			m_KDB = Integer.parseInt(MK);
			m_BN = Integer.parseInt(MK);
			m_Chordalysis = Integer.parseInt(MK);
		}

		String MP = Utils.getOption('P', options);
		if (MP.length() != 0) {
			// m_P = Integer.parseInt(MP);
			m_P = MP;
		}

		m_Regularization = Utils.getFlag('R', options);

		String strL = Utils.getOption('L', options);
		if (strL.length() != 0) {
			m_Lambda = (Double.valueOf(strL));
		}

		Utils.checkForRemainingOptions(options);
	}

	@Override
	public String[] getOptions() {
		String[] options = new String[3];
		int current = 0;
		while (current < options.length) {
			options[current++] = "";
		}
		return options;
	}

	public static void main(String[] argv) {
		runClassifier(new wdBayes(), argv);
	}

	public boolean getRegularization() {
		return m_Regularization;
	}

	public double getLambda() {
		return m_Lambda;
	}

	public int getNInstances() {
		return nInstances;
	}

	public int getNc() {
		return nc;
	}

	public xxyDist getXxyDist_() {
		return xxyDist_;
	}

	public wdBayesParametersTree getdParameters_() {
		return dParameters_;
	}

	public Instances getM_Instances() {
		return m_Instances;
	}

	public int[] getM_Order() {
		return m_Order;
	}

	public boolean isM_MVerb() {
		return m_MVerb;
	}

	public int getnAttributes() {
		return nAttributes;
	}

}
