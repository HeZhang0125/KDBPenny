package weka.classifiers.mmall.DataStructure.Bayes;

import weka.classifiers.mmall.DataStructure.xxyDist;
import weka.classifiers.mmall.DataStructure.xyDist;
import weka.classifiers.mmall.Utils.SUtils;
import weka.core.Instance;
import weka.core.Instances;

import Penny.KDBPenny;

public class wdBayesParametersTree {

	public double[] parameters;
	public int np;

	public wdBayesNode[] wdBayesNode_;
	public int[] activeNumNodes;

	public static Instances instances;
	public int n; // num of attributes
	public int nc; // num of classes
	public static int nInstances;// num of instances

	public int[] m_ParamsPerAtt;
	public int[] order;
	public int[][] parents;

	public int scheme;
	double[][] parentProbs;
	/**
	 * Constructor called by wdBayes
	 */
	public wdBayesParametersTree(Instances data,int[] paramsPerAtt, int[] m_Order, int[][] m_Parents, int m_P) {

		instances = data;
		nInstances = instances.numInstances();
		this.n = instances.numAttributes() - 1;
		this.nc = instances.numClasses();
		scheme = m_P;

		m_ParamsPerAtt = new int[n]; // num of values of each attributes
		for (int u = 0; u < n; u++) {
			m_ParamsPerAtt[u] = paramsPerAtt[u];
		}

		order = new int[n];
		parents = new int[n][];

		for (int u = 0; u < n; u++) {
			order[u] = m_Order[u];
		}

		activeNumNodes = new int[n];

		for (int u = 0; u < n; u++) {
			if (m_Parents[u] != null) {
				parents[u] = new int[m_Parents[u].length];
				for (int p = 0; p < m_Parents[u].length; p++) {
					parents[u][p] = m_Parents[u][p];
				}
			}
		}

		wdBayesNode_ = new wdBayesNode[n];
		for (int u = 0; u < n; u++) {
			wdBayesNode_[u] = new wdBayesNode(scheme);
			wdBayesNode_[u].init(nc, paramsPerAtt[m_Order[u]]);
			// wdBayesNode_[u].init(nc, paramsPerAtt[u]);
		}
	}

	/*
	 * -------------------------------------------------------------------------
	 * ---------------- Update count statistics that is: relevant ***xyCount***
	 * in every node
	 * -------------------------------------------------------------------------
	 * ----------------
	 */

	public void update(Instance instance) {
		for (int u = 0; u < n; u++) {
			updateAttributeTrie(instance, u, order[u], parents[u]);
		}
	}

	public void updateAttributeTrie(Instance instance, int i, int u, int[] lparents) {

		int x_C = (int) instance.classValue();
		int x_u = (int) instance.value(u);

		wdBayesNode_[i].incrementXYCount(x_u, x_C);

		if (lparents != null) {

			wdBayesNode currentdtNode_ = wdBayesNode_[i];

			for (int d = 0; d < lparents.length; d++) {
				int p = lparents[d];

				if (currentdtNode_.att == -1 || currentdtNode_.children == null) {
					currentdtNode_.children = new wdBayesNode[m_ParamsPerAtt[p]];
					currentdtNode_.att = p;
				}

				int x_up = (int) instance.value(p);
				currentdtNode_.att = p;

				// the child has not yet been allocated, so allocate it
				if (currentdtNode_.children[x_up] == null) {
					currentdtNode_.children[x_up] = new wdBayesNode(scheme);
					currentdtNode_.children[x_up].init(nc, m_ParamsPerAtt[u]);
				}

				currentdtNode_.children[x_up].incrementXYCount(x_u, x_C);
				currentdtNode_ = currentdtNode_.children[x_up];
			}
		}
	}
	
	/*
	 * -------------------------------------------------------------------------
	 * ---------------- Convert count into probabilities
	 * -------------------------------------------------------------------------
	 * ----------------
	 */

	public void countsToProbability(xxyDist xxyDist_) {
		for (int u = 0; u < n; u++) {
			convertCountToProbs(order[u], parents[u], wdBayesNode_[u]);
		}
	}

	public void convertCountToProbs(int u, int[] lparents, wdBayesNode pt) {
		int att = pt.att;

		if (att == -1) {// leaf node
			int[][] tempArray = new int[m_ParamsPerAtt[u]][nc];
			for (int y = 0; y < nc; y++) {
				for (int uval = 0; uval < m_ParamsPerAtt[u]; uval++) {
					tempArray[uval][y] = (int) pt.getXYCount(uval, y);
				}
			}
			for (int y = 0; y < nc; y++) {
				for (int uval = 0; uval < m_ParamsPerAtt[u]; uval++) {
					int denom = 0;
					for (int dval = 0; dval < m_ParamsPerAtt[u]; dval++) {
						denom += tempArray[dval][y];
					}
					double prob = KDBPenny.laplace(tempArray[uval][y], denom, m_ParamsPerAtt[u]);
					pt.setXYProbability(uval, y, prob);
				}
			}
			return;
		}

		while (att != -1) {
			/*
			 * Now convert non-leaf node counts to probs
			 */
			int[][] tempArray = new int[m_ParamsPerAtt[u]][nc];
			for (int y = 0; y < nc; y++) {
				for (int uval = 0; uval < m_ParamsPerAtt[u]; uval++) {
					tempArray[uval][y] = (int) pt.getXYCount(uval, y);
				}
			}
			for (int y = 0; y < nc; y++) {
				for (int uval = 0; uval < m_ParamsPerAtt[u]; uval++) {
					int denom = 0;
					for (int dval = 0; dval < m_ParamsPerAtt[u]; dval++) {
						denom += tempArray[dval][y];
					}

					double prob = KDBPenny.laplace(tempArray[uval][y], denom, m_ParamsPerAtt[u]);
					pt.setXYProbability(uval, y, prob);
				}
			}

			int numChildren = pt.children.length;
			for (int c = 0; c < numChildren; c++) {
				wdBayesNode next = pt.children[c];
				if (next != null)
					convertCountToProbs(u, lparents, next);
				// Flag end of nodes
				att = -1;
			}
		}
		return;
	}

	/*
	 * -------------------------------------------------------------------------
	 * ---------------- xyCount are updated and xyParameters are initialized to
	 * zero
	 * -------------------------------------------------------------------------
	 * ----------------
	 */

	public void initialize(Instance instance) {
		for (int u = 0; u < n; u++) {
			updateTrieStructure(instance, u, order[u], parents[u]);
		}
	}

	public void updateTrieStructure(Instance instance, int i, int u, int[] lparents) {
		int x_C = (int) instance.classValue();
		int x_u = (int) instance.value(u);

		wdBayesNode_[i].setXYParameter(x_u, x_C, 0);
		wdBayesNode_[i].incrementXYCount(x_u, x_C);

		if (lparents != null) {

			wdBayesNode currentdtNode_ = wdBayesNode_[i];

			for (int d = 0; d < lparents.length; d++) {
				int p = lparents[d];

				if (currentdtNode_.att == -1 || currentdtNode_.children == null) {
					currentdtNode_.children = new wdBayesNode[m_ParamsPerAtt[p]];
					currentdtNode_.att = p;
					currentdtNode_.index = -1;
				}

				int x_up = (int) instance.value(p);
				currentdtNode_.att = p;
				currentdtNode_.index = -1;

				// the child has not yet been allocated, so allocate it
				if (currentdtNode_.children[x_up] == null) {
					currentdtNode_.children[x_up] = new wdBayesNode(scheme);
					currentdtNode_.children[x_up].init(nc, m_ParamsPerAtt[u]);
				}

				currentdtNode_.children[x_up].setXYParameter(x_u, x_C, 0);
				currentdtNode_.children[x_up].incrementXYCount(x_u, x_C);

				currentdtNode_ = currentdtNode_.children[x_up];
			}
		}
	}

	/*
	 * -------------------------------------------------------------------------
	 * ---------------- Allocate Parameters
	 * -------------------------------------------------------------------------
	 * ----------------
	 */

	public void allocate() {
		// count active nodes in Trie
		np = nc;
		for (int u = 0; u < n; u++) {
			wdBayesNode pt = wdBayesNode_[u];
			activeNumNodes[u] = countActiveNumNodes(u, order[u], parents[u], pt);
		}
		System.out.println("Allocating dParameters of size: " + np);
		parameters = new double[np];
	}

	public int countActiveNumNodes(int i, int u, int[] lparents, wdBayesNode pt) {
		int numNodes = 0;
		int att = pt.att;

		if (att == -1) {
			pt.index = np;
			np += m_ParamsPerAtt[u] * nc;
			return 1;
		}

		while (att != -1) {
			int numChildren = pt.children.length;
			for (int c = 0; c < numChildren; c++) {
				wdBayesNode next = pt.children[c];
				if (next != null)
					numNodes += countActiveNumNodes(i, u, lparents, next);
				att = -1;
			}
		}

		return numNodes;
	}

	/*
	 * -------------------------------------------------------------------------
	 * ---------------- xyParameters to Parameters
	 * -------------------------------------------------------------------------
	 * ----------------
	 */

	public void reset() {
		// convert a trie into an array
		for (int u = 0; u < n; u++) {
			wdBayesNode pt = wdBayesNode_[u];
			trieToArray(u, order[u], parents[u], pt);
		}
	}

	private int trieToArray(int i, int u, int[] parents, wdBayesNode pt) {
		int att = pt.att;

		if (att == -1) {
			int index = pt.index;
			for (int j = 0; j < m_ParamsPerAtt[u]; j++) {
				for (int c = 0; c < nc; c++) {
					// System.out.println(index + (c * paramsPerAtt[u] + j));
					parameters[index + (c * m_ParamsPerAtt[u] + j)] = pt.getXYParameter(j, c);
				}
			}
			return 0;
		}

		while (att != -1) {
			int numChildren = pt.children.length;
			for (int c = 0; c < numChildren; c++) {
				wdBayesNode next = pt.children[c];
				if (next != null)
					trieToArray(i, u, parents, next);
				att = -1;
			}
		}

		return 0;
	}

	/* ----------------------------------------------------------------------------------
	// Parameters to xyParameters
	 ----------------------------------------------------------------------------------*/

	public void copyParameters(double[] params) {
		for (int i = 0; i < params.length; i++) {
			parameters[i] = params[i];
		}

		// convert an array into a trie
		for (int u = 0; u < n; u++) {
			wdBayesNode pt = wdBayesNode_[u];
			arrayToTrie(u, order[u], parents[u], pt);
		}
	}

	private int arrayToTrie(int i, int u, int[] parents, wdBayesNode pt) {
		int att = pt.att;

		if (att == -1) {
			int index = pt.index;
			for (int j = 0; j < m_ParamsPerAtt[u]; j++) {
				for (int c = 0; c < nc; c++) {
					double val = parameters[index + (c * m_ParamsPerAtt[u] + j)];
					pt.setXYParameter(j, c, val);
				}
			}
			return 0;
		}

		while (att != -1) {
			int numChildren = pt.children.length;
			for (int c = 0; c < numChildren; c++) {
				wdBayesNode next = pt.children[c];
				if (next != null)
					arrayToTrie(i, u, parents, next);
				att = -1;
			}
		}

		return 0;
	}

	/*
	 * -------------------------------------------------------------------------
	 * ---------------- Initialize Parameters
	 * -------------------------------------------------------------------------
	 * ----------------
	 */

	public void initializeParameters(int m_WeightingInitialization, double[] classProbs) {
		if (m_WeightingInitialization == -1) {
			for (int c = 0; c < nc; c++) {
				parameters[c] = classProbs[c];
			}
			initializeParametersWithProbs();
		} else {
			// Arrays.fill(parameters, m_WeightingInitialization);
			for (int c = 0; c < nc; c++) {
				parameters[c] = m_WeightingInitialization;
			}
			initializeParametersWithVal(m_WeightingInitialization);
		}

	}

	public void initializeParametersWithProbs() {
		for (int u = 0; u < n; u++) {
			wdBayesNode pt = wdBayesNode_[u];
			initWithProbs(u, order[u], parents[u], pt);
		}
	}

	public void initializeParametersWithVal(double initVal) {
		for (int u = 0; u < n; u++) {
			wdBayesNode pt = wdBayesNode_[u];
			initWithVal(u, order[u], parents[u], pt, initVal);
		}
	}

	private int initWithProbs(int i, int u, int[] parents, wdBayesNode pt) {
		int att = pt.att;

		if (att == -1) {
			int index = pt.index;
			for (int j = 0; j < m_ParamsPerAtt[u]; j++) {
				for (int c = 0; c < nc; c++) {
					double val = pt.getXYCount(j, c);
					pt.setXYParameter(j, c, val);
					parameters[index + (c * m_ParamsPerAtt[u] + j)] = pt.getXYParameter(j, c);
				}
			}
			return 0;
		}

		while (att != -1) {
			int numChildren = pt.children.length;
			for (int c = 0; c < numChildren; c++) {
				wdBayesNode next = pt.children[c];
				if (next != null)
					initWithProbs(i, u, parents, next);
				att = -1;
			}
		}

		return 0;
	}

	private int initWithVal(int i, int u, int[] parents, wdBayesNode pt, double initVal) {
		int att = pt.att;

		if (att == -1) {
			int index = pt.index;
			for (int j = 0; j < m_ParamsPerAtt[u]; j++) {
				for (int c = 0; c < nc; c++) {
					pt.setXYParameter(j, c, initVal);
					parameters[index + (c * m_ParamsPerAtt[u] + j)] = initVal;
				}
			}
			return 0;
		}

		while (att != -1) {
			int numChildren = pt.children.length;
			for (int c = 0; c < numChildren; c++) {
				wdBayesNode next = pt.children[c];
				if (next != null)
					initWithProbs(i, u, parents, next);
				att = -1;
			}
		}

		return 0;
	}

	/* ----------------------------------------------------------------------------------
	 Access Functions
	 ----------------------------------------------------------------------------------*/

	public double getClassParameter(int c) {
		return parameters[c];
	}

	public double getXYParameter(Instance instance, int c, int i, int u, int[] m_Parents) {

		wdBayesNode pt = wdBayesNode_[i];
		int att = pt.att;

		// find the appropriate leaf
		while (att != -1) {
			int v = (int) instance.value(att);
			wdBayesNode next = pt.children[v];
			if (next == null)
				break;
			pt = next;
			att = pt.att;
		}

		return pt.getXYParameter((int) instance.value(u), c);
	}

	public int getClassIndex(int k) {
		return k;
	}

	public int getXYParameterIndex(Instance instance, int c, int i, int u, int[] m_Parents) {

		wdBayesNode pt = wdBayesNode_[i];
		int att = pt.att;

		// find the appropriate leaf
		while (att != -1) {
			int v = (int) instance.value(att);
			wdBayesNode next = pt.children[v];
			if (next == null)
				break;
			pt = next;
			att = pt.att;
		}

		return pt.getXYIndex((int) instance.value(u), c);
	}

	public wdBayesNode getBayesNode(Instance instance, int i, int u, int[] m_Parents) {

		wdBayesNode pt = wdBayesNode_[i];
		int att = pt.att;

		// find the appropriate leaf
		while (att != -1) {
			int v = (int) instance.value(att);
			wdBayesNode next = pt.children[v];
			if (next == null)
				break;
			pt = next;
			att = pt.att;
		}

		return pt;
	}

	public wdBayesNode getBayesNode(Instance instance, int i) {

		wdBayesNode pt = wdBayesNode_[i];
		int att = pt.att;

		// find the appropriate leaf
		while (att != -1) {
			int v = (int) instance.value(att);
			wdBayesNode next = pt.children[v];
			if (next == null)
				break;
			pt = next;
			att = pt.att;
		}
		return pt;
	}

	public wdBayesNode[] getInstanceNode(Instance instance, int i) {

		wdBayesNode pt = wdBayesNode_[i];
		wdBayesNode[] nodes = new wdBayesNode[parents[i].length + 1];

		int att = pt.att;
		nodes[0] = pt;

		// find all the children nodes
		for (int k = 1; k < nodes.length; k++) {
			int v = (int) instance.value(att);
			wdBayesNode next = pt.children[v];

			if (next == null)
				break;
			nodes[k] = next;
			pt = next;
			att = pt.att;
		}
		nodes[nodes.length - 1] = getBayesNode(instance, i);
		return nodes;
	}

	// ----------------------------------------------------------------------------------
	// Others
	// ----------------------------------------------------------------------------------

	public double[] getParameters() {
		return parameters;
	}

	public int getNp() {
		return np;
	}

	public int getNAttributes() {
		return n;
	}

	public double getNLL_MAP(Instances instances, xyDist xyDist) {

		double nll = 0;
		int N = instances.numInstances();
		double mLogNC = -Math.log(nc);
		double[] myProbs = new double[nc];

		for (int i = 0; i < N; i++) {
			Instance instance = instances.instance(i);

			int x_C = (int) instance.classValue();

			// unboxed logDistributionForInstance_d(instance,nodes);
			for (int c = 0; c < nc; c++) {
				myProbs[c] = xyDist.pp(c);
			}
			for (int u = 0; u < n; u++) {
				wdBayesNode bNode = getBayesNode(instance, u);
				for (int c = 0; c < nc; c++) {
					myProbs[c] += bNode.getXYCount((int) instance.value(order[u]), c);
				}
			}
			SUtils.normalizeInLogDomain(myProbs);
			nll += (mLogNC - myProbs[x_C]);
			// nll += (- myProbs[x_C]);
		}

		return nll;
	}

	public double getNLL_dCCBN(Instances instances) {
		double nll = 0;
		int N = instances.numInstances();
		double mLogNC = -Math.log(nc);
		double[] myProbs = new double[nc];

		for (int i = 0; i < N; i++) {
			Instance instance = instances.instance(i);

			int x_C = (int) instance.classValue();

			// unboxed logDistributionForInstance_d(instance,nodes);
			for (int c = 0; c < nc; c++) {
				myProbs[c] = getClassParameter(c);
			}
			for (int u = 0; u < n; u++) {
				// wdBayesNode bNode = myNodes[u];
				wdBayesNode bNode = getBayesNode(instance, u);
				for (int c = 0; c < nc; c++) {
					myProbs[c] += bNode.getXYParameter((int) instance.value(order[u]), c);
				}
			}
			SUtils.normalizeInLogDomain(myProbs);
			nll += (mLogNC - myProbs[x_C]);
			// nll += (- myProbs[x_C]);
		}

		return nll;
	}

	public double getNLL_eCCBN(Instances instances) {

		double nll = 0;
		int N = instances.numInstances();
		double mLogNC = -Math.log(nc);
		double[] myProbs = new double[nc];

		for (int i = 0; i < N; i++) {
			Instance instance = instances.instance(i);

			int x_C = (int) instance.classValue();

			for (int c = 0; c < nc; c++) {
				myProbs[c] = getClassParameter(c);
			}
			for (int u = 0; u < n; u++) {
				wdBayesNode bNode = getBayesNode(instance, u);
				for (int c = 0; c < nc; c++) {
					myProbs[c] += bNode.getXYParameter((int) instance.value(order[u]), c);
				}
			}
			SUtils.normalizeInLogDomain(myProbs);
			nll += (mLogNC - myProbs[x_C]);
		}

		return nll;
	}

	public double getNLL_wCCBN(Instances instances, xyDist xyDist) {

		double nll = 0;
		int N = instances.numInstances();
		double mLogNC = -Math.log(nc);
		double[] myProbs = new double[nc];

		for (int i = 0; i < N; i++) {
			Instance instance = instances.instance(i);

			int x_C = (int) instance.classValue();

			// unboxed logDistributionForInstance_d(instance,nodes);
			for (int c = 0; c < nc; c++) {
				myProbs[c] = xyDist.pp(c) * getClassParameter(c);
			}
			for (int u = 0; u < n; u++) {
				wdBayesNode bNode = getBayesNode(instance, u);
				for (int c = 0; c < nc; c++) {
					myProbs[c] += bNode.getXYParameter((int) instance.value(order[u]), c)
							* bNode.getXYCount((int) instance.value(order[u]), c);
				}
			}
			SUtils.normalizeInLogDomain(myProbs);
			nll += (mLogNC - myProbs[x_C]);
			// nll += (- myProbs[x_C]);
		}

		return nll;
	}

	public void printProbabilities() {
		// System.out.println();
		//
		// for (int c = 0; c < nc; c++) {
		// //System.out.print(xxyDist_.xyDist_.getClassCount(c) + ", ");
		// }
		//
		// for (int u = 0; u < n; u++) {
		// for (int uval = 0; uval < m_ParamsPerAtt[u]; uval++) {
		// for (int c = 0; c < nc; c++) {
		// System.out.print("P(x_" + u + "=" + uval + " | y = " + c + ") = " +
		// xxyDist_.xyDist_.getCount(u, uval, c) + ", " );
		// }
		// System.out.println();
		// }
		// }
	}
}